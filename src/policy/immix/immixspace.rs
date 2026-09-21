use super::defrag::StatsForDefrag;
use super::line::*;
use super::{block::*, defrag::Defrag};
use crate::plan::VectorObjectQueue;
use crate::policy::gc_work::{TraceKind, DEFAULT_TRACE, TRACE_KIND_TRANSITIVE_PIN};
use crate::policy::sft::GCWorkerMutRef;
use crate::policy::sft::SFT;
use crate::policy::sft_map::SFTMap;
use crate::policy::space::{CommonSpace, Space};
use crate::util::alloc::allocator::AllocationOptions;
use crate::util::alloc::allocator::AllocatorContext;
use crate::util::constants::LOG_BYTES_IN_PAGE;
use crate::util::heap::chunk_map::*;
use crate::util::heap::BlockPageResource;
use crate::util::heap::PageResource;
use crate::util::linear_scan::{Region, RegionIterator};
use crate::util::metadata::log_bit::UnlogBitsOperation;
use crate::util::metadata::side_metadata::SideMetadataSpec;
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
use crate::util::metadata::{self, MetadataSpec};
use crate::util::object_enum::ObjectEnumerator;
use crate::util::object_forwarding;
use crate::util::{copy::*, epilogue, object_enum};
use crate::util::{Address, ObjectReference};
use crate::vm::*;
use crate::{
    plan::ObjectQueue,
    scheduler::{GCWork, GCWorkScheduler, GCWorker, WorkBucketStage},
    util::opaque_pointer::{VMThread, VMWorkerThread},
    MMTK,
};
use atomic::Ordering;
use std::sync::{atomic::AtomicBool, atomic::AtomicU8, atomic::AtomicUsize, Arc};

pub(crate) const TRACE_KIND_FAST: TraceKind = 0;
pub(crate) const TRACE_KIND_DEFRAG: TraceKind = 1;

// ── RC double-free / use-after-free block tracker (MMTK_RC_DEBUG) ──────────────────────────────
// Tracks the set of blocks currently believed to be on the free list. Freeing a block already in
// the set is a DOUBLE-FREE (the freelist would hand it out twice -> aliasing -> heap-address
// SIGSEGV under reallocation pressure, which is exactly the freeing-on / tight-heap crash class).
// Panics loudly with the block address so the offending sweep is pinpointed. Active only under
// MMTK_RC_DEBUG; zero cost otherwise.
static RC_FREE_BLOCKS: std::sync::Mutex<Option<std::collections::HashSet<usize>>> =
    std::sync::Mutex::new(None);

fn rc_debug_track_free(site: &str, block: Block) {
    if !crate::plan::lxr::rc::rc_debug_on() {
        return;
    }
    let addr = block.start().as_usize();
    let end = addr + crate::policy::immix::block::Block::BYTES;
    // Log the freed block's [start, end) range so a later heap-address SIGSEGV can be matched
    // against the blocks this pause freed (if the faulting address falls in a freed block's range,
    // the sweep freed the crashing block -> confirms the UAF + names the block + free site).
    if std::env::var_os("MMTK_RC_LOG_FREES").is_some() {
        eprintln!("[RC-FREE] {site}: block [{addr:#x}, {end:#x})");
    }
    let mut g = RC_FREE_BLOCKS.lock().unwrap();
    let set = g.get_or_insert_with(std::collections::HashSet::new);
    if !set.insert(addr) {
        panic!(
            "[RC-DOUBLE-FREE] {site}: block {:#x} freed while already on the free list \
             (double-free -> aliasing).",
            addr
        );
    }
}

fn rc_debug_track_alloc(block: Block) {
    if !crate::plan::lxr::rc::rc_debug_on() {
        return;
    }
    let addr = block.start().as_usize();
    let mut g = RC_FREE_BLOCKS.lock().unwrap();
    if let Some(set) = g.as_mut() {
        set.remove(&addr);
    }
}

pub struct ImmixSpace<VM: VMBinding> {
    common: CommonSpace<VM>,
    pr: BlockPageResource<VM, Block>,
    /// Allocation status for all chunks in immix space
    pub chunk_map: ChunkMap,
    /// Current line mark state
    pub line_mark_state: AtomicU8,
    /// Line mark state in previous GC
    line_unavail_state: AtomicU8,
    /// A list of all reusable blocks
    pub reusable_blocks: ReusableBlockPool,
    /// Live (allocated) blocks counted by the most recent sweep — reset when
    /// sweep tasks are generated, accumulated by SweepChunk packets. With
    /// `reusable_blocks.len()` this yields a post-sweep fragmentation metric:
    /// the fraction of live blocks that are only partially occupied.
    pub swept_live_blocks: std::sync::atomic::AtomicUsize,
    /// Defrag utilities
    pub(super) defrag: Defrag,
    /// COMPACT-ALL madvise scope: true from a compact-all GC's prepare until
    /// the next major's prepare — blocks freed in that window return their
    /// pages to the OS regardless of MMTK_RELEASE_FREED_PAGES.
    madvise_freed_this_gc: AtomicBool,
    /// Bytes of objects newly marked/forwarded by the CURRENT major marking
    /// epoch (reset at each major's prepare). The truthful live measure for
    /// the compaction law: post-sweep reserved pages cannot distinguish a
    /// dense heap from one whose 256B lines are pinned by interleaved small
    /// live objects (mature_mutation: 6.6MB live pinning 17-46MB), and every
    /// line/block statistic is equally blind. ~One relaxed add per marked
    /// object per major.
    major_live_bytes: AtomicUsize,
    /// How many lines have been consumed since last GC?
    lines_consumed: AtomicUsize,
    /// Object mark state
    mark_state: u8,
    /// Work packet scheduler
    scheduler: Arc<GCWorkScheduler<VM>>,
    /// Some settings for this space
    space_args: ImmixSpaceArgs,
    /// lxr: does this Immix space run the RC overlays? (= `constraints.rc_enabled`,
    /// captured at construction). `false` for every non-RC plan, so the overlays
    /// stay inert and the space is byte-identical to upstream Immix.
    pub rc_enabled: bool,
    /// LXR reference-counting helper (typed `RC_TABLE` accessor). Zero-sized
    /// (`PhantomData`); the read/trace overlays consult it only when `rc_enabled`.
    pub rc: crate::util::rc::RefCountHelper<VM>,
    /// lxr P2.D: true only at the end of a SATB cycle or a full GC, when the RC
    /// read-side (is_live / is_reachable) must additionally consult the mark bit +
    /// defrag-source + forwarding. The P3 LXR plan drives this flag from its
    /// scheduler; under the gate it is always `false` so the refinement never fires.
    pub is_end_of_satb_or_full_gc: bool,
    /// lxr P2.D: count of nursery blocks promoted in place this GC. Written only by
    /// the P3 nursery-promotion gc_work; init-only / inert today.
    pub in_place_promoted_nursery_blocks: AtomicUsize,
    // ── LXR (P3, additive) — RC block-allocation + lazy mature-sweep bookkeeping ──
    // All inert until the LXR plan runs (`rc_enabled` stays false). Vendored/adapted from
    // lxr-v0.32.0 immixspace.rs.
    /// Per-mutator-phase clean/reusable nursery block tracking + nursery sweep.
    pub block_allocation: crate::policy::immix::block_allocation::BlockAllocation<VM>,
    /// Mature blocks that *may* have gone fully dead after a batch of decrements (deduplicated by
    /// the per-block log bit). Drained into `SweepBlocksAfterDecs` packets by
    /// `schedule_rc_block_sweeping_tasks`.
    possibly_dead_mature_blocks: crossbeam::queue::SegQueue<(Block, bool)>,
    /// Clean blocks released this GC, by source (young / mature / lazily). Stats only.
    pub num_clean_blocks_released_young: AtomicUsize,
    pub num_clean_blocks_released_mature: AtomicUsize,
    pub num_clean_blocks_released_lazy: AtomicUsize,
    /// Bytes the RC copy-allocator handed out this GC (mature evac, deferred). Stats only.
    pub copy_alloc_bytes: AtomicUsize,
    /// Lines consumed by mutators reusing partially-free (recycled) blocks this phase. Drives
    /// `get_mutator_recycled_lines_in_pages`. Written by the reuse allocator path (deferred); 0 today.
    reused_lines_consumed: AtomicUsize,
}

/// Some arguments for Immix Space.
pub struct ImmixSpaceArgs {
    /// Whether this ImmixSpace instance contains both young and old objects.
    /// This affects the updating of valid-object bits.  If some lines or blocks of this ImmixSpace
    /// instance contain young objects, their VO bits need to be updated during this GC.  Currently
    /// only StickyImmix is affected.  GenImmix allocates young objects in a separete CopySpace
    /// nursery and its VO bits can be cleared in bulk.
    pub mixed_age: bool,
    /// Disable copying for this Immix space.
    pub never_move_objects: bool,
}

unsafe impl<VM: VMBinding> Sync for ImmixSpace<VM> {}

impl<VM: VMBinding> SFT for ImmixSpace<VM> {
    fn name(&self) -> &'static str {
        self.get_name()
    }

    fn get_forwarded_object(&self, object: ObjectReference) -> Option<ObjectReference> {
        // If we never move objects, look no further.
        if !self.is_movable() {
            return None;
        }

        if object_forwarding::is_forwarded::<VM>(object) {
            Some(object_forwarding::read_forwarding_pointer::<VM>(object))
        } else {
            None
        }
    }

    fn is_live(&self, object: ObjectReference) -> bool {
        // lxr P2.1/P2.E: under RC, liveness is "ref-count > 0 (or already forwarded)", not the
        // mark bit. At the end of a SATB cycle or a full GC, the read-side is refined to also
        // consult the mark bit + defrag-source + forwarding (the end-of-SATB/full-GC branch).
        // Gated, so every non-RC plan keeps the mark-bit semantics below byte-for-byte; and
        // is_end_of_satb_or_full_gc is always false until the P3 LXR plan drives it.
        if self.rc_enabled {
            if self.is_end_of_satb_or_full_gc {
                if self.is_marked(object) {
                    let block = Block::from_unaligned_address(object.to_raw_address());
                    if block.is_defrag_source() {
                        if object_forwarding::is_forwarded::<VM>(object) {
                            let forwarded =
                                object_forwarding::read_forwarding_pointer::<VM>(object);
                            return self.is_marked(forwarded) && self.rc.count(forwarded) > 0;
                        } else {
                            return false;
                        }
                    }
                    return self.rc.count(object) > 0;
                } else if object_forwarding::is_forwarded::<VM>(object) {
                    let forwarded = object_forwarding::read_forwarding_pointer::<VM>(object);
                    return self.is_marked(forwarded) && self.rc.count(forwarded) > 0;
                } else {
                    return false;
                }
            }
            return self.rc.count(object) > 0 || object_forwarding::is_forwarded::<VM>(object);
        }
        // If the mark bit is set, it is live.
        if self.is_marked(object) {
            return true;
        }

        // If we never move objects, look no further.
        if !self.is_movable() {
            return false;
        }

        // If the object is forwarded, it is live, too.
        object_forwarding::is_forwarded::<VM>(object)
    }

    fn is_reachable(&self, object: ObjectReference) -> bool {
        // lxr P2.E: under RC, reachability follows forwarding then requires both the mark bit
        // and a positive ref-count. Gated; for every non-RC plan we fall through to the SFT
        // default (delegate to is_live), preserving upstream behaviour byte-for-byte.
        if self.rc_enabled {
            if object_forwarding::is_forwarded::<VM>(object) {
                let forwarded = object_forwarding::read_forwarding_pointer::<VM>(object);
                return self.is_marked(forwarded) && self.rc.count(forwarded) > 0;
            }
            return self.is_marked(object) && self.rc.count(object) > 0;
        }
        self.is_live(object)
    }
    #[cfg(feature = "object_pinning")]
    fn pin_object(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.pin_object::<VM>(object)
    }
    #[cfg(feature = "object_pinning")]
    fn unpin_object(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.unpin_object::<VM>(object)
    }
    #[cfg(feature = "object_pinning")]
    fn is_object_pinned(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC.is_object_pinned::<VM>(object)
    }
    fn is_movable(&self) -> bool {
        !self.space_args.never_move_objects
    }

    #[cfg(feature = "sanity")]
    fn is_sane(&self) -> bool {
        true
    }
    fn initialize_object_metadata(&self, _object: ObjectReference) {
        #[cfg(feature = "vo_bit")]
        crate::util::metadata::vo_bit::set_vo_bit(_object);
    }
    #[cfg(feature = "is_mmtk_object")]
    fn is_mmtk_object(&self, addr: Address) -> Option<ObjectReference> {
        crate::util::metadata::vo_bit::is_vo_bit_set_for_addr(addr)
    }
    #[cfg(feature = "is_mmtk_object")]
    fn find_object_from_internal_pointer(
        &self,
        ptr: Address,
        max_search_bytes: usize,
    ) -> Option<ObjectReference> {
        // We don't need to search more than the max object size in the immix space.
        let search_bytes = usize::min(super::MAX_IMMIX_OBJECT_SIZE, max_search_bytes);
        crate::util::metadata::vo_bit::find_object_from_internal_pointer::<VM>(ptr, search_bytes)
    }
    fn sft_trace_object(
        &self,
        _queue: &mut VectorObjectQueue,
        _object: ObjectReference,
        _worker: GCWorkerMutRef,
    ) -> ObjectReference {
        panic!("We do not use SFT to trace objects for Immix. sft_trace_object() cannot be used.")
    }

    fn debug_print_object_info(&self, object: ObjectReference) {
        println!("marked  = {}", self.is_marked(object));
        println!(
            "line marked = {}",
            Line::from_unaligned_address(object.to_raw_address()).is_marked(self.mark_state)
        );
        println!(
            "block state = {:?}",
            Block::from_unaligned_address(object.to_raw_address()).get_state()
        );
        object_forwarding::debug_print_object_forwarding_info::<VM>(object);
        self.common.debug_print_object_global_info(object);
    }
}

impl<VM: VMBinding> Space<VM> for ImmixSpace<VM> {
    fn as_space(&self) -> &dyn Space<VM> {
        self
    }
    fn as_sft(&self) -> &(dyn SFT + Sync + 'static) {
        self
    }
    fn get_page_resource(&self) -> &dyn PageResource<VM> {
        &self.pr
    }
    fn maybe_get_page_resource_mut(&mut self) -> Option<&mut dyn PageResource<VM>> {
        Some(&mut self.pr)
    }
    fn common(&self) -> &CommonSpace<VM> {
        &self.common
    }
    fn initialize_sft(&self, sft_map: &mut dyn SFTMap) {
        self.common().initialize_sft(self.as_sft(), sft_map)
    }
    fn release_multiple_pages(&mut self, _start: Address) {
        panic!("immixspace only releases pages enmasse")
    }
    fn set_copy_for_sft_trace(&mut self, _semantics: Option<CopySemantics>) {
        panic!("We do not use SFT to trace objects for Immix. set_copy_context() cannot be used.")
    }

    fn enumerate_objects(&self, enumerator: &mut dyn ObjectEnumerator) {
        object_enum::enumerate_blocks_from_chunk_map::<Block>(enumerator, &self.chunk_map);
    }

    fn clear_side_log_bits(&self) {
        // Remove the following warning if we have a legitimate use case.
        warn!("ImmixSpace::clear_side_log_bits is single-treaded.  Consider clearing side metadata in per-chunk work packets.");

        let log_bit = VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.extract_side_spec();
        for chunk in self.chunk_map.all_chunks() {
            log_bit.bzero_metadata(chunk.start(), Chunk::BYTES);
        }
    }

    fn set_side_log_bits(&self) {
        // Remove the following warning if we have a legitimate use case.
        warn!("ImmixSpace::set_side_log_bits is single-treaded.  Consider setting side metadata in per-chunk work packets.");

        let log_bit = VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.extract_side_spec();
        for chunk in self.chunk_map.all_chunks() {
            log_bit.bset_metadata(chunk.start(), Chunk::BYTES);
        }
    }
}

impl<VM: VMBinding> crate::policy::gc_work::PolicyTraceObject<VM> for ImmixSpace<VM> {
    fn trace_object<Q: ObjectQueue, const KIND: TraceKind>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        copy: Option<CopySemantics>,
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        if KIND == TRACE_KIND_TRANSITIVE_PIN {
            self.trace_object_without_moving(queue, object)
        } else if KIND == TRACE_KIND_DEFRAG {
            if Block::containing(object).is_defrag_source() {
                debug_assert!(self.in_defrag());
                debug_assert!(
                    !crate::plan::is_nursery_gc(worker.mmtk.get_plan()),
                    "Calling PolicyTraceObject on Immix in nursery GC"
                );
                self.trace_object_with_opportunistic_copy(
                    queue,
                    object,
                    copy.unwrap(),
                    worker,
                    // This should not be nursery collection. Nursery collection does not use PolicyTraceObject.
                    false,
                )
            } else {
                self.trace_object_without_moving(queue, object)
            }
        } else if KIND == TRACE_KIND_FAST {
            self.trace_object_without_moving(queue, object)
        } else {
            unreachable!()
        }
    }

    fn post_scan_object(&self, object: ObjectReference) {
        if super::MARK_LINE_AT_SCAN_TIME && !super::BLOCK_ONLY {
            debug_assert!(self.in_space(object));
            self.mark_lines(object);
        }
    }

    #[allow(clippy::if_same_then_else)] // DEFAULT_TRACE needs a workaround which is documented below.
    fn may_move_objects<const KIND: TraceKind>() -> bool {
        if KIND == TRACE_KIND_DEFRAG {
            true
        } else if KIND == TRACE_KIND_FAST || KIND == TRACE_KIND_TRANSITIVE_PIN {
            false
        } else if KIND == DEFAULT_TRACE {
            // FIXME: This is hacky. When we do a default trace, this should be a nonmoving space.
            // The only exception is the nursery GC for sticky immix, for which, we use default trace.
            // This function is only used for PlanProcessEdges, and for sticky immix nursery GC, we use
            // GenNurseryProcessEdges. So it still works. But this is quite hacky anyway.
            // See https://github.com/mmtk/mmtk-core/issues/1314 for details.
            false
        } else {
            unreachable!()
        }
    }
}

impl<VM: VMBinding> ImmixSpace<VM> {
    #[allow(unused)]
    const UNMARKED_STATE: u8 = 0;
    const MARKED_STATE: u8 = 1;

    /// Get side metadata specs
    fn side_metadata_specs(rc_enabled: bool) -> Vec<SideMetadataSpec> {
        let mut meta = if super::BLOCK_ONLY {
            vec![
                MetadataSpec::OnSide(Block::DEFRAG_STATE_TABLE),
                MetadataSpec::OnSide(Block::MARK_TABLE),
                *VM::VMObjectModel::LOCAL_MARK_BIT_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC,
                #[cfg(feature = "object_pinning")]
                *VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC,
            ]
        } else {
            vec![
                MetadataSpec::OnSide(Line::MARK_TABLE),
                MetadataSpec::OnSide(Block::DEFRAG_STATE_TABLE),
                MetadataSpec::OnSide(Block::MARK_TABLE),
                *VM::VMObjectModel::LOCAL_MARK_BIT_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC,
                *VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC,
                #[cfg(feature = "object_pinning")]
                *VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC,
            ]
        };
        // lxr P2.0/P2.C: gated RC tables. The non-RC list above is left byte-identical;
        // the RC straddle-line table plus the per-block/per-line RC tables register only
        // inside this branch. No plan sets rc_enabled until the P3 LXR plan, so this
        // branch is inert today (defined-but-not-mapped for all non-LXR plans).
        if rc_enabled {
            meta.push(MetadataSpec::OnSide(crate::util::rc::RC_STRADDLE_LINES));
            meta.push(MetadataSpec::OnSide(Block::LOG_TABLE));
            meta.push(MetadataSpec::OnSide(Block::NURSERY_PROMOTION_STATE_TABLE));
            meta.push(MetadataSpec::OnSide(Block::PHASE_EPOCH));
            meta.push(MetadataSpec::OnSide(Line::IX_LINE_REUSE_COUNT));
        }
        metadata::extract_side_metadata(&meta)
    }

    pub fn new(
        args: crate::policy::space::PlanCreateSpaceArgs<VM>,
        mut space_args: ImmixSpaceArgs,
    ) -> Self {
        if args.unlog_traced_object {
            assert!(
                args.constraints.needs_log_bit,
                "Invalid args when the plan does not use log bit"
            );
        }

        // Make sure we override the space args if we force non moving Immix
        if cfg!(feature = "immix_non_moving") && !space_args.never_move_objects {
            info!(
                "Overriding never_moves_objects for Immix Space {}, as the immix_non_moving feature is set. Block size: 2^{}",
                args.name,
                Block::LOG_BYTES,
            );
            space_args.never_move_objects = true;
        }

        // validate features
        if super::BLOCK_ONLY {
            assert!(
                space_args.never_move_objects,
                "Block-only immix must not move objects"
            );
        }
        assert!(
            Block::LINES / 2 <= u8::MAX as usize - 2,
            "Number of lines in a block should not exceed BlockState::MARK_MARKED"
        );

        #[cfg(feature = "vo_bit")]
        vo_bit::helper::validate_config::<VM>();
        let vm_map = args.vm_map;
        let scheduler = args.scheduler.clone();
        let rc_enabled = args.constraints.rc_enabled;
        let common = CommonSpace::new(args.into_policy_args(
            true,
            false,
            Self::side_metadata_specs(rc_enabled),
        ));
        let space_index = common.descriptor.get_index();
        ImmixSpace {
            rc_enabled,
            rc: crate::util::rc::RefCountHelper::NEW,
            is_end_of_satb_or_full_gc: false,
            in_place_promoted_nursery_blocks: AtomicUsize::new(0),
            block_allocation: crate::policy::immix::block_allocation::BlockAllocation::new(),
            possibly_dead_mature_blocks: crossbeam::queue::SegQueue::new(),
            num_clean_blocks_released_young: AtomicUsize::new(0),
            num_clean_blocks_released_mature: AtomicUsize::new(0),
            num_clean_blocks_released_lazy: AtomicUsize::new(0),
            copy_alloc_bytes: AtomicUsize::new(0),
            reused_lines_consumed: AtomicUsize::new(0),
            pr: if common.vmrequest.is_discontiguous() {
                BlockPageResource::new_discontiguous(
                    Block::LOG_PAGES,
                    vm_map,
                    scheduler.num_workers(),
                )
            } else {
                BlockPageResource::new_contiguous(
                    Block::LOG_PAGES,
                    common.start,
                    common.extent,
                    vm_map,
                    scheduler.num_workers(),
                )
            },
            common,
            chunk_map: ChunkMap::new(space_index),
            line_mark_state: AtomicU8::new(Line::RESET_MARK_STATE),
            line_unavail_state: AtomicU8::new(Line::RESET_MARK_STATE),
            lines_consumed: AtomicUsize::new(0),
            reusable_blocks: ReusableBlockPool::new(scheduler.num_workers()),
            swept_live_blocks: std::sync::atomic::AtomicUsize::new(0),
            defrag: Defrag::default(),
            madvise_freed_this_gc: AtomicBool::new(false),
            major_live_bytes: AtomicUsize::new(0),
            // Set to the correct mark state when inititialized. We cannot rely on prepare to set it (prepare may get skipped in nursery GCs).
            mark_state: Self::MARKED_STATE,
            scheduler: scheduler.clone(),
            space_args,
        }
    }

    /// Flush the thread-local queues in BlockPageResource
    pub fn flush_page_resource(&self) {
        self.reusable_blocks.flush_all();
        #[cfg(target_pointer_width = "64")]
        self.pr.flush_all()
    }

    // ── LXR (P3, additive) — RC block-allocation + lazy mature-sweep support ──────
    // Inert until the LXR plan runs (`rc_enabled` stays false). Vendored/adapted from
    // lxr-v0.32.0 immixspace.rs. `block_allocation.rs` (a sibling module) reaches the private
    // `pr`/`defrag` fields through these `pub(super)`/`pub(crate)` accessors.

    /// The block page resource (private field). Exposed to the `block_allocation` sibling module
    /// for the bulk nursery release / reset.
    pub(super) fn block_page_resource(&self) -> &BlockPageResource<VM, Block> {
        &self.pr
    }

    /// Notify the defrag bookkeeping that a fresh clean block was handed out. Exposed for the
    /// `block_allocation` sibling module (the `defrag` field is `pub(super)`).
    pub(super) fn notify_new_clean_block(&self, copy: bool) {
        self.defrag.notify_new_clean_block(copy);
    }

    /// Pages worth of lines mutators consumed by reusing partially-free blocks this phase. Used by
    /// `block_allocation::total_young_allocation_in_bytes`. Currently 0 (the reuse-allocator path
    /// that bumps `reused_lines_consumed` is deferred).
    pub(crate) fn get_mutator_recycled_lines_in_pages(&self) -> usize {
        debug_assert!(self.rc_enabled);
        self.reused_lines_consumed.load(Ordering::Relaxed)
            >> (LOG_BYTES_IN_PAGE - Line::LOG_BYTES as u8)
    }

    /// Record a mature block that may have died after a batch of decrements. Deduplicated by the
    /// per-block log bit so a block is only swept once per epoch. Drained by
    /// `schedule_rc_block_sweeping_tasks`.
    ///
    /// Only blocks in a live (allocated) state are recorded: a nursery block is in
    /// `BlockState::Unallocated` (the RC nursery convention) and is reclaimed by the *nursery* sweep
    /// (`rc_sweep_nursery_blocks`), which `rc_sweep_mature` would refuse anyway (it returns early on
    /// `Unallocated`). Excluding nursery blocks here keeps the two sweeps' block sets disjoint, so a
    /// block is never a candidate for both free paths in the same pause.
    pub fn add_to_possibly_dead_mature_blocks(&self, block: Block, is_defrag_source: bool) {
        if block.get_state() == BlockState::Unallocated {
            return;
        }
        if block.log() {
            self.possibly_dead_mature_blocks
                .push((block, is_defrag_source));
        }
    }

    /// Drain `possibly_dead_mature_blocks` into per-worker `SweepBlocksAfterDecs` packets,
    /// prioritised into the always-open `Unconstrained` bucket. Vendored from lxr-v0.32.0.
    pub fn schedule_rc_block_sweeping_tasks(&self, counter: crate::LazySweepingJobsCounter) {
        let size = self.possibly_dead_mature_blocks.len();
        let num_bins = self.scheduler().num_workers();
        let bin_cap = size / num_bins + if size % num_bins == 0 { 0 } else { 1 };
        let mut bins = (0..num_bins)
            .map(|_| Vec::with_capacity(bin_cap))
            .collect::<Vec<Vec<(Block, bool)>>>();
        'out: for bin in bins.iter_mut() {
            for _ in 0..bin_cap {
                if let Some(block) = self.possibly_dead_mature_blocks.pop() {
                    bin.push(block);
                } else {
                    break 'out;
                }
            }
        }
        let packets: Vec<Box<dyn GCWork<VM>>> = bins
            .into_iter()
            .map(|blocks| {
                Box::new(super::rc_work::SweepBlocksAfterDecs::new(
                    blocks,
                    counter.clone(),
                )) as Box<dyn GCWork<VM>>
            })
            .collect();
        // Plain bulk_add (not bulk_add_prioritized): our base's Unconstrained bucket has no
        // prioritized queue (the reference's does) — bulk_add_prioritized unwraps None and aborts.
        self.scheduler().work_buckets[WorkBucketStage::Unconstrained].bulk_add(packets);
    }

    /// RC-pause prepare. Minimal cut: only the `Pause::RefCount` path is implemented (reset the
    /// per-GC block-release / copy counters). The `Pause::Full`/`InitialMark` branches (mature-evac
    /// selection, mark-state setup) are DEFERRED — they are never reached because the minimal LXR
    /// plan only ever schedules a `RefCount` pause. Vendored/adapted from lxr-v0.32.0 immixspace.rs.
    pub fn prepare_rc(&mut self, pause: crate::plan::lxr::Pause) {
        use crate::plan::lxr::Pause;
        debug_assert!(
            pause == Pause::RefCount || pause == Pause::Full,
            "minimal LXR cut only schedules RefCount + Full (backup-trace) pauses"
        );
        self.num_clean_blocks_released_young
            .store(0, Ordering::SeqCst);
        self.num_clean_blocks_released_mature
            .store(0, Ordering::SeqCst);
        self.num_clean_blocks_released_lazy
            .store(0, Ordering::SeqCst);
        self.copy_alloc_bytes.store(0, Ordering::SeqCst);
        // Full (backup trace): the mark bit is temporarily authoritative over RC for liveness, so
        // weak-ref / finalizer `is_live`/`is_reachable` queries during this pause AND the trace
        // result (consult is_marked) — `is_end_of_satb_or_full_gc` enables that read-side. Cleared
        // in `release_rc`.
        if pause == Pause::Full {
            self.is_end_of_satb_or_full_gc = true;
        }
    }

    /// Schedule the object-mark-bit zeroing tasks (Full pause prologue) so the backup trace marks
    /// into a clean table. Fans out one `ChunkMarkZeroing` per chunk batch into `Unconstrained`.
    pub fn schedule_mark_table_zeroing(&self) {
        let tasks = self.chunk_map.generate_tasks(|chunk| {
            Box::new(super::rc_work::ChunkMarkZeroing {
                chunks: chunk..chunk.next(),
            })
        });
        self.scheduler().work_buckets[WorkBucketStage::Unconstrained].bulk_add(tasks);
    }

    /// Schedule the dead-cycle sweep (Full pause epilogue): scans all mature blocks and reclaims
    /// objects that are `rc>0` but were NOT marked by the backup trace (dead cyclic garbage). Fans
    /// out one `SweepDeadCycles` per chunk batch into `Unconstrained` (runs in the post-decs epilogue).
    pub fn schedule_dead_cycle_sweep(&self) {
        let tasks = self.chunk_map.generate_tasks(|chunk| {
            Box::new(super::rc_work::SweepDeadCycles::<VM>::new(
                chunk..chunk.next(),
                crate::LazySweepingJobsCounter::new_decs(),
            )) as Box<dyn GCWork<VM>>
        });
        self.scheduler().work_buckets[WorkBucketStage::Unconstrained].bulk_add(tasks);
    }

    /// RC-pause release. Minimal cut (`Pause::RefCount`): sweep the unpromoted nursery blocks back
    /// to the page resource's free list, flush, reset the inc buffer + reused-line counter. The
    /// post-SATB mature sweeping and lazy-decrement draining are DEFERRED.
    pub fn release_rc(&mut self, pause: crate::plan::lxr::Pause) {
        use crate::plan::lxr::Pause;
        debug_assert!(pause == Pause::RefCount || pause == Pause::Full);
        if crate::plan::lxr::rc::rc_debug_on() {
            eprintln!(
                "[RC-PHASE] release_rc start: incs_total={} promoted={}",
                crate::plan::lxr::rc::RC_INCS_TOTAL.load(Ordering::Relaxed),
                crate::plan::lxr::rc::RC_INCS_PROMOTED.load(Ordering::Relaxed),
            );
        }
        // NOTE: the nursery sweep is NOT done here. It is deferred to the post-decrement epilogue
        // (`RCBlockSweepEpilogue`, after `STWRCDecsAndSweep` drains) so that NO decrement — neither
        // the prev-root decs nor the field-barrier decs (old overwritten values, which can point at
        // YOUNG objects) — can read an object whose nursery block we already freed. Freeing nursery
        // blocks here (in Release, before the decs) was a use-after-free: a dec would dereference a
        // dangling young object -> SIGSEGV at a heap address.
        self.flush_page_resource();
        self.rc.reset_inc_buffer_size();
        self.is_end_of_satb_or_full_gc = false;
        self.reused_lines_consumed.store(0, Ordering::Relaxed);
    }

    /// Sweep the unpromoted nursery blocks handed out this phase. Frees only blocks from the
    /// `block_allocation` per-phase list (NOT a chunk scan): each was explicitly recorded in
    /// `initialize_new_clean_block`, so it is provably a real, mapped, this-phase nursery
    /// allocation. The previous chunk-scan version faulted by touching blocks never handed out as
    /// nursery allocations (page-resource free-queue blocks, a mid-spawn domain's freshly-acquired
    /// allocator block whose per-block metadata is not in a sweepable state).
    ///
    /// `MMTK_RC_NO_NURSERY_SWEEP` / `MMTK_RC_NO_FREE` keep the list-drain accounting but skip the
    /// actual freeing (bisect knobs).
    pub(crate) fn rc_sweep_nursery_blocks(&self) {
        let no_free = std::env::var_os("MMTK_RC_NO_NURSERY_SWEEP").is_some()
            || std::env::var_os("MMTK_RC_NO_FREE").is_some();
        let released = if no_free {
            // Drain + reset the list/counters but do not free (so the bisect still measures).
            self.block_allocation.reset_nursery_counters();
            0
        } else {
            self.block_allocation.sweep_nursery_blocks()
        };
        if released != 0 {
            self.num_clean_blocks_released_young
                .fetch_add(released, Ordering::Relaxed);
        }
        if crate::plan::lxr::rc::rc_debug_on() {
            eprintln!(
                "[RC-SWEEP] nursery: {released} unpromoted nursery blocks freed{}",
                if no_free {
                    " SKIPPED (MMTK_RC_NO_NURSERY_SWEEP/NO_FREE)"
                } else {
                    ""
                }
            );
        }
    }

    /// Get the number of defrag headroom pages.
    pub fn defrag_headroom_pages(&self) -> usize {
        self.defrag.defrag_headroom_pages(self)
    }

    /// Check if current GC is a defrag GC.
    pub fn in_defrag(&self) -> bool {
        self.defrag.in_defrag()
    }

    /// check if the current GC should do defragmentation.
    pub fn decide_whether_to_defrag(
        &self,
        emergency_collection: bool,
        collect_whole_heap: bool,
        collection_attempts: usize,
        user_triggered_collection: bool,
        full_heap_system_gc: bool,
    ) -> bool {
        self.defrag.decide_whether_to_defrag(
            self.is_defrag_enabled(),
            emergency_collection,
            collect_whole_heap,
            collection_attempts,
            user_triggered_collection,
            self.reusable_blocks.len() == 0,
            full_heap_system_gc,
            *self.common.options.immix_always_defrag,
            self.rc_enabled,
        );
        self.defrag.in_defrag()
    }

    /// Get work packet scheduler
    fn scheduler(&self) -> &GCWorkScheduler<VM> {
        &self.scheduler
    }

    pub(crate) fn prepare(
        &mut self,
        major_gc: bool,
        plan_stats: Option<StatsForDefrag>,
        unlog_bits_op: UnlogBitsOperation,
    ) {
        // lxr P2.G: the P3 LXR plan drives a separate prepare_rc path; under our gate no plan
        // sets rc_enabled, so this mark-based prepare must never run for an RC space. The assert
        // documents that and can never fire (rc_enabled always false until P3).
        debug_assert!(!self.rc_enabled);
        if major_gc {
            // New major marking epoch: restart the live-bytes tally (see
            // `major_live_bytes` — the compaction law's denominator).
            self.major_live_bytes.store(0, Ordering::Relaxed);
            // Update mark_state
            if VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.is_on_side() {
                self.mark_state = Self::MARKED_STATE;
            } else {
                // For header metadata, we use cyclic mark bits.
                unimplemented!("cyclic mark bits is not supported at the moment");
            }

            // Prepare defrag info
            if self.is_defrag_enabled() {
                self.defrag.prepare(self, plan_stats.unwrap());
            }

            // Prepare each block for GC
            let threshold = self.defrag.defrag_spill_threshold.load(Ordering::Acquire);
            // # Safety: ImmixSpace reference is always valid within this collection cycle.
            let space = unsafe { &*(self as *const Self) };
            let compact_all = self.defrag.compact_all_active() && space.in_defrag();
            // Arm the compaction-epoch madvise scope (cleared by the next
            // major's prepare): covers this GC's sweep — in-pause or the
            // deferred quanta draining across later nursery pauses.
            self.madvise_freed_this_gc
                .store(compact_all, Ordering::Relaxed);
            let work_packets = self.chunk_map.generate_tasks(|chunk| {
                Box::new(PrepareBlockState {
                    space,
                    chunk,
                    defrag_threshold: if space.in_defrag() {
                        Some(threshold)
                    } else {
                        None
                    },
                    compact_all,
                    unlog_bits_op,
                })
            });
            self.scheduler().work_buckets[WorkBucketStage::Prepare].bulk_add(work_packets);

            if !super::BLOCK_ONLY {
                self.line_mark_state.fetch_add(1, Ordering::AcqRel);
                if self.line_mark_state.load(Ordering::Acquire) > Line::MAX_MARK_STATE {
                    self.line_mark_state
                        .store(Line::RESET_MARK_STATE, Ordering::Release);
                }
            }
        }

        #[cfg(feature = "vo_bit")]
        if vo_bit::helper::need_to_clear_vo_bits_before_tracing::<VM>() {
            let maybe_scope = if major_gc {
                // If it is major GC, we always clear all VO bits because we are doing full-heap
                // tracing.
                Some(VOBitsClearingScope::FullGC)
            } else if self.space_args.mixed_age {
                // StickyImmix nursery GC.
                // Some lines (or blocks) contain only young objects,
                // while other lines (or blocks) contain only old objects.
                if super::BLOCK_ONLY {
                    // Block only.  Young objects are only allocated into fully empty blocks.
                    // Only clear unmarked blocks.
                    Some(VOBitsClearingScope::BlockOnly)
                } else {
                    // Young objects are allocated into empty lines.
                    // Only clear unmarked lines.
                    let line_mark_state = self.line_mark_state.load(Ordering::SeqCst);
                    Some(VOBitsClearingScope::Line {
                        state: line_mark_state,
                    })
                }
            } else {
                // GenImmix nursery GC.  We do nothing to the ImmixSpace because the nursery is a
                // separate CopySpace.  It'll clear its own VO bits.
                None
            };

            if let Some(scope) = maybe_scope {
                let work_packets = self
                    .chunk_map
                    .generate_tasks(|chunk| Box::new(ClearVOBitsAfterPrepare { chunk, scope }));
                self.scheduler.work_buckets[WorkBucketStage::ClearVOBits].bulk_add(work_packets);
            }
        }
    }

    /// Release for the immix space.
    pub(crate) fn release(&mut self, major_gc: bool, unlog_bits_op: UnlogBitsOperation) {
        // lxr P2.G: the P3 LXR plan drives a separate release_rc path; under our gate no plan
        // sets rc_enabled, so this mark-based release must never run for an RC space. The assert
        // documents that and can never fire (rc_enabled always false until P3).
        debug_assert!(!self.rc_enabled);
        if major_gc {
            // Update line_unavail_state for hole searching after this GC.
            if !super::BLOCK_ONLY {
                self.line_unavail_state.store(
                    self.line_mark_state.load(Ordering::Acquire),
                    Ordering::Release,
                );
            }
        }
        // Clear reusable blocks list
        if !super::BLOCK_ONLY {
            self.reusable_blocks.reset();
        }
        // Sweep chunks and blocks
        let work_packets = self.generate_sweep_tasks(unlog_bits_op);
        self.scheduler().work_buckets[WorkBucketStage::Release].bulk_add(work_packets);

        self.lines_consumed.store(0, Ordering::Relaxed);
    }

    /// Like [`Self::release`], but RETURNS the chunk-sweep packets instead of
    /// scheduling them, so a plan can execute the sweep INCREMENTALLY (e.g.
    /// Bactrian's sweep quanta inside subsequent nursery pauses — stock
    /// OCaml's sweep slices). The caller owns correctness of the deferral:
    /// no state the packets read (line_mark_state, defrag histograms, chunk
    /// map) may be re-prepared until every packet has run, i.e. the next
    /// full/cycle prepare must be gated on drain completion. Blocks freed by
    /// a deferred packet flow to the page resource exactly as in the
    /// immediate path; the FlushPageResource epilogue fires when the LAST
    /// packet (whenever it runs) completes.
    pub(crate) fn release_deferred_sweep(
        &mut self,
        major_gc: bool,
        unlog_bits_op: UnlogBitsOperation,
    ) -> Vec<Box<dyn GCWork<VM>>> {
        debug_assert!(!self.rc_enabled);
        if major_gc && !super::BLOCK_ONLY {
            self.line_unavail_state.store(
                self.line_mark_state.load(Ordering::Acquire),
                Ordering::Release,
            );
        }
        if !super::BLOCK_ONLY {
            self.reusable_blocks.reset();
        }
        self.lines_consumed.store(0, Ordering::Relaxed);
        self.generate_sweep_tasks(unlog_bits_op)
    }

    /// This is called when a GC finished.
    /// Return whether this GC was a defrag GC, as a plan may want to know this.
    pub fn end_of_gc(&mut self) -> bool {
        let did_defrag = self.defrag.in_defrag();
        if self.is_defrag_enabled() {
            self.defrag.reset_in_defrag();
        }
        did_defrag
    }

    /// Post-sweep fragmentation: (partially-occupied live blocks, live blocks).
    /// Meaningful after the sweep (immediate or deferred) has fully drained.
    /// Request a one-shot COMPACT-ALL: the next defrag collection treats
    /// every in-use block as a defrag source (see `Defrag::compact_all_once`
    /// — the remedy for intra-line waste that hole-based selection cannot
    /// see). Also forces that collection to BE a defrag collection.
    pub fn request_compact_all(&self) {
        self.defrag.request_compact_all();
    }

    /// Bytes marked/forwarded by the last major marking epoch — the
    /// truthful mature live measure (see the `major_live_bytes` field).
    pub fn major_live_bytes(&self) -> usize {
        self.major_live_bytes.load(Ordering::Relaxed)
    }

    /// Add to the major live tally. UP window: plain read-add-store (a
    /// relaxed fetch_add is still a LOCK XADD, paid per marked object).
    fn add_major_live_bytes(&self, bytes: usize) {
        if crate::util::up_trace::up() {
            self.major_live_bytes.store(
                self.major_live_bytes.load(Ordering::Relaxed) + bytes,
                Ordering::Relaxed,
            );
        } else {
            self.major_live_bytes.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    pub fn post_sweep_fragmentation(&self) -> (usize, usize) {
        (
            self.reusable_blocks.len(),
            self.swept_live_blocks.load(Ordering::Relaxed),
        )
    }

    /// Generate chunk sweep tasks
    fn generate_sweep_tasks(&self, unlog_bits_op: UnlogBitsOperation) -> Vec<Box<dyn GCWork<VM>>> {
        self.swept_live_blocks.store(0, Ordering::Relaxed);
        self.defrag.mark_histograms.lock().clear();
        // # Safety: ImmixSpace reference is always valid within this collection cycle.
        let space = unsafe { &*(self as *const Self) };
        let epilogue = Arc::new(FlushPageResource {
            space,
            counter: AtomicUsize::new(0),
        });
        let tasks = self.chunk_map.generate_tasks(|chunk| {
            Box::new(SweepChunk {
                space,
                chunk,
                unlog_bits_op,
                epilogue: epilogue.clone(),
            })
        });
        epilogue.counter.store(tasks.len(), Ordering::SeqCst);
        tasks
    }

    /// Release a block.
    pub fn release_block(&self, block: Block) {
        rc_debug_track_free("release_block", block);
        // RC: `deinit_rc` also clears the in-place-promoted / owner / defrag-source per-block state.
        if self.rc_enabled {
            block.deinit_rc(self);
        } else {
            block.deinit();
        }
        // COMPACT-ALL epoch: blocks freed by a compaction's sweep return
        // their pages to the OS unconditionally — reclaiming residency is
        // the compaction's purpose (mature_mutation: reserved collapsed
        // 46→7MB but RSS stayed flat without this). Steady-state recycling
        // between majors keeps the fast path (MMTK_RELEASE_FREED_PAGES).
        self.pr
            .release_block_with(block, self.madvise_freed_this_gc.load(Ordering::Relaxed));
    }

    /// Push an already-deinitialised block back to the page resource free list (accounting +
    /// freelist). Used by the RC mature sweep (`Block::rc_sweep_mature`), which has already run
    /// `deinit_rc` under the per-block lock, so we must NOT deinit again here.
    pub(crate) fn release_block_to_free_list(&self, block: Block) {
        rc_debug_track_free("release_block_to_free_list", block);
        self.pr.release_block(block);
    }

    /// Allocate a clean block.
    pub fn get_clean_block(
        &self,
        tls: VMThread,
        copy: bool,
        alloc_options: AllocationOptions,
    ) -> Option<Block> {
        let block_address = self.acquire(tls, Block::PAGES, alloc_options);
        if block_address.is_zero() {
            return None;
        }
        let block = Block::from_aligned_address(block_address);
        rc_debug_track_alloc(block);
        if self.rc_enabled {
            // RC: route through block_allocation so the new block gets its RC tables (mark / field
            // unlog), nursery-block accounting, and `init_rc`. `cm_enabled = false` (CM deferred).
            // Ensure block_allocation's `&ImmixSpace` back-pointer is set: the plan-level lazy init
            // only runs at the first GC, but clean-block allocation happens earlier (mutator startup),
            // and `initialize_new_clean_block`/`self.space()` would deref a NULL space. `self` here is
            // the stable (post-boxing) space address; `init` just (idempotently) stores it.
            self.block_allocation.init(self);
            self.block_allocation
                .initialize_new_clean_block(block, copy, false);
        } else {
            self.defrag.notify_new_clean_block(copy);
            block.init(copy);
        }
        self.chunk_map.set_allocated(block.chunk(), true);
        self.lines_consumed
            .fetch_add(Block::LINES, Ordering::SeqCst);
        Some(block)
    }

    /// Pop a reusable block from the reusable block list.
    pub fn get_reusable_block(&self, copy: bool) -> Option<Block> {
        if super::BLOCK_ONLY {
            return None;
        }
        // Minimal RC cut: DISABLE partial-block (recycled-line) reuse — always allocate fresh clean
        // blocks. The RC reuse path (`init_rc(copy, reuse=true)` + `reused_lines_consumed` tracking +
        // the per-line RC reuse counter) is the most fragile part of the LXR allocator (phase-epoch
        // asserts, straddle-line bookkeeping on partially-live blocks); deferring it keeps the first
        // RC bring-up correct. Throughput cost only. Reusable blocks are still populated by the
        // standard sweep but never handed back out under RC.
        if self.rc_enabled {
            return None;
        }
        loop {
            if let Some(block) = self.reusable_blocks.pop() {
                // Skip blocks that should be evacuated.
                if copy && block.is_defrag_source() {
                    continue;
                }

                // Get available lines. Do this before block.init which will reset block state.
                let lines_delta = match block.get_state() {
                    BlockState::Reusable { unavailable_lines } => {
                        Block::LINES - unavailable_lines as usize
                    }
                    BlockState::Unmarked => Block::LINES,
                    _ => unreachable!("{:?} {:?}", block, block.get_state()),
                };
                self.lines_consumed.fetch_add(lines_delta, Ordering::SeqCst);

                block.init(copy);
                return Some(block);
            } else {
                return None;
            }
        }
    }

    /// Trace and mark objects without evacuation.
    pub fn trace_object_without_moving(
        &self,
        queue: &mut impl ObjectQueue,
        object: ObjectReference,
    ) -> ObjectReference {
        #[cfg(feature = "vo_bit")]
        vo_bit::helper::on_trace_object::<VM>(object);

        if self.attempt_mark(object, self.mark_state) {
            self.add_major_live_bytes(VM::VMObjectModel::get_current_size(object));
            // lxr P2.F: under RC, a straddle continuation line carries no RC entry of its own,
            // so a marked straddle object is short-circuited before line/block marking. Gated;
            // dead for all non-RC plans (rc_enabled always false until the P3 LXR plan).
            if self.rc_enabled {
                let line = Line::from_aligned_address(Line::align(object.to_raw_address()));
                if self.rc.is_straddle_line(line) {
                    return object;
                }
            }
            // Mark block and lines
            if !super::BLOCK_ONLY {
                if !super::MARK_LINE_AT_SCAN_TIME {
                    self.mark_lines(object);
                }
            } else {
                Block::containing(object).set_state(BlockState::Marked);
            }

            #[cfg(feature = "vo_bit")]
            vo_bit::helper::on_object_marked::<VM>(object);

            // Visit node
            queue.enqueue(object);
            self.unlog_object_if_needed(object);
            return object;
        }
        object
    }

    /// Trace object and do evacuation if required.
    #[allow(clippy::assertions_on_constants)]
    pub fn trace_object_with_opportunistic_copy(
        &self,
        queue: &mut impl ObjectQueue,
        object: ObjectReference,
        semantics: CopySemantics,
        worker: &mut GCWorker<VM>,
        nursery_collection: bool,
    ) -> ObjectReference {
        let copy_context = worker.get_copy_context_mut();
        debug_assert!(!super::BLOCK_ONLY);

        #[cfg(feature = "vo_bit")]
        vo_bit::helper::on_trace_object::<VM>(object);

        let forwarding_status = object_forwarding::attempt_to_forward::<VM>(object);
        if object_forwarding::state_is_forwarded_or_being_forwarded(forwarding_status) {
            // We lost the forwarding race as some other thread has set the forwarding word; wait
            // until the object has been forwarded by the winner. Note that the object may not
            // necessarily get forwarded since Immix opportunistically moves objects.
            #[allow(clippy::let_and_return)]
            let new_object =
                object_forwarding::spin_and_get_forwarded_object::<VM>(object, forwarding_status);
            #[cfg(debug_assertions)]
            {
                if new_object == object {
                    debug_assert!(
                        self.is_marked(object) || self.defrag.space_exhausted() || self.is_pinned(object),
                        "Forwarded object is the same as original object {} even though it should have been copied",
                        object,
                    );
                } else {
                    // new_object != object
                    debug_assert!(
                        !Block::containing(new_object).is_defrag_source(),
                        "Block {:?} containing forwarded object {} should not be a defragmentation source",
                        Block::containing(new_object),
                        new_object,
                    );
                }
            }
            new_object
        } else if self.is_marked(object) {
            // We won the forwarding race but the object is already marked so we clear the
            // forwarding status and return the unmoved object
            object_forwarding::clear_forwarding_bits::<VM>(object);
            object
        } else {
            // We won the forwarding race; actually forward and copy the object if it is not pinned
            // and we have sufficient space in our copy allocator
            let new_object = if self.is_pinned(object)
                || (!nursery_collection && self.defrag.space_exhausted())
            {
                if self.attempt_mark(object, self.mark_state) {
                    self.add_major_live_bytes(VM::VMObjectModel::get_current_size(object));
                }
                object_forwarding::clear_forwarding_bits::<VM>(object);
                Block::containing(object).set_state(BlockState::Marked);

                #[cfg(feature = "vo_bit")]
                vo_bit::helper::on_object_marked::<VM>(object);

                if !super::MARK_LINE_AT_SCAN_TIME {
                    self.mark_lines(object);
                }

                self.unlog_object_if_needed(object);

                object
            } else {
                // We are forwarding objects. When the copy allocator allocates the block, it should
                // mark the block. So we do not need to explicitly mark it here.

                object_forwarding::forward_object::<VM>(
                    object,
                    semantics,
                    copy_context,
                    |new_object| {
                        self.add_major_live_bytes(VM::VMObjectModel::get_current_size(new_object));
                        // post_copy should have set the unlog bit
                        // if `unlog_traced_object` is true.
                        debug_assert!(
                            !self.common.unlog_traced_object
                                || VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                                    .is_unlogged::<VM>(new_object, Ordering::Relaxed)
                        );
                        #[cfg(feature = "vo_bit")]
                        vo_bit::helper::on_object_forwarded::<VM>(new_object);
                    },
                )
            };
            debug_assert_eq!(
                Block::containing(new_object).get_state(),
                BlockState::Marked
            );

            queue.enqueue(new_object);
            debug_assert!(new_object.is_live());
            new_object
        }
    }

    fn unlog_object_if_needed(&self, object: ObjectReference) {
        if self.common.unlog_traced_object {
            // Make sure the side metadata for the line can fit into one byte. For smaller line size, we should
            // use `mark_as_unlogged` instead to mark the bit.
            const_assert!(
                Line::BYTES
                    >= (1
                        << (crate::util::constants::LOG_BITS_IN_BYTE
                            + crate::util::constants::LOG_MIN_OBJECT_SIZE))
            );
            const_assert_eq!(
                crate::vm::object_model::specs::VMGlobalLogBitSpec::LOG_NUM_BITS,
                0
            ); // We should put this to the addition, but type casting is not allowed in constant assertions.

            // Every immix line is 256 bytes, which is mapped to 4 bytes in the side metadata.
            // If we have one object in the line that is mature, we can assume all the objects in the line are mature objects.
            // So we can just mark the byte.
            VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                .mark_byte_as_unlogged::<VM>(object, Ordering::Relaxed);
        }
    }

    /// Mark all the lines that the given object spans.
    #[allow(clippy::assertions_on_constants)]
    pub fn mark_lines(&self, object: ObjectReference) {
        debug_assert!(!super::BLOCK_ONLY);
        // lxr P2.F: RC does not use the line mark-state sweep (lines are reclaimed by ref-count,
        // not tracing), so line marking is a no-op under RC. Gated; dead for all non-RC plans.
        if self.rc_enabled {
            return;
        }
        Line::mark_lines_for_object::<VM>(object, self.line_mark_state.load(Ordering::Acquire));
    }

    /// Atomically mark an object.
    fn attempt_mark(&self, object: ObjectReference, mark_state: u8) -> bool {
        // UP window (round 32): a single stopped-world tracer needs no CAS —
        // plain load/test/store. Major marking pays this per marked object
        // (a SeqCst load + SeqCst compare-exchange otherwise).
        if crate::util::up_trace::up() {
            let side = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.extract_side_spec();
            let addr = object.to_raw_address();
            let old: u8 = unsafe { side.load::<u8>(addr) };
            if old == mark_state {
                return false;
            }
            unsafe { side.store::<u8>(addr, mark_state) };
            return true;
        }
        loop {
            let old_value = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.load_atomic::<VM, u8>(
                object,
                None,
                Ordering::SeqCst,
            );
            if old_value == mark_state {
                return false;
            }

            if VM::VMObjectModel::LOCAL_MARK_BIT_SPEC
                .compare_exchange_metadata::<VM, u8>(
                    object,
                    old_value,
                    mark_state,
                    None,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                break;
            }
        }
        true
    }

    /// Check if an object is marked.
    fn is_marked_with(&self, object: ObjectReference, mark_state: u8) -> bool {
        let old_value = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.load_atomic::<VM, u8>(
            object,
            None,
            Ordering::SeqCst,
        );
        old_value == mark_state
    }

    pub(crate) fn is_marked(&self, object: ObjectReference) -> bool {
        self.is_marked_with(object, self.mark_state)
    }

    // ── LXR / RC mark helpers (P3.4) ──────────────────────────────────────────
    // The additive parallel of attempt_mark/unmark for the RC plan: LXR fixes the
    // mark state to 1 (0->1 / 1->0 fetch_update) rather than CAS-ing against a passed
    // mark_state, so the other plans keep the existing 2-arg `attempt_mark(object,
    // mark_state)` unchanged (byte-identical). Vendored from lxr-v0.32.0 immixspace.rs.
    // Used by the RC trace + cm.rs; `allow(dead_code)` until the LXR plan lands (P3.6).

    /// Atomically mark an object (0 -> 1). Returns true iff this call did the marking.
    #[allow(dead_code)]
    pub(crate) fn attempt_mark_rc(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_MARK_BIT_SPEC
            .fetch_update_metadata::<VM, u8, _>(object, Ordering::Relaxed, Ordering::Relaxed, |v| {
                if v != 0 {
                    None
                } else {
                    Some(1)
                }
            })
            .is_ok()
    }

    /// Atomically unmark an object (1 -> 0). Returns true iff this call did the unmarking.
    #[allow(dead_code)]
    pub(crate) fn unmark_rc(&self, object: ObjectReference) -> bool {
        VM::VMObjectModel::LOCAL_MARK_BIT_SPEC
            .fetch_update_metadata::<VM, u8, _>(object, Ordering::Relaxed, Ordering::Relaxed, |v| {
                if v != 1 {
                    None
                } else {
                    Some(0)
                }
            })
            .is_ok()
    }

    /// Check if an object is pinned.
    fn is_pinned(&self, _object: ObjectReference) -> bool {
        #[cfg(feature = "object_pinning")]
        return self.is_object_pinned(_object);

        #[cfg(not(feature = "object_pinning"))]
        false
    }

    /// Hole searching.
    ///
    /// Linearly scan lines in a block to search for the next
    /// hole, starting from the given line. If we find available lines,
    /// return a tuple of the start line and the end line (non-inclusive).
    ///
    /// Returns None if the search could not find any more holes.
    #[allow(clippy::assertions_on_constants)]
    pub fn get_next_available_lines(&self, search_start: Line) -> Option<(Line, Line)> {
        debug_assert!(!super::BLOCK_ONLY);
        let unavail_state = self.line_unavail_state.load(Ordering::Acquire);
        let current_state = self.line_mark_state.load(Ordering::Acquire);
        let block = search_start.block();
        let mark_data = block.line_mark_table();
        let start_cursor = search_start.get_index_within_block();
        let mut cursor = start_cursor;
        // Find start
        while cursor < mark_data.len() {
            let mark = mark_data.get(cursor);
            if mark != unavail_state && mark != current_state {
                break;
            }
            cursor += 1;
        }
        if cursor == mark_data.len() {
            return None;
        }
        let start = search_start.next_nth(cursor - start_cursor);
        // Find limit
        while cursor < mark_data.len() {
            let mark = mark_data.get(cursor);
            if mark == unavail_state || mark == current_state {
                break;
            }
            cursor += 1;
        }
        let end = search_start.next_nth(cursor - start_cursor);
        debug_assert!(RegionIterator::<Line>::new(start, end)
            .all(|line| !line.is_marked(unavail_state) && !line.is_marked(current_state)));
        Some((start, end))
    }

    pub fn is_last_gc_exhaustive(&self, did_defrag_for_last_gc: bool) -> bool {
        if self.is_defrag_enabled() {
            did_defrag_for_last_gc
        } else {
            // If defrag is disabled, every GC is exhaustive.
            true
        }
    }

    pub(crate) fn get_pages_allocated(&self) -> usize {
        self.lines_consumed.load(Ordering::SeqCst) >> (LOG_BYTES_IN_PAGE - Line::LOG_BYTES as u8)
    }

    /// Post copy routine for Immix copy contexts
    fn post_copy(&self, object: ObjectReference, _bytes: usize) {
        // lxr P2.F: under RC, RC metadata travels with the copy in the trace path (P2.3) and
        // the mark-bit / line-mark post-copy fixups below do not apply, so this is a no-op.
        // Gated; dead for all non-RC plans (rc_enabled always false until the P3 LXR plan).
        if self.rc_enabled {
            return;
        }
        // Mark the object. UP window (round 32): one tracer, world stopped —
        // the SeqCst store compiles to a full-fence XCHG on x86, paid per
        // promoted object (binarytrees: 19.7M of them). A plain side-table
        // store is sound under the single-tracer invariant (same argument as
        // UP-trace); the pause's release lock sequences publish it before
        // any mutator or second worker runs.
        if crate::util::up_trace::up() {
            let side = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.extract_side_spec();
            unsafe { side.store::<u8>(object.to_raw_address(), self.mark_state) };
        } else {
            VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.store_atomic::<VM, u8>(
                object,
                self.mark_state,
                None,
                Ordering::SeqCst,
            );
        }
        // Mark the line
        if !super::MARK_LINE_AT_SCAN_TIME {
            self.mark_lines(object);
        }
        if self.common.unlog_traced_object {
            VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                .mark_byte_as_unlogged::<VM>(object, Ordering::Relaxed);
        }
    }

    pub(crate) fn prefer_copy_on_nursery_gc(&self) -> bool {
        self.is_nursery_copy_enabled()
    }

    pub(crate) fn is_nursery_copy_enabled(&self) -> bool {
        !self.space_args.never_move_objects && !cfg!(feature = "sticky_immix_non_moving_nursery")
    }

    pub(crate) fn is_defrag_enabled(&self) -> bool {
        !self.space_args.never_move_objects
    }
}

/// A work packet to prepare each block for a major GC.
/// Performs the action on a range of chunks.
pub struct PrepareBlockState<VM: VMBinding> {
    #[allow(dead_code)]
    pub space: &'static ImmixSpace<VM>,
    pub chunk: Chunk,
    pub defrag_threshold: Option<usize>,
    /// COMPACT-ALL (one-shot, see `Defrag::compact_all_once`): every in-use
    /// block is a defrag source this GC, regardless of hole count — the
    /// hole-bucket selection cannot see intra-line waste.
    pub compact_all: bool,
    pub unlog_bits_op: UnlogBitsOperation,
}

impl<VM: VMBinding> PrepareBlockState<VM> {
    /// Clear object mark table
    fn reset_object_mark(&self) {
        // NOTE: We reset the mark bits because cyclic mark bit is currently not supported, yet.
        // See `ImmixSpace::prepare`.
        if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_MARK_BIT_SPEC {
            side.bzero_metadata(self.chunk.start(), Chunk::BYTES);
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for PrepareBlockState<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        // Clear object mark table for this chunk
        self.reset_object_mark();
        // Iterate over all blocks in this chunk
        for block in self.chunk.iter_region::<Block>() {
            let state = block.get_state();
            // Skip unallocated blocks.
            if state == BlockState::Unallocated {
                continue;
            }
            // Check if this block needs to be defragmented.
            let is_defrag_source = if !self.space.is_defrag_enabled() {
                // Do not set any block as defrag source if defrag is disabled.
                false
            } else if *mmtk.options.immix_defrag_every_block || self.compact_all {
                // Set every block as defrag source if so desired.
                true
            } else if let Some(defrag_threshold) = self.defrag_threshold {
                // This GC is a defrag GC.
                block.get_holes() > defrag_threshold
            } else {
                // Not a defrag GC.
                false
            };
            block.set_as_defrag_source(is_defrag_source);
            // Clear block mark data.
            block.set_state(BlockState::Unmarked);
            debug_assert!(!block.get_state().is_reusable());
            debug_assert_ne!(block.get_state(), BlockState::Marked);
        }

        self.unlog_bits_op
            .execute::<VM>(self.chunk.start(), Chunk::BYTES);
    }
}

/// Chunk sweeping work packet.
struct SweepChunk<VM: VMBinding> {
    space: &'static ImmixSpace<VM>,
    chunk: Chunk,
    unlog_bits_op: UnlogBitsOperation,
    /// A destructor invoked when all `SweepChunk` packets are finished.
    epilogue: Arc<FlushPageResource<VM>>,
}

impl<VM: VMBinding> GCWork<VM> for SweepChunk<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        assert!(self.space.chunk_map.get(self.chunk).unwrap().is_allocated());

        let mut histogram = self.space.defrag.new_histogram();
        let line_mark_state = if super::BLOCK_ONLY {
            None
        } else {
            Some(self.space.line_mark_state.load(Ordering::Acquire))
        };
        // Hints for clearing side forwarding bits.
        let is_moving_gc = mmtk.get_plan().current_gc_may_move_object();
        let is_defrag_gc = self.space.defrag.in_defrag();
        // number of allocated blocks.
        let mut allocated_blocks = 0;
        // Iterate over all allocated blocks in this chunk.
        for block in self
            .chunk
            .iter_region::<Block>()
            .filter(|block| block.get_state() != BlockState::Unallocated)
        {
            // Clear side forwarding bits.
            // In the beginning of the next GC, no side forwarding bits shall be set.
            // In this way, we can omit clearing forwarding bits when copying object.
            // See `GCWorkerCopyContext::post_copy`.
            // Note, `block.sweep()` overwrites `DEFRAG_STATE_TABLE` with the number of holes,
            // but we need it to know if a block is a defrag source.
            // We clear forwarding bits before `block.sweep()`.
            if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC {
                if is_moving_gc {
                    let objects_may_move = if is_defrag_gc {
                        // If it is a defrag GC, we only clear forwarding bits for defrag sources.
                        block.is_defrag_source()
                    } else {
                        // Otherwise, it must be a nursery GC of StickyImmix with copying nursery.
                        // We don't have information about which block contains moved objects,
                        // so we have to clear forwarding bits for all blocks.
                        true
                    };
                    if objects_may_move {
                        side.bzero_metadata(block.start(), Block::BYTES);
                    }
                }
            }

            if !block.sweep(self.space, &mut histogram, line_mark_state) {
                // Block is live. Increment the allocated block count.
                allocated_blocks += 1;
            }
        }
        probe!(mmtk, sweep_chunk, allocated_blocks);
        // Accumulate the live-block count for the post-sweep fragmentation
        // metric (see `post_sweep_fragmentation`).
        self.space
            .swept_live_blocks
            .fetch_add(allocated_blocks, Ordering::Relaxed);
        // Set this chunk as free if there is not live blocks.
        if allocated_blocks == 0 {
            self.space.chunk_map.set_allocated(self.chunk, false)
        }
        self.space.defrag.add_completed_mark_histogram(histogram);

        self.unlog_bits_op
            .execute::<VM>(self.chunk.start(), Chunk::BYTES);

        self.epilogue.finish_one_work_packet();
    }
}

/// Count number of remaining work pacets, and flush page resource if all packets are finished.
struct FlushPageResource<VM: VMBinding> {
    space: &'static ImmixSpace<VM>,
    counter: AtomicUsize,
}

impl<VM: VMBinding> FlushPageResource<VM> {
    /// Called after a related work packet is finished.
    fn finish_one_work_packet(&self) {
        if 1 == self.counter.fetch_sub(1, Ordering::SeqCst) {
            // We've finished releasing all the dead blocks to the BlockPageResource's thread-local queues.
            // Now flush the BlockPageResource.
            self.space.flush_page_resource()
        }
    }
}

impl<VM: VMBinding> Drop for FlushPageResource<VM> {
    fn drop(&mut self) {
        epilogue::debug_assert_counter_zero(&self.counter, "FlushPageResource::counter");
    }
}

use crate::policy::copy_context::PolicyCopyContext;
use crate::util::alloc::Allocator;
use crate::util::alloc::ImmixAllocator;

/// Normal immix copy context. It has one copying Immix allocator.
/// Most immix plans use this copy context.
pub struct ImmixCopyContext<VM: VMBinding> {
    allocator: ImmixAllocator<VM>,
}

impl<VM: VMBinding> PolicyCopyContext for ImmixCopyContext<VM> {
    type VM = VM;

    fn prepare(&mut self) {
        self.allocator.reset();
    }
    fn release(&mut self) {
        self.allocator.reset();
    }
    fn alloc_copy(
        &mut self,
        _original: ObjectReference,
        bytes: usize,
        align: usize,
        offset: usize,
    ) -> Address {
        self.allocator.alloc(bytes, align, offset)
    }
    fn post_copy(&mut self, obj: ObjectReference, bytes: usize) {
        self.get_space().post_copy(obj, bytes)
    }
}

impl<VM: VMBinding> ImmixCopyContext<VM> {
    pub(crate) fn new(
        tls: VMWorkerThread,
        context: Arc<AllocatorContext<VM>>,
        space: &'static ImmixSpace<VM>,
    ) -> Self {
        ImmixCopyContext {
            allocator: ImmixAllocator::new(tls.0, Some(space), context, true),
        }
    }

    fn get_space(&self) -> &ImmixSpace<VM> {
        self.allocator.immix_space()
    }
}

/// Hybrid Immix copy context. It includes two different immix allocators. One with `copy = true`
/// is used for defrag GCs, and the other is used for other purposes (such as promoting objects from
/// nursery to Immix mature space). This is used by generational immix.
pub struct ImmixHybridCopyContext<VM: VMBinding> {
    copy_allocator: ImmixAllocator<VM>,
    defrag_allocator: ImmixAllocator<VM>,
}

impl<VM: VMBinding> PolicyCopyContext for ImmixHybridCopyContext<VM> {
    type VM = VM;

    fn prepare(&mut self) {
        self.copy_allocator.reset();
        self.defrag_allocator.reset();
    }
    fn release(&mut self) {
        self.copy_allocator.reset();
        self.defrag_allocator.reset();
    }
    fn alloc_copy(
        &mut self,
        _original: ObjectReference,
        bytes: usize,
        align: usize,
        offset: usize,
    ) -> Address {
        if self.get_space().in_defrag() {
            self.defrag_allocator.alloc(bytes, align, offset)
        } else {
            self.copy_allocator.alloc(bytes, align, offset)
        }
    }
    fn post_copy(&mut self, obj: ObjectReference, bytes: usize) {
        self.get_space().post_copy(obj, bytes)
    }
}

impl<VM: VMBinding> ImmixHybridCopyContext<VM> {
    pub(crate) fn new(
        tls: VMWorkerThread,
        context: Arc<AllocatorContext<VM>>,
        space: &'static ImmixSpace<VM>,
    ) -> Self {
        ImmixHybridCopyContext {
            copy_allocator: ImmixAllocator::new(tls.0, Some(space), context.clone(), false),
            defrag_allocator: ImmixAllocator::new(tls.0, Some(space), context, true),
        }
    }

    fn get_space(&self) -> &ImmixSpace<VM> {
        // Both copy allocators should point to the same space.
        debug_assert_eq!(
            self.defrag_allocator.immix_space().common().descriptor,
            self.copy_allocator.immix_space().common().descriptor
        );
        // Just get the space from either allocator
        self.defrag_allocator.immix_space()
    }
}

#[cfg(feature = "vo_bit")]
#[derive(Clone, Copy)]
enum VOBitsClearingScope {
    /// Clear all VO bits in all blocks.
    FullGC,
    /// Clear unmarked blocks, only.
    BlockOnly,
    /// Clear unmarked lines, only.  (i.e. lines with line mark state **not** equal to `state`).
    Line { state: u8 },
}

/// A work packet to clear VO bit metadata after Prepare.
#[cfg(feature = "vo_bit")]
struct ClearVOBitsAfterPrepare {
    chunk: Chunk,
    scope: VOBitsClearingScope,
}

#[cfg(feature = "vo_bit")]
impl<VM: VMBinding> GCWork<VM> for ClearVOBitsAfterPrepare {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, _mmtk: &'static MMTK<VM>) {
        match self.scope {
            VOBitsClearingScope::FullGC => {
                vo_bit::bzero_vo_bit(self.chunk.start(), Chunk::BYTES);
            }
            VOBitsClearingScope::BlockOnly => {
                self.clear_blocks(None);
            }
            VOBitsClearingScope::Line { state } => {
                self.clear_blocks(Some(state));
            }
        }
    }
}

#[cfg(feature = "vo_bit")]
impl ClearVOBitsAfterPrepare {
    fn clear_blocks(&mut self, line_mark_state: Option<u8>) {
        for block in self
            .chunk
            .iter_region::<Block>()
            .filter(|block| block.get_state() != BlockState::Unallocated)
        {
            block.clear_vo_bits_for_unmarked_regions(line_mark_state);
        }
    }
}
