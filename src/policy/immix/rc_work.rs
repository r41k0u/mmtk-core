//! LXR lazy mature-block sweeping work packet (P3, additive).
//!
//! Vendored and adapted from the LXR research fork's `policy/immix/rc_work.rs`. Only the minimal
//! single-domain RC subset is ported: `SweepBlocksAfterDecs`, the packet that sweeps mature blocks
//! flagged as possibly-dead by a batch of decrements. The reference's defrag/mature-evac selection
//! (`SelectDefragBlocks`, `MatureEvacuationSet`), the dead-cycle sweeper (`SweepDeadCycles`), and
//! the concurrent mark-table zeroing (`ConcurrentChunkMetadataZeroing`, `PrepareChunksForFullGC`)
//! are **deferred** (CM / mature evac are out of the minimal cut). Inert until the LXR plan runs
//! (`rc_enabled` stays false).
//!
//! ## Adaptation notes (LXR `lxr/lxr-v0.32.0`  vs  our base)
//!
//! * `rc_sweep_mature` here is the 3-arg `rc_sweep_mature::<VM>(space, defrag, rc_dead)` we added to
//!   `Block`.
//! * The reference also bumps `num_clean_blocks_released_*` stats inside a `current_pause().is_none()
//!   || STWRCDecsAndSweep.is_open()` gate; kept verbatim (the stats fields exist on `ImmixSpace`).

use std::ops::Range;

use atomic::Ordering;

use crate::{
    scheduler::{GCWork, GCWorker, WorkBucketStage},
    util::{heap::chunk_map::Chunk, linear_scan::Region, rc, ObjectReference},
    vm::{ObjectModel, VMBinding},
    LazySweepingJobsCounter, MMTK,
};

use super::block::{Block, BlockState};
use super::line::Line;
use crate::plan::lxr::LXR;

/// Sweep the mature blocks that a batch of decrements flagged as possibly-dead. Any block whose RC
/// table is now all-zero is deinitialised and its pages bulk-released.
pub(crate) struct SweepBlocksAfterDecs {
    blocks: Vec<(Block, bool)>,
    _counter: LazySweepingJobsCounter,
}

impl SweepBlocksAfterDecs {
    pub fn new(blocks: Vec<(Block, bool)>, counter: LazySweepingJobsCounter) -> Self {
        Self {
            blocks,
            _counter: counter,
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for SweepBlocksAfterDecs {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        if self.blocks.is_empty() {
            return;
        }
        let mut count = 0;
        for (block, defrag) in &self.blocks {
            block.unlog();
            if block.rc_sweep_mature::<VM>(&lxr.immix_space, *defrag, false) {
                count += 1;
            } else {
                assert!(
                    !*defrag,
                    "defrag block is freed? {:?} {:?} {}",
                    block,
                    block.get_state(),
                    block.is_defrag_source()
                );
            }
        }
        // NOTE: `rc_sweep_mature` now pushes each freed block back to the page-resource free list
        // itself (via `release_block_to_free_list`), so we do NOT call `bulk_release_blocks` here —
        // that was an accounting-only release that leaked the blocks under our free-list PR.
        if count != 0
            && (lxr.current_pause().is_none()
                || mmtk.scheduler.work_buckets[WorkBucketStage::STWRCDecsAndSweep].is_open())
        {
            lxr.immix_space
                .num_clean_blocks_released_mature
                .fetch_add(count, Ordering::Relaxed);
            lxr.immix_space
                .num_clean_blocks_released_lazy
                .fetch_add(count, Ordering::Relaxed);
        }
    }
}

// ── Backup-trace (cycle collector) work packets (P5, additive) ─────────────────────────────────
// Vendored + adapted from lxr-v0.32.0 rc_work.rs (`ConcurrentChunkMetadataZeroing`,
// `SweepDeadCycles`). These power the periodic STOP-THE-WORLD backup mark/sweep that reclaims the
// CYCLIC garbage pure RC cannot (a dead cycle's members keep each other's RC > 0 forever). Only the
// Full (`Pause::Full`) pause schedules them; the steady-state RC pause never touches them.

/// Zero the object MARK-BIT side-metadata across a range of chunks, scheduled at the START of a
/// Full pause so the backup trace marks into a clean table. (Reference: `ConcurrentChunkMetadataZeroing`.)
pub(crate) struct ChunkMarkZeroing {
    pub chunks: Range<Chunk>,
}

impl ChunkMarkZeroing {
    #[inline]
    fn reset_object_mark<VM: VMBinding>(chunk: Chunk) {
        VM::VMObjectModel::LOCAL_MARK_BIT_SPEC
            .extract_side_spec()
            .bzero_metadata(chunk.start(), Chunk::BYTES);
    }
}

impl<VM: VMBinding> GCWork<VM> for ChunkMarkZeroing {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let ix = &mmtk
            .get_plan()
            .downcast_ref::<LXR<VM>>()
            .unwrap()
            .immix_space;
        let num_chunks = (self.chunks.end.start() - self.chunks.start.start()) >> Chunk::LOG_BYTES;
        for i in 0..num_chunks {
            let chunk = self.chunks.start.next_nth(i);
            if !ix.chunk_map.get(chunk).is_some() {
                continue;
            }
            Self::reset_object_mark::<VM>(chunk);
        }
    }
}

/// The DEAD-CYCLE sweep: after the backup trace has set the mark bit on every reachable object,
/// scan each mature block; an object with `rc.count != 0` but NOT marked is **dead cyclic garbage**
/// (it had references — RC > 0 — only because of an unreachable cycle the trace did not visit).
/// Reclaim it (`rc.set(o, 0)` + unmark its straddle lines), and free any block left with no live
/// object. (Reference: `SweepDeadCycles`.)
///
/// ADAPTATION: dropped the concurrent-marking `SeqCst` fence + RC re-check (no concurrent decs in the
/// STW cut) and the defrag-source / mature-evac handling (no copying). `to_object_reference` →
/// `ObjectReference::from_raw_address_unchecked`; `Line::from`/`is_aligned` → `Line::of` /
/// `is_aligned_to(Line::BYTES)`.
pub(crate) struct SweepDeadCycles<VM: VMBinding> {
    chunks: Range<Chunk>,
    _counter: LazySweepingJobsCounter,
    rc: rc::RefCountHelper<VM>,
}

impl<VM: VMBinding> SweepDeadCycles<VM> {
    pub fn new(chunks: Range<Chunk>, counter: LazySweepingJobsCounter) -> Self {
        Self {
            chunks,
            _counter: counter,
            rc: rc::RefCountHelper::NEW,
        }
    }

    fn process_dead_object(&mut self, o: ObjectReference) {
        if !crate::args::BLOCK_ONLY {
            self.rc.unmark_straddle_object(o);
        }
        self.rc.set(o, 0);
    }

    /// Returns true iff the block has NO live (rc != 0 && marked) object — i.e. it can be freed.
    fn process_block(&mut self, block: Block, immix_space: &super::ImmixSpace<VM>) -> bool {
        let mut has_live = false;
        let mut cursor = block.start();
        let limit = block.end();
        while cursor < limit {
            let o = unsafe { ObjectReference::from_raw_address_unchecked(cursor) };
            cursor += rc::MIN_OBJECT_SIZE;
            let c = self.rc.count(o);
            if c != 0 && !immix_space.is_marked(o) {
                // rc>0 but unreachable => dead cyclic garbage. Skip straddle CONTINUATION cells
                // (a >1-line object's continuation lines carry an rc==1 straddle marker, not a real
                // object header): only the object START is a real object.
                if !crate::args::BLOCK_ONLY && o.to_raw_address().is_aligned_to(Line::BYTES) {
                    if c == 1 && self.rc.is_straddle_line(Line::of(o.to_raw_address())) {
                        continue;
                    }
                }
                self.process_dead_object(o);
            } else if c != 0 {
                has_live = true;
            }
        }
        !has_live
    }
}

impl<VM: VMBinding> GCWork<VM> for SweepDeadCycles<VM> {
    fn do_work(&mut self, _worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        let immix_space = &lxr.immix_space;
        let mut dead_blocks = 0;
        let num_chunks = (self.chunks.end.start() - self.chunks.start.start()) >> Chunk::LOG_BYTES;
        for i in 0..num_chunks {
            let chunk = self.chunks.start.next_nth(i);
            if !immix_space.chunk_map.get(chunk).is_some() {
                continue;
            }
            for block in chunk
                .iter_region::<Block>()
                .filter(|b| b.get_state() != BlockState::Unallocated)
            {
                let dead = self.process_block(block, immix_space);
                // `rc_dead=true`: SweepDeadCycles already zeroed every dead object's RC above, so the
                // block is genuinely empty — force the dealloc path (which also frees to the list).
                if dead && block.rc_sweep_mature::<VM>(immix_space, false, true) {
                    dead_blocks += 1;
                }
            }
        }
        if dead_blocks != 0 {
            immix_space
                .num_clean_blocks_released_mature
                .fetch_add(dead_blocks, Ordering::Relaxed);
        }
    }
}
