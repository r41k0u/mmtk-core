use super::defrag::Histogram;
use super::line::Line;
use super::ImmixSpace;
use crate::util::constants::*;
use crate::util::heap::blockpageresource::BlockPool;
use crate::util::heap::chunk_map::Chunk;
use crate::util::linear_scan::{Region, RegionIterator};
use crate::util::metadata::side_metadata::{MetadataByteArrayRef, SideMetadataSpec};
#[cfg(feature = "vo_bit")]
use crate::util::metadata::vo_bit;
#[cfg(feature = "object_pinning")]
use crate::util::metadata::MetadataSpec;
use crate::util::object_enum::BlockMayHaveObjects;
use crate::util::Address;
use crate::vm::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// LXR/RC: the global phase-epoch counter — bumped at the end of every mutator and GC
/// phase. A block whose per-block `PHASE_EPOCH` matches is "nursery/reusing" for the
/// current phase. Vendored from lxr-v0.32.0 block.rs; read only by RC paths (inert
/// otherwise, so the shipping plans are unaffected).
static GLOBAL_PHASE_EPOCH: AtomicU8 = AtomicU8::new(1);

/// The block allocation state.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BlockState {
    /// the block is not allocated.
    Unallocated,
    /// the block is allocated but not marked.
    Unmarked,
    /// the block is allocated and marked.
    Marked,
    /// the block is marked as reusable.
    Reusable { unavailable_lines: u8 },
}

impl BlockState {
    /// Private constant
    const MARK_UNALLOCATED: u8 = 0;
    /// Private constant
    const MARK_UNMARKED: u8 = u8::MAX;
    /// Private constant
    const MARK_MARKED: u8 = u8::MAX - 1;
}

impl From<u8> for BlockState {
    fn from(state: u8) -> Self {
        match state {
            Self::MARK_UNALLOCATED => BlockState::Unallocated,
            Self::MARK_UNMARKED => BlockState::Unmarked,
            Self::MARK_MARKED => BlockState::Marked,
            unavailable_lines => BlockState::Reusable { unavailable_lines },
        }
    }
}

impl From<BlockState> for u8 {
    fn from(state: BlockState) -> Self {
        match state {
            BlockState::Unallocated => BlockState::MARK_UNALLOCATED,
            BlockState::Unmarked => BlockState::MARK_UNMARKED,
            BlockState::Marked => BlockState::MARK_MARKED,
            BlockState::Reusable { unavailable_lines } => unavailable_lines,
        }
    }
}

impl BlockState {
    /// Test if the block is reuasable.
    pub const fn is_reusable(&self) -> bool {
        matches!(self, BlockState::Reusable { .. })
    }
}

/// Data structure to reference an immix block.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct Block(Address);

impl Region for Block {
    #[cfg(not(feature = "immix_smaller_block"))]
    const LOG_BYTES: usize = 15;
    #[cfg(feature = "immix_smaller_block")]
    const LOG_BYTES: usize = 13;

    fn from_aligned_address(address: Address) -> Self {
        debug_assert!(address.is_aligned_to(Self::BYTES));
        Self(address)
    }

    fn start(&self) -> Address {
        self.0
    }
}

impl BlockMayHaveObjects for Block {
    fn may_have_objects(&self) -> bool {
        self.get_state() != BlockState::Unallocated
    }
}

impl Block {
    /// Log pages in block
    pub const LOG_PAGES: usize = Self::LOG_BYTES - LOG_BYTES_IN_PAGE as usize;
    /// Pages in block
    pub const PAGES: usize = 1 << Self::LOG_PAGES;
    /// Log lines in block
    pub const LOG_LINES: usize = Self::LOG_BYTES - Line::LOG_BYTES;
    /// Lines in block
    pub const LINES: usize = 1 << Self::LOG_LINES;

    /// Block defrag state table (side)
    pub const DEFRAG_STATE_TABLE: SideMetadataSpec =
        crate::util::metadata::side_metadata::spec_defs::IX_BLOCK_DEFRAG;

    /// Block mark table (side)
    pub const MARK_TABLE: SideMetadataSpec =
        crate::util::metadata::side_metadata::spec_defs::IX_BLOCK_MARK;

    // ---- LXR (P2, additive) ----
    // Const aliases naming the LXR-only per-block RC side-metadata specs. These only
    // name a defined spec; they are registered into the ImmixSpace metadata vec solely
    // inside the `if rc_enabled` branch of `ImmixSpace::side_metadata_specs`, so for all
    // non-LXR plans they are defined-but-not-mapped. No behaviour.
    /// Per-block "logged" bit table (side) — LXR only.
    pub const LOG_TABLE: SideMetadataSpec =
        crate::util::metadata::side_metadata::spec_defs::IX_BLOCK_LOG;
    /// Per-block nursery-promotion state table (side) — LXR only.
    pub const NURSERY_PROMOTION_STATE_TABLE: SideMetadataSpec =
        crate::util::metadata::side_metadata::spec_defs::NURSERY_PROMOTION_STATE;
    /// Per-block GC phase-epoch table (side) — LXR only.
    pub const PHASE_EPOCH: SideMetadataSpec =
        crate::util::metadata::side_metadata::spec_defs::PHASE_EPOCH;
    // NOTE: the LXR fork also has per-block `BLOCK_IN_USE` (sweep spin-lock) and `BLOCK_OWNER`
    // (copying-GC owner) side-metadata. Our minimal in-place STW cut needs NEITHER (the mature
    // sweep is uncontended STW, and we never set/read an owner), and crucially neither is mapped
    // in `ImmixSpace::side_metadata_specs` — so touching them read UNMAPPED metadata (the BLOCK_OWNER
    // and BLOCK_IN_USE atomic-load SIGSEGVs). They are intentionally not used; their `spec_defs`
    // entries stay (unmapped, defined-but-unused) only to preserve the offsets of later specs.

    /// Get the chunk containing the block.
    pub fn chunk(&self) -> Chunk {
        Chunk::from_unaligned_address(self.0)
    }

    /// Get the address range of the block's line mark table.
    #[allow(clippy::assertions_on_constants)]
    pub fn line_mark_table(&self) -> MetadataByteArrayRef<{ Block::LINES }> {
        debug_assert!(!super::BLOCK_ONLY);
        MetadataByteArrayRef::<{ Block::LINES }>::new(&Line::MARK_TABLE, self.start(), Self::BYTES)
    }

    /// Get block mark state.
    pub fn get_state(&self) -> BlockState {
        let byte = Self::MARK_TABLE.load_atomic::<u8>(self.start(), Ordering::SeqCst);
        byte.into()
    }

    /// Set block mark state.
    pub fn set_state(&self, state: BlockState) {
        let state = u8::from(state);
        Self::MARK_TABLE.store_atomic::<u8>(self.start(), state, Ordering::SeqCst);
    }

    // Defrag byte

    const DEFRAG_SOURCE_STATE: u8 = u8::MAX;

    /// Test if the block is marked for defragmentation.
    pub fn is_defrag_source(&self) -> bool {
        let byte = Self::DEFRAG_STATE_TABLE.load_atomic::<u8>(self.start(), Ordering::SeqCst);
        // The byte should be 0 (not defrag source) or 255 (defrag source) if this is a major defrag GC, as we set the values in PrepareBlockState.
        // But it could be any value in a nursery GC.
        byte == Self::DEFRAG_SOURCE_STATE
    }

    /// Mark the block for defragmentation.
    pub fn set_as_defrag_source(&self, defrag: bool) {
        let byte = if defrag { Self::DEFRAG_SOURCE_STATE } else { 0 };
        Self::DEFRAG_STATE_TABLE.store_atomic::<u8>(self.start(), byte, Ordering::SeqCst);
    }

    // ── LXR / RC per-block log + field-unlog (P3.4) ───────────────────────────
    // Vendored from the LXR fork (lxr-v0.32.0 block.rs). Operate on the per-block
    // LOG_TABLE (added in P2) + the VM-side field-unlog table; inert until the RC
    // (LXR) plan runs, so the shipping plans stay byte-identical.

    /// Atomically set this block's per-block log bit. Returns true iff it was previously
    /// unlogged (so the caller performs the once-per-epoch block bookkeeping).
    pub fn log(&self) -> bool {
        loop {
            let old_value: u8 = Self::LOG_TABLE.load_atomic::<u8>(self.start(), Ordering::Relaxed);
            if old_value == 1 {
                return false;
            }
            if Self::LOG_TABLE
                .compare_exchange_atomic::<u8>(
                    self.start(),
                    0u8,
                    1u8,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Clear this block's per-block log bit.
    pub fn unlog(&self) {
        Self::LOG_TABLE.store_atomic::<u8>(self.start(), 0u8, Ordering::Relaxed);
    }

    /// Zero this block's slice of the per-field unlog table (a fresh block has no logged
    /// fields). Uses the VM-side `GLOBAL_FIELD_UNLOG_BIT_SPEC`.
    pub fn clear_field_unlog_table<VM: VMBinding>(&self) {
        use crate::vm::ObjectModel;
        VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
            .as_spec()
            .extract_side_spec()
            .bzero_metadata(self.start(), Block::BYTES);
    }

    // ── LXR / RC phase-epoch nursery identification (P3.4) ─────────────────────
    // A block is "nursery/reusing" for the current phase iff its per-block PHASE_EPOCH
    // (P2 spec) matches GLOBAL_PHASE_EPOCH (odd = mutator phase, even = GC phase).
    // Vendored from lxr-v0.32.0 block.rs; read by ProcessIncs to spot fresh nursery
    // objects. update_global_phase_epoch is deferred (needs the page-resource RC API).

    /// The global phase-epoch (bumped at the end of every mutator and GC phase).
    pub fn global_phase_epoch() -> u8 {
        GLOBAL_PHASE_EPOCH.load(Ordering::Relaxed)
    }

    /// This block's phase epoch — the last phase it was used for object allocation.
    pub fn phase_epoch(&self) -> u8 {
        Self::PHASE_EPOCH.load_atomic::<u8>(self.start(), Ordering::Relaxed)
    }

    /// Stamp this block's phase epoch with the current global epoch.
    pub fn update_phase_epoch(&self) {
        Self::PHASE_EPOCH.store_atomic::<u8>(
            self.start(),
            Self::global_phase_epoch(),
            Ordering::Relaxed,
        );
    }

    /// True iff this block was allocated (clean) or reused (partially free) in the
    /// current phase.
    pub fn is_nursery_or_reusing(&self) -> bool {
        let ge = Self::global_phase_epoch();
        let e = self.phase_epoch();
        if (ge & 1) == 1 {
            e == ge
        } else {
            e == ge - 1
        }
    }

    /// True iff this is a fresh (unallocated) nursery block this phase.
    pub fn is_nursery(&self) -> bool {
        self.get_state() == BlockState::Unallocated && self.is_nursery_or_reusing()
    }

    /// True iff this is a partially-free block being reused this phase.
    pub fn is_reusing(&self) -> bool {
        self.get_state() != BlockState::Unallocated && self.is_nursery_or_reusing()
    }

    /// Record the number of holes in the block.
    pub fn set_holes(&self, holes: usize) {
        Self::DEFRAG_STATE_TABLE.store_atomic::<u8>(self.start(), holes as u8, Ordering::SeqCst);
    }

    /// Get the number of holes.
    pub fn get_holes(&self) -> usize {
        let byte = Self::DEFRAG_STATE_TABLE.load_atomic::<u8>(self.start(), Ordering::SeqCst);
        debug_assert_ne!(byte, Self::DEFRAG_SOURCE_STATE);
        byte as usize
    }

    /// Initialize a clean block after acquired from page-resource.
    pub fn init(&self, copy: bool) {
        self.set_state(if copy {
            BlockState::Marked
        } else {
            BlockState::Unmarked
        });
        Self::DEFRAG_STATE_TABLE.store_atomic::<u8>(self.start(), 0, Ordering::SeqCst);
    }

    /// Deinitalize a block before releasing.
    pub fn deinit(&self) {
        self.set_state(BlockState::Unallocated);
    }

    // ── LXR / RC block lifecycle + sweep (P3.5, additive) ──────────────────────
    // Vendored and adapted from lxr-v0.32.0 block.rs. Inert until the LXR plan runs
    // (`rc_enabled` stays false), so the existing `init`/`deinit` above (used by the 10
    // shipping plans) are kept byte-identical and these RC variants are added alongside.
    //
    // ADAPTATION: the reference *replaced* `Block::init`/`deinit` with `init<VM>(copy, reuse,
    // &ImmixSpace)` / `deinit<VM>(&ImmixSpace)`. We instead add `init_rc`/`deinit_rc` so the
    // non-RC callers (`ImmixSpace::get_clean_block` etc.) keep calling the original 1-/0-arg
    // versions unchanged. `init_rc`'s `!rc_enabled` arm matches the original `init` exactly.

    /// RC-aware clean/reused block initialisation. (Reference: `Block::init(copy, reuse, space)`.)
    pub fn init_rc<VM: VMBinding>(&self, copy: bool, reuse: bool, space: &ImmixSpace<VM>) {
        self.update_phase_epoch();
        if space.rc_enabled {
            if !reuse {
                debug_assert_eq!(self.get_state(), BlockState::Unallocated);
            }
            self.clear_in_place_promoted();
            // NOTE: the reference asserted the per-block phase-epoch parity here (odd in a mutator
            // phase, even in a GC phase). We use the SINGLE-bump scheme (one bump per GC), under
            // which a mutator phase's parity alternates each GC, so those asserts do not hold — and
            // the parity is no longer load-bearing (the RC sweeps key on block STATE, not epoch).
            // Asserts dropped.
            if copy {
                if reuse {
                    debug_assert!(!self.is_defrag_source());
                }
                self.set_state(BlockState::Unmarked);
                self.set_as_defrag_source(false);
            } else if reuse {
                debug_assert!(!self.is_defrag_source());
            } else {
                debug_assert_eq!(self.get_state(), BlockState::Unallocated);
                self.set_as_defrag_source(false);
            }
        } else {
            self.set_state(if copy {
                BlockState::Marked
            } else {
                BlockState::Unmarked
            });
            if !reuse {
                Self::DEFRAG_STATE_TABLE.store_atomic::<u8>(self.start(), 0, Ordering::SeqCst);
            }
        }
    }

    /// RC-aware deinit before release. (Reference: `Block::deinit(space)`.)
    pub fn deinit_rc<VM: VMBinding>(&self, space: &ImmixSpace<VM>) {
        self.set_state(BlockState::Unallocated);
        if space.rc_enabled {
            self.clear_in_place_promoted();
            // NB: the reference clears `BLOCK_OWNER` here, but that field tracks the owning mutator
            // for thread-local / copying GC — irrelevant to our in-place RC cut (`moves_objects =
            // false`, no copy reservation). Its side-metadata is deliberately NOT registered in
            // `ImmixSpace::side_metadata_specs`, so storing to it faults on unmapped metadata
            // (the first free's UAF). We neither set nor read BLOCK_OWNER, so drop the clear.
            self.set_as_defrag_source(false);
            // Clear the per-block "possibly-dead-mature" log bit so a freed block does NOT carry a
            // stale `log() == 1` into its next life: otherwise `add_to_possibly_dead_mature_blocks`
            // (which uses `log()` to dedup) would refuse to re-queue the reincarnated block when an
            // object in it later dies -> that block would never be swept (silent leak). The nursery
            // sweep frees via `release_block` (which does NOT call `unlog()`), so the clear must live
            // here to cover both the nursery- and mature-sweep free paths.
            self.unlog();
        }
    }

    /// True during a mutator (odd) phase epoch.
    fn in_mutatar_phase() -> bool {
        (Self::global_phase_epoch() & 1) == 1
    }

    /// Update the global phase epoch (bumped at the end of every mutator and GC phase). On the
    /// 8-bit wrap (254→1) the reference also bulk-zeroes the per-block PHASE_EPOCH metadata via
    /// `space.pr.reset_nursery_state()` — see the note on `BlockPageResource::reset_nursery_state`
    /// (deferred no-op in our free-list page resource).
    pub fn update_global_phase_epoch<VM: VMBinding>(space: &ImmixSpace<VM>) {
        let old = GLOBAL_PHASE_EPOCH.load(Ordering::SeqCst);
        if old == 254 {
            GLOBAL_PHASE_EPOCH.store(1, Ordering::SeqCst);
            space.block_page_resource().reset_nursery_state();
        } else {
            GLOBAL_PHASE_EPOCH.store(old + 1, Ordering::SeqCst);
        }
    }

    // ── RC per-block side-table operations ─────────────────────────────────────

    pub fn clear_rc_table(&self) {
        crate::util::rc::RC_TABLE.bzero_metadata(self.start(), Block::BYTES);
    }

    pub fn clear_striddle_table(&self) {
        crate::util::rc::RC_STRADDLE_LINES.bzero_metadata(self.start(), Block::BYTES);
    }

    pub(super) fn clear_mark_table<VM: VMBinding>(&self) {
        VM::VMObjectModel::LOCAL_MARK_BIT_SPEC
            .extract_side_spec()
            .bzero_metadata(self.start(), Self::BYTES);
    }

    pub(super) fn initialize_mark_table_as_marked<VM: VMBinding>(&self) {
        let meta = VM::VMObjectModel::LOCAL_MARK_BIT_SPEC.extract_side_spec();
        let start: *mut u8 =
            crate::util::metadata::side_metadata::address_to_meta_address(meta, self.start())
                .to_mut_ptr();
        let limit: *mut u8 =
            crate::util::metadata::side_metadata::address_to_meta_address(meta, self.end())
                .to_mut_ptr();
        unsafe {
            let bytes = limit.offset_from(start) as usize;
            std::ptr::write_bytes(start, 0xffu8, bytes);
        }
    }

    pub fn initialize_field_unlog_table_as_unlogged<VM: VMBinding>(&self) {
        let meta = *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
            .as_spec()
            .extract_side_spec();
        let start: *mut u8 =
            crate::util::metadata::side_metadata::address_to_meta_address(&meta, self.start())
                .to_mut_ptr();
        let limit: *mut u8 =
            crate::util::metadata::side_metadata::address_to_meta_address(&meta, self.end())
                .to_mut_ptr();
        unsafe {
            let bytes = limit.offset_from(start) as usize;
            std::ptr::write_bytes(start, 0xffu8, bytes);
        }
    }

    /// True iff every RC entry covering this block is zero (the whole block is dead).
    pub fn rc_dead(&self) -> bool {
        type UInt = u128;
        const LOG_BITS_IN_UINT: usize =
            (std::mem::size_of::<UInt>() << 3).trailing_zeros() as usize;
        const {
            assert!(
                Self::LOG_BYTES - crate::util::rc::LOG_MIN_OBJECT_SIZE
                    + crate::util::rc::LOG_REF_COUNT_BITS
                    >= LOG_BITS_IN_UINT
            )
        };
        let start = crate::util::metadata::side_metadata::address_to_meta_address(
            &crate::util::rc::RC_TABLE,
            self.start(),
        )
        .to_ptr::<UInt>();
        let limit = crate::util::metadata::side_metadata::address_to_meta_address(
            &crate::util::rc::RC_TABLE,
            self.end(),
        )
        .to_ptr::<UInt>();
        let rc_table = unsafe { std::slice::from_raw_parts(start, limit.offset_from(start) as _) };
        for x in rc_table {
            if *x != 0 {
                return false;
            }
        }
        true
    }

    // ── RC in-place nursery promotion state ────────────────────────────────────

    pub fn set_as_in_place_promoted<VM: VMBinding>(&self, space: &ImmixSpace<VM>) {
        if self.is_in_place_promoted() {
            return;
        }
        loop {
            let old_value: u8 =
                Self::NURSERY_PROMOTION_STATE_TABLE.load_atomic(self.start(), Ordering::Relaxed);
            if old_value == 1 {
                return;
            }
            if Self::NURSERY_PROMOTION_STATE_TABLE
                .compare_exchange_atomic(self.start(), 0u8, 1u8, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                space
                    .block_allocation
                    .in_place_promoted_nursery_blocks
                    .fetch_add(1, Ordering::Relaxed);
                self.set_state(BlockState::Unmarked);
                self.update_phase_epoch();
                return;
            }
        }
    }

    pub fn is_in_place_promoted(&self) -> bool {
        Self::NURSERY_PROMOTION_STATE_TABLE.load_atomic::<u8>(self.start(), Ordering::Relaxed) != 0
    }

    fn clear_in_place_promoted(&self) {
        Self::NURSERY_PROMOTION_STATE_TABLE.store_atomic(self.start(), 0u8, Ordering::Relaxed);
    }

    // ── RC mature-sweep block guard (in-place STW cut) ─────────────────────────
    // The reference guards the mature sweep against a CONCURRENT mutator reusing a block with a
    // per-block spin-lock in the `BLOCK_IN_USE` side-metadata table. Our minimal cut runs the
    // decrement + mature sweep STOP-THE-WORLD (in `STWRCDecsAndSweep` + the post-decs epilogue,
    // mutators stopped), so the lock is ALWAYS uncontended and serves no purpose — and, like the
    // deleted `BLOCK_OWNER` clear, `BLOCK_IN_USE` is NOT registered in `side_metadata_specs`, so
    // touching it reads UNMAPPED side-metadata → the atomic-load SIGSEGV the mature sweep hit.
    // So the lock is a no-op: `lock_skip_reusing_or_unallocated` keeps only the SKIP check (don't
    // sweep an unallocated or actively-reused block), and `unlock` does nothing.

    /// Returns true iff the block may be swept now (not unallocated, not a mutator-reused block this
    /// phase). No actual locking (STW: uncontended).
    fn lock_skip_reusing_or_unallocated(&self) -> bool {
        let state = self.get_state();
        if state == BlockState::Unallocated || (Self::in_mutatar_phase() && self.is_reusing()) {
            return false;
        }
        true
    }

    pub fn unlock(&self) {}

    /// Set block mark state via a fetch-update closure. (Reference: `Block::fetch_update_state`.)
    pub fn fetch_update_state(
        &self,
        mut f: impl FnMut(BlockState) -> Option<BlockState> + Copy,
    ) -> Result<BlockState, BlockState> {
        // The inner closure must be `Copy` (the side-metadata `fetch_update_atomic` bound). Capture
        // `f` by value with `move` so the inner closure owns a `Copy` of it (a non-`move` closure
        // would capture `f` by `&mut`, which is not `Copy`).
        Self::MARK_TABLE
            .fetch_update_atomic::<u8, _>(
                self.start(),
                Ordering::SeqCst,
                Ordering::SeqCst,
                move |s| f(s.into()).map(u8::from),
            )
            .map(|x| (x).into())
            .map_err(|x| (x).into())
    }

    /// Try to atomically transition the block to Unallocated, refusing if a mutator is still
    /// reusing it. Returns true iff the block was deallocated.
    fn attempt_dealloc(&self) -> bool {
        self.fetch_update_state(|s| {
            if (Self::in_mutatar_phase() && self.is_reusing()) || s == BlockState::Unallocated {
                None
            } else {
                Some(BlockState::Unallocated)
            }
        })
        .is_ok()
    }

    /// Sweep a (possibly) dead mature block. Returns true iff the block was freed (the caller
    /// then bulk-releases its pages). `defrag` forces the dealloc (mature evac, deferred); `rc_dead`
    /// lets the caller assert deadness without re-scanning the RC table.
    pub fn rc_sweep_mature<VM: VMBinding>(
        &self,
        space: &ImmixSpace<VM>,
        defrag: bool,
        rc_dead: bool,
    ) -> bool {
        if self.get_state() == BlockState::Unallocated {
            return false;
        }
        if defrag || rc_dead || self.rc_dead() {
            if !self.lock_skip_reusing_or_unallocated() {
                return false;
            }
            let dead = if defrag || self.attempt_dealloc() {
                self.deinit_rc(space);
                // Return the block to the page resource's free list so it is actually reusable.
                // (The reference's nosweep PR recycled via a bulk accounting release + cursor
                // reset; our standard free-list `BlockPageResource` needs each block pushed back
                // individually, else freed blocks leak.) This does the accounting release too.
                space.release_block_to_free_list(*self);
                true
            } else {
                false
            };
            self.unlock();
            return dead;
        }
        false
    }

    pub fn start_line(&self) -> Line {
        Line::from_aligned_address(self.start())
    }

    pub fn end_line(&self) -> Line {
        Line::from_aligned_address(self.end())
    }

    /// Get the range of lines within the block.
    #[allow(clippy::assertions_on_constants)]
    pub fn lines(&self) -> RegionIterator<Line> {
        debug_assert!(!super::BLOCK_ONLY);
        RegionIterator::<Line>::new(self.start_line(), self.end_line())
    }

    /// Sweep this block.
    /// Return true if the block is swept.
    pub fn sweep<VM: VMBinding>(
        &self,
        space: &ImmixSpace<VM>,
        mark_histogram: &mut Histogram,
        line_mark_state: Option<u8>,
    ) -> bool {
        if super::BLOCK_ONLY {
            match self.get_state() {
                BlockState::Unallocated => false,
                BlockState::Unmarked => {
                    #[cfg(feature = "vo_bit")]
                    vo_bit::helper::on_region_swept::<VM, _>(self, false);

                    // If the pin bit is not on the side, we cannot bulk zero.
                    // We shouldn't need to clear it here in that case, since the pin bit
                    // should be overwritten at each object allocation. The same applies below
                    // when we are sweeping on a line granularity.
                    #[cfg(feature = "object_pinning")]
                    if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC {
                        side.bzero_metadata(self.start(), Block::BYTES);
                    }

                    // Release the block if it is allocated but not marked by the current GC.
                    space.release_block(*self);
                    true
                }
                BlockState::Marked => {
                    #[cfg(feature = "vo_bit")]
                    vo_bit::helper::on_region_swept::<VM, _>(self, true);

                    // The block is live.
                    false
                }
                _ => unreachable!(),
            }
        } else {
            // Calculate number of marked lines and holes.
            let mut marked_lines = 0;
            let mut holes = 0;
            let mut prev_line_is_marked = true;
            let line_mark_state = line_mark_state.unwrap();

            for line in self.lines() {
                if line.is_marked(line_mark_state) {
                    marked_lines += 1;
                    prev_line_is_marked = true;
                } else {
                    if prev_line_is_marked {
                        holes += 1;
                    }
                    // We need to clear the line mark state at least twice in every 128 GC
                    // otherwise, the line mark state of the last GC will stick around
                    if line_mark_state > Line::MAX_MARK_STATE - 2 {
                        line.mark(0);
                    }
                    #[cfg(feature = "immix_zero_on_release")]
                    crate::util::memory::zero(line.start(), Line::BYTES);

                    // We need to clear the pin bit if it is on the side, as this line can be reused
                    #[cfg(feature = "object_pinning")]
                    if let MetadataSpec::OnSide(side) = *VM::VMObjectModel::LOCAL_PINNING_BIT_SPEC {
                        side.bzero_metadata(line.start(), Line::BYTES);
                    }

                    prev_line_is_marked = false;
                }
            }

            if marked_lines == 0 {
                #[cfg(feature = "vo_bit")]
                vo_bit::helper::on_region_swept::<VM, _>(self, false);

                // Release the block if non of its lines are marked.
                space.release_block(*self);
                true
            } else {
                // There are some marked lines. Keep the block live.
                if marked_lines != Block::LINES {
                    // There are holes. Mark the block as reusable.
                    self.set_state(BlockState::Reusable {
                        unavailable_lines: marked_lines as _,
                    });
                    space.reusable_blocks.push(*self)
                } else {
                    // Clear mark state.
                    self.set_state(BlockState::Unmarked);
                }
                // Update mark_histogram
                mark_histogram[holes] += marked_lines;
                // Record number of holes in block side metadata.
                self.set_holes(holes);

                #[cfg(feature = "vo_bit")]
                vo_bit::helper::on_region_swept::<VM, _>(self, true);

                false
            }
        }
    }

    /// Clear VO bits metadata for unmarked regions.
    /// This is useful for clearing VO bits during nursery GC for StickyImmix
    /// at which time young objects (allocated in unmarked regions) may die
    /// but we always consider old objects (in marked regions) as live.
    #[cfg(feature = "vo_bit")]
    pub fn clear_vo_bits_for_unmarked_regions(&self, line_mark_state: Option<u8>) {
        match line_mark_state {
            None => {
                match self.get_state() {
                    BlockState::Unmarked => {
                        // It may contain young objects.  Clear it.
                        vo_bit::bzero_vo_bit(self.start(), Self::BYTES);
                    }
                    BlockState::Marked => {
                        // It contains old objects.  Skip it.
                    }
                    _ => unreachable!(),
                }
            }
            Some(state) => {
                // With lines.
                for line in self.lines() {
                    if !line.is_marked(state) {
                        // It may contain young objects.  Clear it.
                        vo_bit::bzero_vo_bit(line.start(), Line::BYTES);
                    }
                }
            }
        }
    }
}

/// A non-block single-linked list to store blocks.
pub struct ReusableBlockPool {
    queue: BlockPool<Block>,
    num_workers: usize,
}

impl ReusableBlockPool {
    /// Create empty block list
    pub fn new(num_workers: usize) -> Self {
        Self {
            queue: BlockPool::new(num_workers),
            num_workers,
        }
    }

    /// Get number of blocks in this list.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Add a block to the list.
    pub fn push(&self, block: Block) {
        self.queue.push(block)
    }

    /// Pop a block out of the list.
    pub fn pop(&self) -> Option<Block> {
        self.queue.pop()
    }

    /// Clear the list.
    pub fn reset(&mut self) {
        self.queue = BlockPool::new(self.num_workers);
    }

    /// Iterate all the blocks in the queue. Call the visitor for each reported block.
    pub fn iterate_blocks(&self, mut f: impl FnMut(Block)) {
        self.queue.iterate_blocks(&mut f);
    }

    /// Flush the block queue
    pub fn flush_all(&self) {
        self.queue.flush_all();
    }
}
