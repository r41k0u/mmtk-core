use super::barrier::BactrianBarrier;
use super::global::Bactrian;
use crate::plan::concurrent::Pause;
use crate::plan::mutator_context::create_allocator_mapping;
use crate::plan::mutator_context::Mutator;
use crate::plan::mutator_context::MutatorBuilder;
use crate::plan::mutator_context::MutatorConfig;
use crate::plan::mutator_context::ReservedAllocators;
use crate::plan::AllocationSemantics;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::alloc::BumpAllocator;
use crate::util::alloc::ImmixAllocator;
use crate::util::{VMMutatorThread, VMWorkerThread};
use crate::vm::VMBinding;
use crate::MMTK;
use enum_map::EnumMap;

#[cfg(feature = "marksweep_as_nonmoving")]
fn common_nonmoving_prepare<VM: VMBinding>(mutator: &mut Mutator<VM>) {
    unsafe {
        mutator
            .allocators
            .get_typed_allocator_mut::<crate::util::alloc::FreeListAllocator<VM>>(
                AllocatorSelector::FreeList(0),
            )
    }
    .prepare();
}

#[cfg(feature = "marksweep_as_nonmoving")]
fn common_nonmoving_release<VM: VMBinding>(mutator: &mut Mutator<VM>) {
    unsafe {
        mutator
            .allocators
            .get_typed_allocator_mut::<crate::util::alloc::FreeListAllocator<VM>>(
                AllocatorSelector::FreeList(0),
            )
    }
    .release();
}

// Reset the pretenure ImmixAllocator in both prepare and release
// (ConcurrentImmix precedent: InitialMark schedules no mutator release and
// FinalMark no prepare, so both hooks must invalidate the stale bump cursor).
fn reset_pretenure_allocator<VM: VMBinding>(mutator: &mut Mutator<VM>) {
    // In freelist mode NonMoving maps to FreeList(0) (handled by
    // common_nonmoving_prepare/release); the reserved Immix(0) mutator
    // allocator is then unused and needs no reset.
    if medium_to_freelist() {
        return;
    }
    let immix_allocator = unsafe {
        mutator
            .allocators
            .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::NonMoving])
    }
    .downcast_mut::<ImmixAllocator<VM>>()
    .unwrap();
    immix_allocator.reset();
}

pub fn bactrian_mutator_prepare<VM: VMBinding>(mutator: &mut Mutator<VM>, _tls: VMWorkerThread) {
    #[cfg(feature = "marksweep_as_nonmoving")]
    common_nonmoving_prepare(mutator);
    reset_pretenure_allocator(mutator);
    let current_pause = mutator.plan.concurrent().unwrap().current_pause().unwrap();
    // Arm the SATB half of the barrier for the marking cycle that starts when this
    // pause ends. (Concurrent marking state proper is armed in the plan's prepare.)
    if current_pause == Pause::InitialMark {
        mutator
            .barrier
            .downcast_mut::<BactrianBarrier<VM>>()
            .unwrap()
            .set_satb_enabled(true);
    }
}

pub fn bactrian_mutator_release<VM: VMBinding>(mutator: &mut Mutator<VM>, _tls: VMWorkerThread) {
    // Reset the nursery allocator: the nursery was evacuated (every pause except
    // Full is nursery-anchored; Full collects the nursery too).
    let bump_allocator = unsafe {
        mutator
            .allocators
            .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::Default])
    }
    .downcast_mut::<BumpAllocator<VM>>()
    .unwrap();
    bump_allocator.reset();

    // The freelist (nonmoving MS) mutator release pairs with the SPACE-side
    // MarkSweepSpace::release handshake (pending_release_packets =
    // num_mutators + 1), which round 31 confines to the pauses whose marks
    // are complete for that space: Full and FinalMark. Running it at other
    // pauses both corrupts (it frees unmarked-since-prepare blocks) and
    // underflows the unarmed counter.
    let current_pause = mutator.plan.concurrent().unwrap().current_pause().unwrap();
    #[cfg(feature = "marksweep_as_nonmoving")]
    if matches!(current_pause, Pause::Full | Pause::FinalMark) {
        common_nonmoving_release(mutator);
    }
    reset_pretenure_allocator(mutator);
    // Disarm the SATB half when the marking cycle ends.
    if current_pause == Pause::FinalMark || current_pause == Pause::Full {
        mutator
            .barrier
            .downcast_mut::<BactrianBarrier<VM>>()
            .unwrap()
            .set_satb_enabled(false);
    }
}

/// Bactrian reserves one bump pointer (the nursery TLAB) and one Immix
/// allocator: the MATURE space, exposed to mutators under
/// AllocationSemantics::NonMoving to implement stock OCaml's
/// Max_young_wosize pretenuring — blocks above the boundary are born in the
/// major heap (never transiting the minor heap), exactly as
/// shared_heap.c:504/515 does via pools/malloc. The runtime routes the
/// band of 2056 B and larger here under MMTK_MEDIUM_NONMOVING (see caml_mmtk_semantics).
/// Born-mature objects are unlogged at birth (binding, post-alloc) so the
/// generational barrier remembers their young stores; during concurrent
/// marking the ImmixAllocator's allocate-as-live path keeps them from the
/// FinalMark sweep, same as InitialMark promotions.
const BACTRIAN_RESERVED: ReservedAllocators = ReservedAllocators {
    n_bump_pointer: 1,
    n_immix: 1,
    ..ReservedAllocators::DEFAULT
};

/// Where the pretenured >=2056B band lives (round 30). `immix` (default):
/// the mature Immix space — the round-23..29 behaviour. EXPERIMENTAL
/// `MMTK_MEDIUM_TO=freelist`: the common mark-sweep nonmoving space —
/// VANILLA'S reclamation regime for exactly these objects (shared_heap.c
/// pools: a dead cell relinks into a size-class free list and is reused in
/// place, no mark cycle needed), the root-cause fix for fragmed's 3.06x/
/// 189MB gap. Staged OFF because it is UNSOUND under concurrent cycles as
/// of round 30d: the MS space's lazy sweep runs at block-acquisition time
/// using the in-flight cycle's INCOMPLETE marks and frees not-yet-marked
/// live cells (reproduced: fragmed segfault, a marking quantum scanning a
/// freed cell whose header was a free-list link); the eager_sweeping
/// feature deadlocks under Bactrian's pause schedule. The sound design
/// (next round): mid-cycle block acquisition serves CLEAN blocks only —
/// no sweep may consume incomplete marks — plus the allocate-black path
/// already added to FreeListAllocator. Requires `marksweep_as_nonmoving`.
fn medium_to_freelist() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        cfg!(feature = "marksweep_as_nonmoving")
            && matches!(
                std::env::var("MMTK_MEDIUM_TO").as_deref(),
                Ok("freelist") | Ok("Freelist") | Ok("FreeList")
            )
    })
}

lazy_static::lazy_static! {
    static ref ALLOCATOR_MAPPING: EnumMap<AllocationSemantics, AllocatorSelector> = {
        let mut map = create_allocator_mapping(BACTRIAN_RESERVED, true);
        map[AllocationSemantics::Default] = AllocatorSelector::BumpPointer(0);
        // freelist mode keeps the common mapping (NonMoving -> FreeList(0),
        // the common mark-sweep nonmoving space).
        if !medium_to_freelist() {
            map[AllocationSemantics::NonMoving] = AllocatorSelector::Immix(0);
        }
        map
    };
}

pub fn create_bactrian_mutator<VM: VMBinding>(
    mutator_tls: VMMutatorThread,
    mmtk: &'static MMTK<VM>,
) -> Mutator<VM> {
    let bactrian = mmtk.get_plan().downcast_ref::<Bactrian<VM>>().unwrap();
    let config = MutatorConfig {
        allocator_mapping: &ALLOCATOR_MAPPING,
        space_mapping: Box::new({
            let mut vec = crate::plan::mutator_context::create_space_mapping(
                BACTRIAN_RESERVED,
                true,
                mmtk.get_plan(),
            );
            vec.push((AllocatorSelector::BumpPointer(0), &bactrian.gen.nursery));
            vec.push((AllocatorSelector::Immix(0), &bactrian.immix_space));
            vec
        }),
        prepare_func: &bactrian_mutator_prepare,
        release_func: &bactrian_mutator_release,
    };

    let builder = MutatorBuilder::new(mutator_tls, mmtk, config);
    let mut mutator = builder
        .barrier(Box::new(BactrianBarrier::new(mmtk, mutator_tls)))
        .build();

    // A mutator created mid-cycle (e.g. a new domain spawned during concurrent
    // marking) must start with the SATB barrier armed.
    mutator
        .barrier
        .downcast_mut::<BactrianBarrier<VM>>()
        .unwrap()
        .set_satb_enabled(bactrian.is_concurrent_marking_active());

    mutator
}
