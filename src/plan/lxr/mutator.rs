use super::barrier::LXRFieldBarrierSemantics;
use super::LXR;
use crate::plan::barriers::FieldBarrier;
use crate::plan::mutator_context::create_allocator_mapping;
use crate::plan::mutator_context::create_space_mapping;
use crate::plan::mutator_context::Mutator;
use crate::plan::mutator_context::MutatorBuilder;
use crate::plan::mutator_context::MutatorConfig;
use crate::plan::mutator_context::ReservedAllocators;
use crate::plan::AllocationSemantics;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::alloc::ImmixAllocator;
use crate::util::opaque_pointer::{VMMutatorThread, VMWorkerThread};
use crate::vm::VMBinding;
use crate::MMTK;
use enum_map::EnumMap;

// P3.5: cloned from plan/immix/mutator.rs (Immix -> LXR). One Immix allocator at
// AllocationSemantics::Default; the LXR field barrier is installed in a later P3 step.
pub fn lxr_mutator_release<VM: VMBinding>(mutator: &mut Mutator<VM>, _tls: VMWorkerThread) {
    let immix_allocator = unsafe {
        mutator
            .allocators
            .get_allocator_mut(mutator.config.allocator_mapping[AllocationSemantics::Default])
    }
    .downcast_mut::<ImmixAllocator<VM>>()
    .unwrap();
    immix_allocator.reset();

    // The plan-level release passes full_heap = false, so a mark-sweep
    // nonmoving space never arms its release handshake. Releasing its
    // allocator here would decrement pending_release_packets once per
    // mutator per pause and wrap it at the first GC.
    #[cfg(not(feature = "marksweep_as_nonmoving"))]
    crate::plan::mutator_context::common_release_func(mutator, _tls);
}

/// Pairs with the plan-level prepare, which also passes full_heap = false:
/// the mark-sweep nonmoving space is not prepared, so neither is its allocator.
pub fn lxr_mutator_prepare<VM: VMBinding>(_mutator: &mut Mutator<VM>, _tls: VMWorkerThread) {
    #[cfg(not(feature = "marksweep_as_nonmoving"))]
    crate::plan::mutator_context::common_prepare_func(_mutator, _tls);
}

pub(in crate::plan) const RESERVED_ALLOCATORS: ReservedAllocators = ReservedAllocators {
    n_immix: 1,
    ..ReservedAllocators::DEFAULT
};

lazy_static! {
    pub static ref ALLOCATOR_MAPPING: EnumMap<AllocationSemantics, AllocatorSelector> = {
        let mut map = create_allocator_mapping(RESERVED_ALLOCATORS, true);
        map[AllocationSemantics::Default] = AllocatorSelector::Immix(0);
        map
    };
}

pub fn create_lxr_mutator<VM: VMBinding>(
    mutator_tls: VMMutatorThread,
    mmtk: &'static MMTK<VM>,
) -> Mutator<VM> {
    let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
    let config = MutatorConfig {
        allocator_mapping: &ALLOCATOR_MAPPING,
        space_mapping: Box::new({
            let mut vec = create_space_mapping(RESERVED_ALLOCATORS, true, lxr);
            vec.push((AllocatorSelector::Immix(0), &lxr.immix_space));
            vec
        }),
        prepare_func: &lxr_mutator_prepare,
        release_func: &lxr_mutator_release,
    };

    // Install the LXR coalescing field-logging write barrier (per-field unlog bit + inc/dec
    // buffering). Mirrors how GenImmix/ConcurrentImmix install theirs. `LXR_CONSTRAINTS.barrier`
    // = `FieldBarrier`, so the framework drives the barrier on `object_reference_write`.
    MutatorBuilder::new(mutator_tls, mmtk, config)
        .barrier(Box::new(FieldBarrier::new(LXRFieldBarrierSemantics::new(
            mmtk,
        ))))
        .build()
}
