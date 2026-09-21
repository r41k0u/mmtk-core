// Use the `{likely, unlikely}` provided by compiler when using nightly
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]

//! Memory Management ToolKit (MMTk) is a portable and high performance memory manager
//! that includes various garbage collection algorithms and provides clean and efficient
//! interfaces to cooperate with language implementations. MMTk features highly modular
//! and highly reusable designs. It includes components such as allocators, spaces and
//! work packets that GC implementers can choose from to compose their own GC plan easily.
//!
//! Logically, this crate includes these major parts:
//! * GC components:
//!   * [Allocators](util/alloc/allocator/trait.Allocator.html): handlers of allocation requests which allocate objects to the bound space.
//!   * [Policies](policy/space/trait.Space.html): definitions of semantics and behaviors for memory regions.
//!     Each space is an instance of a policy, and takes up a unique proportion of the heap.
//!   * [Work packets](scheduler/work/trait.GCWork.html): units of GC work scheduled by the MMTk's scheduler.
//! * [GC plans](plan/global/trait.Plan.html): GC algorithms composed from components.
//! * [Heap implementations](util/heap/index.html): the underlying implementations of memory resources that support spaces.
//! * [Scheduler](scheduler/scheduler/struct.GCWorkScheduler.html): the MMTk scheduler to allow flexible and parallel execution of GC work.
//! * Interfaces: bi-directional interfaces between MMTk and language implementations
//!   i.e. [the memory manager API](memory_manager/index.html) that allows a language's memory manager to use MMTk
//!   and [the VMBinding trait](vm/trait.VMBinding.html) that allows MMTk to call the language implementation.

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
#[macro_use]
extern crate downcast_rs;
#[macro_use]
extern crate static_assertions;
#[macro_use]
extern crate probe;

mod mmtk;
pub use mmtk::MMTKBuilder;
pub(crate) use mmtk::MMAPPER;
pub use mmtk::MMTK;

/// LXR runtime configuration (P1 scaffolding — additive, gated to the future LXR plan).
pub mod args;

mod global_state;
pub use crate::global_state::LiveBytesStats;

mod policy;

pub mod build_info;
pub mod memory_manager;
pub mod plan;
pub mod scheduler;
pub mod util;
pub mod vm;

pub use crate::plan::{
    AllocationSemantics, BarrierSelector, Mutator, MutatorContext, ObjectQueue, Plan,
};

// ---- LXR (P3, additive) — lazy-sweeping job counter ----------------------------------------
//
// Vendored (and *simplified*) from the LXR research fork's `LazySweepingJobsCounter`. In the
// reference it is an RAII token threaded through every decrement / block-sweep packet; when the
// last token of a generation drops it fires the registered `end_of_decs` / `end_of_lazy`
// callbacks (which kick off the lazy mature sweep). That global callback infrastructure
// (`LazySweepingJobs`, the swap-on-GC double-buffering, the `postpone`d sweep jobs) is part of
// the *lazy-decrement / concurrent-sweeping* machinery we are deliberately **deferring** for the
// minimal single-domain RC cut. So here the type is reduced to a trivially-cloneable marker that
// only carries the two reference-counted `Arc<AtomicUsize>` generations so the `ProcessDecs` /
// `SweepBlocksAfterDecs` signatures that take/clone it compile unchanged. It performs no
// callback on drop. When lazy decrements are wired (deferred), restore the reference's Drop +
// `LazySweepingJobs` registry.
/// Counter handles for LXR's lazy sweeping jobs (see `LazySweepingJobs`).
#[allow(dead_code)]
pub struct LazySweepingJobsCounter {
    decs_counter: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
    counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[allow(dead_code)]
impl LazySweepingJobsCounter {
    /// A fresh counter (no decs generation). Minimal port: self-contained `Arc`s, no registry.
    pub fn new() -> Self {
        Self {
            decs_counter: None,
            counter: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// A counter that also participates in the "decs" generation. Minimal port: same as `new`
    /// but seeds the `decs_counter` arm so `clone_with_decs` keeps a decs generation alive.
    pub fn new_decs() -> Self {
        Self {
            decs_counter: Some(std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0))),
            counter: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Clone, sharing the `counter` generation but dropping the decs arm (matches the reference).
    #[allow(clippy::should_implement_trait)]
    pub fn clone(&self) -> Self {
        Self {
            decs_counter: None,
            counter: self.counter.clone(),
        }
    }

    /// Clone, sharing both generations.
    pub fn clone_with_decs(&self) -> Self {
        Self {
            decs_counter: self.decs_counter.clone(),
            counter: self.counter.clone(),
        }
    }
}

impl Default for LazySweepingJobsCounter {
    fn default() -> Self {
        Self::new()
    }
}
