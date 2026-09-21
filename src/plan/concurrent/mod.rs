pub mod bactrian;
pub mod barrier;
pub(super) mod concurrent_marking_work;
pub(super) mod global;

pub mod immix;

use bytemuck::NoUninit;

/// Bring-up diagnostics for generational concurrent plans (Bactrian): global counters
/// of concurrent-marking traffic, printed by the plan when BACTRIAN_TRACE is set.
pub(crate) mod diag {
    use std::sync::atomic::AtomicUsize;
    /// Objects handed to ConcurrentTraceObjects packets (seeds + SATB + recursion).
    pub static ENQUEUED: AtomicUsize = AtomicUsize::new(0);
    /// Objects processed by ConcurrentTraceObjects::trace_object.
    pub static TRACED: AtomicUsize = AtomicUsize::new(0);
    /// Young references skipped by the concurrent trace.
    pub static SKIPPED_YOUNG: AtomicUsize = AtomicUsize::new(0);
    /// InitialMark mark-seed objects pushed.
    pub static SEEDED: AtomicUsize = AtomicUsize::new(0);
    /// SATB old values enqueued by the barrier.
    pub static SATB_ENQ: AtomicUsize = AtomicUsize::new(0);
    /// SATB old values handed to a ConcurrentTraceObjects packet (ProcessModBufSATB ran).
    pub static SATB_RUN: AtomicUsize = AtomicUsize::new(0);
    /// Bytes of objects newly marked by the concurrent/sliced trace (each
    /// object once: counted when it is enqueued for scanning). Per-cycle
    /// deltas are the cycle's marked live size — the honest "live" for heap
    /// sizing and pacing under a sliced cycle, where reserved pages after the
    /// pause also contain the unswept garbage and everything born black.
    pub static MARKED_BYTES: AtomicUsize = AtomicUsize::new(0);
    /// SATB old values dropped as young by the barrier.
    pub static SATB_YOUNG_DROP: AtomicUsize = AtomicUsize::new(0);
}

/// The pause type for a concurrent GC phase.
// TODO: This is probably not be general enough for all the concurrent plans.
// TODO: We could consider moving this to specific plans later.
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Copy, Clone, NoUninit, Default)]
pub enum Pause {
    /// A whole GC (including root scanning, closure, releasing, etc.) happening in a single pause.
    ///
    /// Don't be confused with "full-heap" GC in generational collectors.  `Pause::Full` can also
    /// refer to a nursery GC that happens in a single pause.
    #[default]
    Full = 1,
    /// The initial pause before concurrent marking.
    InitialMark,
    /// The pause after concurrent marking.
    FinalMark,
    /// A nursery collection in a single pause, while a concurrent marking cycle may be
    /// in progress. Used by generational concurrent plans (Bactrian); the marking state
    /// is untouched by this pause.
    Nursery,
}

unsafe impl bytemuck::ZeroableInOption for Pause {}

unsafe impl bytemuck::PodInOption for Pause {}
