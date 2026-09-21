//! LXR — reference counting on a hierarchical Immix heap (Zhao, Blackburn & McKinley,
//! PLDI'22). Minimal single-domain port for RQ1; see `LXR_PORT_PLAN.md`. Vendored and
//! adapted from the LXR fork (wenyuzhao/mmtk-core `lxr-v0.32.0`). Everything here is gated
//! behind the LXR plan (`rc_enabled`/`needs_field_log_bit`); no other plan reaches it, so
//! the 10 shipping plans stay byte-identical until `MMTK_PLAN=LXR` is wired (P3.8).

pub(crate) mod barrier;
pub(super) mod gc_work;
pub(super) mod global;
pub(super) mod mutator;
pub(crate) mod rc;

pub use self::global::LXR;

use bytemuck::NoUninit;

/// LXR collection-pause kind. Distinct from [`crate::plan::concurrent::Pause`] so the
/// ConcurrentImmix plan's pause type (and its exhaustive matches) stay untouched.
///
/// The minimal RC subset uses only [`Pause::RefCount`] (the steady-state pause: process
/// increments/decrements + nursery sweep + lazy block sweep, no tracing) and [`Pause::Full`]
/// (the OOM/emergency fallback — a real full-heap trace, never entered if the heap is sized
/// so OOM cannot fire). [`Pause::FullDefrag`] is unreachable in the RC subset, and
/// [`Pause::InitialMark`]/[`Pause::FinalMark`] are the SATB cycle-collection pauses, deferred
/// under `lxr_no_cm` (OCaml's immutable-by-default heap makes cyclic garbage rare). Discriminant
/// order matches the reference so any ordering-dependent LXR code ports unchanged.
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, NoUninit, Default)]
pub enum Pause {
    /// A whole GC in a single pause (root scan + closure + release). The OOM/emergency
    /// fallback for LXR; sized out of existence for the RQ1 measurement.
    #[default]
    Full = 1,
    /// A full-heap defragmenting pause. Unreachable in the minimal RC subset.
    FullDefrag,
    /// The steady-state reference-counting pause: process incs/decs, sweep unpromoted
    /// nursery blocks, lazily sweep newly-dead mature blocks. No tracing.
    RefCount,
    /// SATB cycle collection: the initial pause before concurrent marking (deferred).
    InitialMark,
    /// SATB cycle collection: the final pause after concurrent marking (deferred).
    FinalMark,
}

// Allow `Atomic<Option<Pause>>` (the plan stores its current/previous pause kind that way, as the
// ConcurrentImmix plan does). `bytemuck` only auto-derives `NoUninit` for the bare enum; storing
// it in an `Option` inside an `Atomic` additionally requires these two niche-optimisation marker
// impls (a `#[repr(u8)]` fieldless enum with no discriminant 0 leaves the all-zero bit pattern free
// for `None`, which both traits attest is sound).
unsafe impl bytemuck::ZeroableInOption for Pause {}
unsafe impl bytemuck::PodInOption for Pause {}
