//! Bactrian: the faithful MMTk realization of OCaml 5's garbage collector.
//!
//! Named for the two-humped camel (OCaml's mascot): the plan has two humps —
//! a copying nursery (from the generational plans' `CommonGenPlan`) and a
//! concurrently-marked, STW-evacuated Immix mature space (from ConcurrentImmix's
//! SATB machinery). Together: generational, copying minor, mostly-concurrent
//! and (near-)non-moving major, SATB deletion barrier, no read barrier — the
//! collector architecture of stock OCaml 5, expressed as an MMTk plan.
//!
//! Purpose (ocaml-mmtk RQ7): stock OCaml and MMTk-Bactrian run the *same* GC
//! algorithm, so comparing them measures pure framework/implementation overhead
//! rather than collector-design differences.
//!
//! # Pauses
//!
//! Every Bactrian pause except `Full` is anchored on a nursery collection,
//! mirroring stock OCaml (whose major-cycle phase changes ride on STW sections
//! that empty the minor heaps):
//!
//! - `Pause::Nursery`: a plain generational minor collection (GenImmix's
//!   nursery GC). May run while concurrent marking is in progress.
//! - `Pause::InitialMark`: a nursery collection *fused with* the marking
//!   snapshot: the nursery trace additionally seeds the concurrent marking
//!   queue with every mature object it touches (root targets and the mature
//!   children of promoted objects). Emptying the nursery here puts all
//!   young-held references into the snapshot, which is what makes it sound for
//!   the concurrent marker to skip young objects entirely.
//! - `Pause::FinalMark`: a nursery collection that also drains the remaining
//!   SATB/marking work, runs weak-reference/finalizer processing over the
//!   completed mark state, and sweeps the mature space.
//! - `Pause::Full`: GenImmix's STW full-heap collection (may defrag the
//!   mature Immix space). Used for user-forced GCs and emergencies; never
//!   scheduled while a marking cycle is in flight.
//!
//! # Barrier
//!
//! `barrier::BactrianBarrier` combines the generational post-write remembering
//! barrier (unchanged from GenImmix) with a slot-granularity SATB deletion
//! barrier that is active only while marking is in progress and filters young
//! referents — exactly stock OCaml's `caml_darken(old)` in `caml_modify`, which
//! also has no per-object dedup bit and ignores young values. This division of
//! labour leaves the per-object unlog bit exclusively owned by the generational
//! barrier, dissolving the metadata conflict between the two protocols.
//!
//! # Soundness of the composition (young objects vs SATB)
//!
//! The nursery is emptied at `InitialMark`, so every young object during a
//! marking cycle is post-snapshot. Post-snapshot objects need no marking (SATB
//! collects the snapshot plus everything allocated since), so the marker and
//! the SATB barrier both skip young references. Objects promoted mid-cycle are
//! born black: `ImmixSpace::post_copy` sets the object mark bit, and
//! `allocate_as_live` (set during marking) makes the copy allocator eagerly
//! mark the lines it acquires, so mid-cycle promotions survive the FinalMark
//! sweep without being scanned — their fields are covered by the standard SATB
//! new-object induction.

pub(in crate::plan) mod barrier;
pub(in crate::plan) mod gc_work;
pub(in crate::plan) mod global;
pub(in crate::plan) mod mutator;

pub use global::Bactrian;
