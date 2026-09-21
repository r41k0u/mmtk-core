use crate::plan::concurrent::Pause;
use crate::plan::Plan;
use crate::scheduler::{GCWork, WorkBucketStage};
use crate::util::ObjectReference;

/// Trait for a concurrent plan.
pub trait ConcurrentPlan: Plan {
    /// Return `true`` if concurrent work (such as concurrent marking) is in progress.
    fn concurrent_work_in_progress(&self) -> bool;
    /// Return the current pause kind.  `None` if not in a pause.
    fn current_pause(&self) -> Option<Pause>;
    /// Return `true` if `object` must NOT be traced by the concurrent marker.
    ///
    /// A generational concurrent plan (Bactrian) overrides this to exclude its copying
    /// nursery: young objects move at every nursery pause, so a young reference held in
    /// a concurrent marking queue would dangle across the pause. Young objects are all
    /// allocated after the snapshot (the `InitialMark` pause empties the nursery), so
    /// skipping them is sound under SATB. Non-generational concurrent plans keep the
    /// default (`false`).
    fn should_skip_concurrent_trace(&self, _object: ObjectReference) -> bool {
        false
    }
    /// Return `true` if the current pause completes the marking cycle (`FinalMark` or a
    /// full STW GC), i.e. mark state is complete and may drive weak-reference clearing
    /// for the whole heap. Bindings use this to distinguish a mid-cycle nursery pause
    /// (stale mature marks) from the cycle-completing pause.
    fn current_pause_finishes_mark(&self) -> bool {
        matches!(
            self.current_pause(),
            Some(Pause::FinalMark) | Some(Pause::Full)
        )
    }

    /// Route a marking work packet (`ConcurrentTraceObjects` /
    /// `ProcessModBufSATB` / marking seeds). Default: park it in the
    /// `Concurrent` bucket for background execution by GC workers — the
    /// worker-concurrent design. A plan running SLICED marking (see
    /// [`Self::marking_confined_to_pauses`]) overrides this to park the packet
    /// in a plan-owned queue that is only drained in budgeted quanta inside
    /// stopped-world pauses.
    fn schedule_marking_packet(&self, w: Box<dyn GCWork<Self::VM>>) {
        self.base().scheduler.work_buckets[WorkBucketStage::Concurrent].add_boxed_no_notify(w);
    }

    /// Is the marking work queue fully drained (cycle ready for `FinalMark`)?
    /// Must agree with wherever [`Self::schedule_marking_packet`] parks work.
    fn marking_queue_drained(&self) -> bool {
        self.base().scheduler.work_buckets[WorkBucketStage::Concurrent].is_drained()
    }

    /// Return `true` if this plan executes ALL marking work inside
    /// stopped-world pauses (sliced-STW marking): no marking packet ever runs
    /// while mutators run. Bindings may use this to keep single-tracer
    /// (plain-op) tracing modes armed across the marking window — with one
    /// worker and a stopped world, the tracer is single even mid-cycle.
    fn marking_confined_to_pauses(&self) -> bool {
        false
    }

    /// Did the pause that JUST ENDED complete a marking cycle (`FinalMark` or
    /// a full STW GC)? Readable after `end_of_gc` has cleared the current
    /// pause — for end-of-collection accounting (a binding's major-cycle
    /// pacing must treat a completed concurrent cycle exactly like a full GC,
    /// as stock collectors do; without this a FinalMark never resets pacing
    /// baselines and the trigger law diverges between the STW and concurrent
    /// modes).
    fn previous_pause_finished_mark(&self) -> bool {
        false
    }

    /// Is the (incrementally executed) post-cycle SWEEP fully drained? For
    /// plans that defer the mature sweep into quanta (Bactrian), the cycle is
    /// only COMPLETE — pacing baselines valid, next cycle/full legal — once
    /// this returns true. Plans that sweep inside the pause return true.
    fn sweep_drained(&self) -> bool {
        true
    }

    /// Request a pause to PROGRESS in-flight incremental work (marking or
    /// sweep quanta) even though no nursery trigger fired — the analog of
    /// stock OCaml running a major slice off major-heap allocation. Called by
    /// bindings from mature-direct allocation paths; honored by the plan's
    /// collection_required at the next poll. Default: no-op.
    fn request_progress_pause(&self) {}

    /// Hint the per-pause mark-quantum budget for the cycle being triggered,
    /// in milliseconds — stock OCaml's mark-slice sizing law, computed by the
    /// binding's pacing at cycle-trigger time: the mark debt (post-sweep
    /// live) spread over the pauses the remaining heap runway will yield
    /// (`debt_ms / (runway / nursery)`). A fixed small budget cannot absorb a
    /// large live set inside a short runway — the un-absorbed remainder used
    /// to drain in one giant FinalMark pause. 0/never-called = the static
    /// MMTK_MARK_SLICE_MS budget. Default: no-op for plans without sliced
    /// marking.
    ///
    /// The mature Immix space's (post-sweep reserved bytes, live bytes
    /// marked by the last major epoch) — the compaction law's inputs.
    /// Immix-only on both sides so LOS residency cannot skew the ratio.
    /// None = plan doesn't support the law. Default: None.
    fn mature_footprint_and_live(&self) -> Option<(usize, usize)> {
        None
    }

    /// Request a compacting major: the next STW Full evacuates EVERY in-use
    /// mature block (bounded by copy headroom — leftovers stay in place and
    /// later compactions converge). Called by the binding's pacing when
    /// mature reserved pages run away from its live estimate — the
    /// line-granular reclamation cannot free 256B lines that interleave
    /// small dead objects with live ones, so byte-level waste is invisible
    /// to both the normal defrag trigger and its hole-bucket candidate
    /// selection (mature_mutation: 8MB live pinning >90MB). Stock OCaml's
    /// analog is `Gc.max_overhead`-paced automatic compaction. Default:
    /// no-op.
    fn request_mature_compaction(&self) {}

    /// `tick_origin` says WHICH pacing site fired: `false` = the post-minor
    /// path (pause cadence = minors — a big-nursery config's minors are
    /// promotion-bound and dwarf any quantum, so slicing is pointless there:
    /// the plan's feasibility gate degrades the cycle to a monolithic Full);
    /// `true` = the mature-direct allocation tick (pause cadence = tick
    /// batches — near-empty nursery collections that stay small at ANY
    /// nursery cap, so the nursery gate must not apply; fragmed flipped from
    /// cycles to 11 monolithic Fulls, D1 3.4→5.0×, when it did).
    fn set_mark_quantum_hint_ms(&self, _ms: f64, _debt_ms: f64, _tick_origin: bool) {}

    /// Did the pause that JUST ENDED start a marking cycle (`InitialMark`, or
    /// a full STW GC — which is a whole cycle in one pause)? For
    /// allocation-denominated cycle pacing: stock-style pacing measures the
    /// budget from cycle START to next cycle start, so allocation during the
    /// marking window counts toward the next trigger.
    fn previous_pause_started_cycle(&self) -> bool {
        false
    }
}
