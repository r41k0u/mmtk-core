use crate::plan::concurrent::bactrian::gc_work::BactrianNurseryGCWorkContext;
use crate::plan::concurrent::bactrian::gc_work::BactrianSTWGCWorkContext;
use crate::plan::concurrent::global::ConcurrentPlan;
use crate::plan::concurrent::Pause;
use crate::plan::generational::global::CommonGenPlan;
use crate::plan::generational::global::GenerationalPlan;
use crate::plan::global::BasePlan;
use crate::plan::global::CommonPlan;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::AllocationSemantics;
use crate::plan::Plan;
use crate::plan::PlanConstraints;
use crate::policy::copyspace::CopySpace;
use crate::policy::gc_work::TraceKind;
use crate::policy::immix::defrag::StatsForDefrag;
use crate::policy::immix::ImmixSpace;
use crate::policy::immix::ImmixSpaceArgs;
use crate::policy::immix::{TRACE_KIND_DEFRAG, TRACE_KIND_FAST};
use crate::policy::space::Space;
use crate::plan::concurrent::bactrian::gc_work::BactrianMarkQuantum;
use crate::plan::concurrent::bactrian::gc_work::BactrianSweepQuantum;
use crate::scheduler::GCWork;
use crate::scheduler::GCWorkScheduler;
use crate::scheduler::GCWorker;
use crate::scheduler::WorkBucketStage;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::copy::*;
use crate::util::heap::gc_trigger::SpaceStats;
use crate::util::heap::VMRequest;
use crate::util::metadata::log_bit::UnlogBitsOperation;
use crate::util::Address;
use crate::util::ObjectReference;
use crate::util::VMWorkerThread;
use crate::vm::ObjectModel;
use crate::vm::VMBinding;
use crate::ObjectQueue;

use atomic::Atomic;
use enum_map::EnumMap;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use mmtk_macros::{HasSpaces, PlanTraceObject};

/// Bactrian: copying nursery + concurrently-marked, STW-evacuated Immix mature space
/// with an SATB deletion barrier — the faithful MMTk realization of OCaml 5's GC.
/// See the module documentation for the design.
#[derive(HasSpaces, PlanTraceObject)]
pub struct Bactrian<VM: VMBinding> {
    /// Generational plan (the copying nursery + common spaces).
    #[parent]
    pub gen: CommonGenPlan<VM>,
    /// The mature space: concurrently marked, evacuated only at STW `Full` pauses.
    #[post_scan]
    #[space]
    #[copy_semantics(CopySemantics::Mature)]
    pub immix_space: ImmixSpace<VM>,
    /// Survivor-aging semispace pair (MMTK_NURSERY_AGE >= 1): nursery survivors
    /// of a plain minor are copied here (staying YOUNG — one extra minor to die)
    /// instead of being promoted; the pair flips each aging minor and the old
    /// to-space's residents (age 1) promote to mature. Both spaces are part of
    /// the young generation for every barrier/SATB/marking young-check (see
    /// is_object_in_nursery). Empty and inert when aging is off (the default).
    /// Full-heap traces evacuate them to mature via the derive attribute below.
    #[space]
    #[copy_semantics(CopySemantics::PromoteToMature)]
    pub aged0: CopySpace<VM>,
    #[space]
    #[copy_semantics(CopySemantics::PromoteToMature)]
    pub aged1: CopySpace<VM>,
    /// Which aged space is the current to-space (mirrors GenCopy's `hi`).
    aged_hi: AtomicBool,
    /// Latched per pause in prepare(): true only for a plain Nursery pause with
    /// aging enabled and no concurrent marking active. Aging MUST be off in any
    /// marking-fused pause and during marking: the SATB barrier skips young
    /// objects, which is sound only because no young object survives a marking
    /// snapshot — a survivor kept young across the snapshot could hold the only
    /// (unlogged) edge to a mature object and the marker would miss it.
    aging_this_gc: AtomicBool,
    /// Whether the last GC was a defrag GC for the immix space.
    last_gc_was_defrag: AtomicBool,
    current_pause: Atomic<Option<Pause>>,
    /// Sliced-STW marking (default ON; MMTK_MARK_SLICED=0 reverts to the
    /// worker-concurrent design): ALL marking work parks in
    /// `parked_marking` and is drained in budgeted quanta inside nursery
    /// pauses — stock OCaml's mutator mark slices, executed as short
    /// stop-the-world quanta on the worker. Rationale (SHAPE.md round 25):
    /// worker-concurrent marking at 1 mutator costs more in cross-core LLC
    /// interference than it saves in pause time (bt@2M: mutator 5.8G ->
    /// 13.7G cycles), while STW fulls cost the D3 tail (78-122ms pauses vs
    /// vanilla's 15ms max). Sliced quanta keep both: no simultaneity, no
    /// tail, and single-tracer (UP) economics stay armed.
    sliced_marking: bool,
    /// Parked marking packets for sliced mode (ConcurrentTraceObjects,
    /// ProcessModBufSATB). Mutator-side SATB flushes push here between
    /// pauses; quanta pop inside pauses. MPMC-safe.
    parked_marking: crossbeam::deque::Injector<Box<dyn GCWork<VM>>>,
    /// INCREMENTAL SWEEP (sliced mode): FinalMark defers its chunk-sweep
    /// packets here instead of running them inside the pause — stock OCaml's
    /// sweep slices. Budgeted BactrianSweepQuantum packets drain them in the
    /// Release stage of subsequent nursery pauses. The next cycle/Full is
    /// gated on drain completion (decide_pause), because the packets read
    /// this cycle's line_mark_state/defrag histograms and the chunk map.
    parked_sweep: crossbeam::deque::Injector<Box<dyn GCWork<VM>>>,
    /// True from FinalMark's release until the last deferred sweep packet
    /// has run. Read by decide_pause gating and the binding's pacing
    /// (ConcurrentPlan::sweep_drained).
    sweep_pending: AtomicBool,
    /// Set by request_progress_pause (mature-direct allocation wants the
    /// in-flight quanta to advance); consumed by collection_required.
    progress_pause_requested: AtomicBool,
    /// Per-pause mark-quantum budget hint in nanoseconds, set by the
    /// binding's pacing at cycle-trigger time (ConcurrentPlan::
    /// set_mark_quantum_hint_ms — stock's slice-sizing law: mark debt over
    /// runway pauses). 0 = use the static MMTK_MARK_SLICE_MS budget.
    pub(in crate::plan) mark_quantum_hint_nanos: AtomicU64,
    /// Did the mature-direct allocation TICK fire the pending cycle (vs the
    /// post-minor path)? Tick-paced cycles progress in near-empty nursery
    /// pauses that stay small at any nursery cap, so the feasibility
    /// escape's nursery gate must not degrade them to monolithic Fulls.
    cycle_tick_origin: AtomicBool,
    /// Pending mature-compaction request (ConcurrentPlan::
    /// request_mature_compaction — the binding's reserved-vs-live runaway
    /// law). Consumed by decide_pause: rides the next major as a COMPACT-ALL
    /// Full (cycles never defragment).
    compact_requested: AtomicBool,
    previous_pause: Atomic<Option<Pause>>,
    concurrent_marking_active: AtomicBool,
}

/// The plan constraints for the Bactrian plan.
pub const BACTRIAN_CONSTRAINTS: PlanConstraints = PlanConstraints {
    // Copying nursery always moves; the mature space moves only at STW Full pauses.
    moves_objects: true,
    // Nursery promotion copies into the mature Immix space, so nursery objects must
    // also respect the max immix object size (same reasoning as GenImmix).
    max_non_los_default_alloc_bytes: crate::util::rust_util::min_of_usize(
        crate::policy::immix::MAX_IMMIX_OBJECT_SIZE,
        crate::plan::plan_constraints::MAX_NON_LOS_ALLOC_BYTES_COPYING_PLAN,
    ),
    generational: true,
    // The unlog bit is owned exclusively by the generational (object/region
    // remembering) half of the barrier; the SATB half is slot-granular and bit-free
    // (like stock OCaml's deletion barrier).
    needs_log_bit: true,
    // The barrier selector is an indicator for VM fast paths; Bactrian's combined
    // barrier subsumes SATB (pre) + object remembering (post).
    barrier: crate::BarrierSelector::SATBBarrier,
    // The object-remembering half may enqueue the same slot/object more than once.
    may_trace_duplicate_edges: true,
    needs_prepare_mutator: true,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for Bactrian<VM> {
    fn constraints(&self) -> &'static PlanConstraints {
        &BACTRIAN_CONSTRAINTS
    }

    fn create_copy_config(&'static self) -> CopyConfig<Self::VM> {
        use enum_map::enum_map;
        CopyConfig {
            copy_mapping: enum_map! {
                CopySemantics::PromoteToMature => CopySelector::ImmixHybrid(0),
                CopySemantics::Mature => CopySelector::ImmixHybrid(0),
                // Young-to-young survivor copies (aging minors only).
                CopySemantics::Nursery => CopySelector::CopySpace(0),
                _ => CopySelector::Unused,
            },
            space_mapping: vec![
                (CopySelector::ImmixHybrid(0), &self.immix_space),
                // Rebound to the current aged to-space in prepare_worker.
                (CopySelector::CopySpace(0), &self.aged0),
            ],
            constraints: &BACTRIAN_CONSTRAINTS,
        }
    }

    fn prepare_worker(&self, worker: &mut GCWorker<Self::VM>) {
        // Keep the CopySpace copy context bound to the current aged to-space.
        unsafe { worker.get_copy_context_mut().copy[0].assume_init_mut() }
            .rebind(self.aged_to());
    }

    fn collection_required(&self, space_full: bool, space: Option<SpaceStats<Self::VM>>) -> bool
    where
        Self: Sized,
    {
        // Concurrent marking finished all its work: transition to FinalMark at the
        // next poll site. (The GC-worker side self-trigger in the scheduler covers
        // the case where no mutator polls; see Scheduler::concurrent_marking_drained.)
        if self.concurrent_marking_in_progress() && self.marking_queue_drained() {
            return true;
        }
        // POLL-TIME cycle request (fragmed's discovery, SHAPE round 30): a
        // pretenure-heavy workload allocates mature-direct with few minors,
        // and the binding's mature-pressure law only ran POST-MINOR — mature
        // grew to the space-full edge (187 of 192 MB) before any cycle fired,
        // where the calibrated law wanted one at baseline x 1.14. The binding
        // sets next_gc_full_heap from its pacing; honor it at poll time so a
        // mature-allocating mutator starts the cycle without waiting for a
        // minor. (Not while a cycle is in flight or sweep is draining —
        // decide_pause would degrade it anyway.)
        if !self.concurrent_marking_in_progress()
            && self.sweep_drained()
            && self.major_request_pending()
        {
            return true;
        }
        // Progress pause for in-flight quanta (fragmed part 2, SHAPE round
        // 30): a mature-direct workload opens a cycle but never minors, so
        // marking/sweep quanta — scheduled only in nursery-class pauses —
        // never run and the cycle floats forever (OOM at 300 waves). Stock
        // paces its slices off major-heap allocation; the binding's
        // mature-alloc tick requests the same via request_progress_pause.
        if self.progress_pause_requested.swap(false, Ordering::SeqCst) {
            return true;
        }
        self.gen.collection_required(self, space_full, space)
    }

    fn last_collection_was_exhaustive(&self) -> bool {
        self.previous_pause() == Some(Pause::Full)
            && self
                .immix_space
                .is_last_gc_exhaustive(self.last_gc_was_defrag.load(Ordering::Relaxed))
    }

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<Self::VM>) {
        let pause = self.decide_pause();
        self.trace_pause("schedule", pause);
        self.current_pause.store(Some(pause), Ordering::SeqCst);
        // `gc_full_heap` drives `is_current_gc_nursery()`: every pause except Full is
        // a nursery-collecting pause (ProcessModBuf/weak-processing rely on this).
        self.gen
            .gc_full_heap
            .store(pause == Pause::Full, Ordering::SeqCst);

        probe!(mmtk, concurrent_pause_determined, pause as usize);

        match pause {
            Pause::Full => {
                crate::plan::immix::global::Immix::schedule_immix_full_heap_collection::<
                    Bactrian<VM>,
                    BactrianSTWGCWorkContext<VM, TRACE_KIND_FAST>,
                    BactrianSTWGCWorkContext<VM, TRACE_KIND_DEFRAG>,
                >(self, &self.immix_space, scheduler);
            }
            Pause::InitialMark | Pause::FinalMark | Pause::Nursery => {
                scheduler.schedule_common_work::<BactrianNurseryGCWorkContext<VM>>(self);
                if self.sliced_marking {
                    match pause {
                        // Mid-cycle nursery pause: drain a bounded mark quantum
                        // AFTER the nursery closure (Release opens once Closure
                        // and weak processing are done). UNBUDGETED on a
                        // genuine emergency (allocation failed mid-cycle: the
                        // runway is gone, so marking must complete now — the
                        // next pause is then FinalMark and its sweep frees the
                        // backlog; a budgeted drain would loop failing polls).
                        Pause::Nursery if self.concurrent_marking_in_progress() => {
                            let emergency = self.genuine_allocation_emergency();
                            let w = if emergency {
                                BactrianMarkQuantum::unbudgeted(self)
                            } else {
                                BactrianMarkQuantum::budgeted(self)
                            };
                            scheduler.work_buckets[WorkBucketStage::Release].add(w);
                        }
                        // FinalMark: drain EVERYTHING parked, unbudgeted, inside
                        // the Closure stage — mutators are stopped and their
                        // late SATB flushes (parked during StopMutators) are all
                        // in by the time Closure opens.
                        Pause::FinalMark => {
                            scheduler.work_buckets[WorkBucketStage::Closure]
                                .add(BactrianMarkQuantum::unbudgeted(self));
                        }
                        _ => {}
                    }
                    // Incremental sweep: one quantum per nursery-class pause
                    // while packets remain. Budgeted normally; UNBUDGETED when
                    // the pacing already wants the next cycle (next_gc_full_heap
                    // is pending) so the cycle isn't held up by more than one
                    // minor. Runs in Release, after this pause's own release
                    // parked FinalMark's packets — so the first quantum can run
                    // in the FinalMark pause itself if budget allows.
                    // NOTE: FinalMark's own first quantum is scheduled from
                    // the RELEASE arm, strictly AFTER the packets are parked —
                    // scheduling it here raced the parking at T>1 (worker A's
                    // quantum popped an empty queue mid-parking, declared the
                    // sweep complete and disarmed allocate-as-live while
                    // worker B was still parking; later real sweeps then freed
                    // live pretenured blocks — the fragmed T4 corruption).
                    if pause != Pause::FinalMark && self.sweep_pending.load(Ordering::SeqCst) {
                        // Unbudgeted ONLY on genuine emergency (allocation
                        // failed: the degraded-from-Full pause must free the
                        // whole backlog now or the retry loop livelocks). A
                        // merely-pending cycle request WAITS on the budgeted
                        // drain — vanilla's cycles likewise wait out the
                        // previous sweep, and an eager drain-all doubled the
                        // minor pause max (8.2 -> 14.4ms measured at bt@2M).
                        let emergency = self.genuine_allocation_emergency();
                        let w = if emergency {
                            BactrianSweepQuantum::unbudgeted(self)
                        } else {
                            BactrianSweepQuantum::budgeted(self)
                        };
                        scheduler.work_buckets[WorkBucketStage::Release].add(w);
                    }
                }
            }
        }
    }

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &crate::plan::generational::ALLOCATOR_MAPPING
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        let pause = self.current_pause().unwrap();
        match pause {
            Pause::Full => {
                // GenImmix's full-heap protocol: bulk-clear unlog bits; the full trace
                // reconstructs them (post_copy / unlog_object_if_needed).
                self.gen.prepare(tls);
                self.immix_space.prepare(
                    true,
                    Some(StatsForDefrag::new(self)),
                    UnlogBitsOperation::BulkClear,
                );
                // Whole young generation (incl. aged survivors) evacuates to
                // mature at a Full pause.
                self.aging_this_gc.store(false, Ordering::SeqCst);
                self.prepare_aged_all_from();
            }
            Pause::InitialMark => {
                // A nursery collection fused with the start of a marking cycle. The
                // nursery prepares as in a minor GC, while the mature/common spaces
                // prepare for a new (full) mark cycle. Unlike ConcurrentImmix we do
                // NOT bulk-set unlog bits: the SATB barrier is slot-granular and
                // bit-free; the unlog bit stays owned by the generational barrier.
                self.gen.full_heap_gc_count.lock().unwrap().inc();
                self.gen.nursery.prepare(true);
                self.gen
                    .nursery
                    .set_copy_for_sft_trace(Some(CopySemantics::PromoteToMature));
                // The common (full-flagged) prepare zeroes the nonmoving
                // mark-sweep space's bits for the cycle. This pause's own
                // release must NOT consume that incomplete state — and does
                // not: InitialMark is nursery-flagged (gc_full_heap is Full-
                // only), so gen.release passes full=false and the round-31
                // gate in release_nonmoving_space skips the space. Marks
                // complete at FinalMark, whose release (explicit full=true)
                // sweeps it.
                self.gen.common.prepare(tls, true);
                self.immix_space.prepare(
                    true,
                    Some(StatsForDefrag::new(self)),
                    UnlogBitsOperation::NoOp,
                );
                // The marking cycle starts NOW, not at end_of_gc: this pause's own
                // Closure stage promotes the whole live nursery into the (just
                // prepared) mature space, and those promotions must be born live —
                // post_copy gives them the new cycle's object mark bit, but with
                // MARK_LINE_AT_SCAN_TIME their LINES are only marked by the eager
                // allocate_as_live path. Arming it here (Prepare stage, strictly
                // before Closure opens) makes InitialMark promotions survive the
                // FinalMark sweep. (ConcurrentImmix flips this at end_of_gc, but it
                // has no in-pause promotions.)
                self.set_concurrent_marking_state(true);
                // No aging in a marking-fused pause (see aging_this_gc docs):
                // the whole young generation promotes so the snapshot holds.
                self.aging_this_gc.store(false, Ordering::SeqCst);
                self.prepare_aged_all_from();
            }
            Pause::Nursery => {
                // Plain minor collection (GenImmix's nursery prepare).
                self.gen.prepare(tls);
                let aging = nursery_age() >= 1 && !self.concurrent_marking_in_progress();
                self.aging_this_gc.store(aging, Ordering::SeqCst);
                if aging {
                    // Flip: last aging minor's to-space (age-1 survivors)
                    // becomes this minor's from-space and promotes to mature;
                    // this minor's nursery survivors copy young into the new
                    // to-space.
                    self.aged_hi
                        .store(!self.aged_hi.load(Ordering::SeqCst), Ordering::SeqCst);
                    let hi = self.aged_hi.load(Ordering::SeqCst);
                    self.aged0.prepare(hi);
                    self.aged1.prepare(!hi);
                    self.gen
                        .nursery
                        .set_copy_for_sft_trace(Some(CopySemantics::Nursery));
                    self.aged_from_mut()
                        .set_copy_for_sft_trace(Some(CopySemantics::PromoteToMature));
                    self.aged_to_mut().set_copy_for_sft_trace(None);
                } else {
                    // Aging off (or marking active): evacuate any aged residue
                    // to mature alongside the nursery, exactly as before.
                    self.prepare_aged_all_from();
                }
            }
            Pause::FinalMark => {
                // A nursery collection that completes the marking cycle. Only the
                // nursery needs preparing; the mature/common spaces were prepared at
                // InitialMark and have been marked concurrently since.
                self.gen.nursery.prepare(true);
                self.gen
                    .nursery
                    .set_copy_for_sft_trace(Some(CopySemantics::PromoteToMature));
                self.aging_this_gc.store(false, Ordering::SeqCst);
                self.prepare_aged_all_from();
            }
        }
    }

    fn release(&mut self, tls: VMWorkerThread) {
        let pause = self.current_pause().unwrap();
        match pause {
            Pause::Full => {
                self.gen.release(tls);
                // Unlog bits were reconstructed during tracing; keep them.
                self.immix_space.release(true, UnlogBitsOperation::NoOp);
                self.aged0.release();
                self.aged1.release();
            }
            Pause::InitialMark => {
                // Minor collection: release the nursery (and common spaces at nursery
                // level). The mature space is untouched — its sweep happens at
                // FinalMark, after marking completes. Both aged spaces were
                // from-spaces (whole young gen promoted for the snapshot).
                self.gen.release(tls);
                self.aged0.release();
                self.aged1.release();
            }
            Pause::Nursery => {
                self.gen.release(tls);
                if self.aging_this_gc.load(Ordering::SeqCst) {
                    // Only the from side was evacuated; the to side holds this
                    // minor's still-young survivors.
                    self.aged_from_mut().release();
                } else {
                    self.aged0.release();
                    self.aged1.release();
                }
            }
            Pause::FinalMark => {
                // Nursery release + mature/common sweep over the completed mark state.
                // Unlog bits stay owned by the generational protocol: remembered
                // objects were re-unlogged by ProcessModBuf during this (nursery)
                // pause; everything else is untouched.
                self.gen.nursery.release();
                self.gen.common.release(tls, true);
                if self.sliced_marking {
                    // INCREMENTAL SWEEP: park the chunk-sweep packets; budgeted
                    // quanta drain them across subsequent nursery pauses (stock's
                    // sweep slices). decide_pause gates the next cycle/Full on
                    // drain completion. LOS/common were swept above (in-pause):
                    // only the mature Immix sweep is bulky enough to slice.
                    let packets = self
                        .immix_space
                        .release_deferred_sweep(true, UnlogBitsOperation::NoOp);
                    for w in packets {
                        self.parked_sweep.push(w);
                    }
                    self.sweep_pending.store(true, Ordering::SeqCst);
                    // First quantum, scheduled only now — after parking and
                    // the pending flag are fully published (see the
                    // schedule_collection note). The plan is 'static in
                    // reality (standard mmtk pattern; see ImmixSpace::release's
                    // identical self-reference).
                    let plan: &'static Self = unsafe { &*(self as *const Self) };
                    self.gen.common.base.scheduler.work_buckets[WorkBucketStage::Release]
                        .add(BactrianSweepQuantum::budgeted(plan));
                } else {
                    self.immix_space.release(true, UnlogBitsOperation::NoOp);
                }
                self.aged0.release();
                self.aged1.release();
            }
        }
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        let pause = self.current_pause().unwrap();

        let next_gc_full_heap = CommonGenPlan::should_next_gc_be_full_heap(self);
        self.gen.end_of_gc(tls, next_gc_full_heap);

        let did_defrag = self.immix_space.end_of_gc();
        self.last_gc_was_defrag.store(did_defrag, Ordering::Relaxed);

        match pause {
            Pause::InitialMark => {
                // Marking state was already armed in prepare() (this pause's own
                // promotions must be born live); nothing further to do here.
                debug_assert!(self.concurrent_marking_in_progress());
            }
            Pause::FinalMark => {
                // End the MARKING half of the cycle, but under INCREMENTAL
                // SWEEP keep allocate-as-live armed until the deferred sweep
                // drains: a pretenured (mature-direct) object born after this
                // pause into a freshly-acquired block has ZEROED line marks,
                // and a deferred SweepChunk visiting its chunk would free the
                // live block (the fragmed corruption, SHAPE round 30 — T1 and
                // T4, pretenure+sliced only). The sweep quantum disarms it at
                // drain completion (allocate_as_live_until_swept). The
                // marking flag itself must clear NOW (decide_pause,
                // barrier state).
                self.concurrent_marking_active
                    .store(false, Ordering::SeqCst);
                if self.sliced_marking && self.sweep_pending.load(Ordering::SeqCst) {
                    // spaces stay in allocate-as-live mode
                } else {
                    use crate::plan::global::HasSpaces;
                    self.for_each_space(&mut |space: &dyn Space<VM>| {
                        space.set_allocate_as_live(false);
                    });
                }
            }
            Pause::Full | Pause::Nursery => (),
        }

        self.previous_pause.store(Some(pause), Ordering::SeqCst);
        self.current_pause.store(None, Ordering::SeqCst);
        self.trace_pause("end", pause);
        info!("{:?} end", pause);
    }

    fn current_gc_may_move_object(&self) -> bool {
        // Every pause moves young objects (nursery evacuation); Full may also defrag.
        true
    }

    fn get_collection_reserved_pages(&self) -> usize {
        // Aged residents will be copied (to mature) at the next collection;
        // reserve for them like the nursery's own copy reserve.
        self.gen.get_collection_reserved_pages()
            + self.aged0.reserved_pages()
            + self.aged1.reserved_pages()
            + self.immix_space.defrag_headroom_pages()
    }

    fn get_used_pages(&self) -> usize {
        self.gen.get_used_pages()
            + self.aged0.reserved_pages()
            + self.aged1.reserved_pages()
            + self.immix_space.reserved_pages()
    }

    /// Return the number of pages available for allocation. Assuming all future
    /// allocations go to the nursery (same as GenImmix).
    fn get_available_pages(&self) -> usize {
        (self
            .get_total_pages()
            .saturating_sub(self.get_reserved_pages()))
            >> 1
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.gen.common.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.gen.common.base
    }

    fn common(&self) -> &CommonPlan<VM> {
        &self.gen.common
    }

    fn generational(&self) -> Option<&dyn GenerationalPlan<VM = VM>> {
        Some(self)
    }

    fn concurrent(&self) -> Option<&dyn ConcurrentPlan<VM = VM>> {
        Some(self)
    }

    fn notify_mutators_paused(&self, _scheduler: &GCWorkScheduler<VM>) {
        let pause = self.current_pause().unwrap();
        match pause {
            Pause::Full | Pause::Nursery => {
                debug_assert!(
                    pause == Pause::Nursery || !self.concurrent_marking_in_progress(),
                    "Full pause scheduled while marking is in progress"
                );
            }
            Pause::InitialMark => {
                debug_assert!(
                    !self.concurrent_marking_in_progress(),
                    "prev pause: {:?}",
                    self.previous_pause()
                );
            }
            Pause::FinalMark => {
                debug_assert!(self.concurrent_marking_in_progress());
                // Mutator SATB buffers were flushed by StopMutators (flush_mutator).
                // Marking stays "active" through this pause: the SATB packets flushed
                // above still route to the (open) Concurrent bucket, which drains
                // before the STW stages advance, and promotions in this pause must be
                // born live. The state is cleared in end_of_gc, after the sweep.
            }
        }
        info!("{:?} start", pause);
    }
}

impl<VM: VMBinding> GenerationalPlan for Bactrian<VM> {
    fn is_current_gc_nursery(&self) -> bool {
        self.gen.is_current_gc_nursery()
    }

    fn is_object_in_nursery(&self, object: ObjectReference) -> bool {
        // The aged pair and young-LOS objects are part of the YOUNG
        // generation: every barrier, SATB young-drop, and concurrent-marking
        // skip routes through this check.
        self.gen.nursery.in_space(object)
            || self.aged0.in_space(object)
            || self.aged1.in_space(object)
            || (self.gen.common.los.in_space(object)
                && self.gen.common.los.is_in_nursery(object))
    }

    fn is_address_in_nursery(&self, addr: Address) -> bool {
        self.gen.nursery.address_in_space(addr)
            || self.aged0.address_in_space(addr)
            || self.aged1.address_in_space(addr)
    }

    fn get_mature_physical_pages_available(&self) -> usize {
        self.immix_space.available_physical_pages()
    }

    fn get_mature_reserved_pages(&self) -> usize {
        // The pretenured medium band lives in the common nonmoving
        // (free-list mark-sweep) space by default (round 30,
        // MMTK_MEDIUM_TO) — it is mature and must be visible to the
        // binding's pressure/cadence pacing, or a band-heavy workload
        // (fragmed) never triggers the majors whose sweeps feed its free
        // lists and runs to the space-full edge (203MB RSS, 1 GC).
        // The LOS is mature for the same reason: dead large objects are
        // only swept (and their pages only returned) at a full, so LOS
        // churn that the pressure law cannot see accumulates until the
        // allocation-cadence backstop — a large-int workload with a ~10MB
        // live set peaked at 585MB, 532MB of it dead LOS pages pooled
        // between backstop-paced fulls.
        self.immix_space.reserved_pages()
            + self.gen.common.get_nonmoving().reserved_pages()
            + self.gen.common.get_los().reserved_pages()
    }

    fn force_full_heap_collection(&self) {
        self.gen.force_full_heap_collection()
    }

    fn nursery_keeps_movable_survivors(&self) -> bool {
        self.aging_this_gc.load(Ordering::SeqCst)
    }

    /// For Bactrian, a "full heap collection" in the generational sense is any pause
    /// that completed a whole-heap reclamation: a STW `Full` GC *or* the `FinalMark`
    /// of a concurrent cycle (which swept the mature space over complete marks).
    /// The binding's mature-pressure pacing and `Gc.major_collections` accounting key
    /// on this — a completed concurrent cycle counts as a major collection, exactly
    /// as in stock OCaml.
    fn last_collection_full_heap(&self) -> bool {
        matches!(
            self.previous_pause(),
            Some(Pause::Full) | Some(Pause::FinalMark)
        )
    }
}

impl<VM: VMBinding> crate::plan::generational::global::GenerationalPlanExt<VM> for Bactrian<VM> {
    fn trace_object_nursery<Q: ObjectQueue, const KIND: TraceKind>(
        &self,
        queue: &mut Q,
        object: ObjectReference,
        worker: &mut GCWorker<VM>,
    ) -> ObjectReference {
        assert!(
            KIND != crate::policy::gc_work::TRACE_KIND_TRANSITIVE_PIN,
            "A copying nursery cannot pin objects"
        );
        // Nursery proper: survivors stay YOUNG (copy to the aged to-space) on
        // an aging minor, else promote to mature as before.
        if self.gen.nursery.in_space(object) {
            let semantics = if self.aging_this_gc.load(Ordering::Relaxed) {
                CopySemantics::Nursery
            } else {
                CopySemantics::PromoteToMature
            };
            return self
                .gen
                .nursery
                .trace_object::<Q>(queue, object, Some(semantics), worker);
        }
        // Aged pair: from-space residents (age 1) promote to mature; to-space
        // objects were copied this GC and CopySpace::trace_object returns them
        // unchanged (its !is_from_space early exit).
        if self.aged0.in_space(object) {
            return self.aged0.trace_object::<Q>(
                queue,
                object,
                Some(CopySemantics::PromoteToMature),
                worker,
            );
        }
        if self.aged1.in_space(object) {
            return self.aged1.trace_object::<Q>(
                queue,
                object,
                Some(CopySemantics::PromoteToMature),
                worker,
            );
        }
        // Large objects allocated young live in the LOS.
        if self.gen.common.get_los().in_space(object) {
            return self.gen.common.get_los().trace_object::<Q>(queue, object);
        }
        object
    }
}

impl<VM: VMBinding> ConcurrentPlan for Bactrian<VM> {
    fn current_pause(&self) -> Option<Pause> {
        self.current_pause.load(Ordering::SeqCst)
    }

    fn concurrent_work_in_progress(&self) -> bool {
        self.concurrent_marking_in_progress()
    }

    fn should_skip_concurrent_trace(&self, object: ObjectReference) -> bool {
        self.is_object_in_nursery(object)
    }

    fn schedule_marking_packet(&self, w: Box<dyn GCWork<VM>>) {
        if self.sliced_marking {
            self.parked_marking.push(w);
        } else {
            self.gen.common.base.scheduler.work_buckets[WorkBucketStage::Concurrent]
                .add_boxed_no_notify(w);
        }
    }

    fn marking_queue_drained(&self) -> bool {
        if self.sliced_marking {
            self.parked_marking.is_empty()
        } else {
            self.gen.common.base.scheduler.work_buckets[WorkBucketStage::Concurrent].is_drained()
        }
    }

    fn marking_confined_to_pauses(&self) -> bool {
        self.sliced_marking
    }

    fn sweep_drained(&self) -> bool {
        !self.sweep_pending.load(Ordering::SeqCst)
    }

    fn request_mature_compaction(&self) {
        self.compact_requested.store(true, Ordering::SeqCst);
    }

    fn mature_footprint_and_live(&self) -> Option<(usize, usize)> {
        let pg = crate::util::constants::BYTES_IN_PAGE;
        Some((
            self.immix_space.reserved_pages() * pg,
            self.immix_space.major_live_bytes(),
        ))
    }

    fn set_mark_quantum_hint_ms(&self, ms: f64, tick_origin: bool) {
        let ns = (ms.max(0.0) * 1e6) as u64;
        self.mark_quantum_hint_nanos.store(ns, Ordering::Relaxed);
        self.cycle_tick_origin.store(tick_origin, Ordering::Relaxed);
    }

    fn request_progress_pause(&self) {
        // Only meaningful with in-flight incremental work; the flag is
        // consumed (or discarded) at the next allocation poll.
        if self.concurrent_marking_in_progress() || self.sweep_pending.load(Ordering::SeqCst) {
            self.progress_pause_requested.store(true, Ordering::SeqCst);
        }
    }

    fn previous_pause_finished_mark(&self) -> bool {
        matches!(
            self.previous_pause(),
            Some(Pause::FinalMark) | Some(Pause::Full)
        )
    }

    fn previous_pause_started_cycle(&self) -> bool {
        matches!(
            self.previous_pause(),
            Some(Pause::InitialMark) | Some(Pause::Full)
        )
    }
}

/// Auto-compaction threshold: percentage of live blocks that may be
/// partially occupied before a compacting Full is requested. 0 disables
/// (the default pending calibration — see SHAPE round 29+).
fn compact_util_threshold_pct() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MMTK_COMPACT_UTIL_PCT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|p| *p <= 100)
            .unwrap_or(0)
    })
}

impl<VM: VMBinding> Bactrian<VM> {
    /// Is survivor aging configured? (The oldify fast path must decline when
    /// it is: young survivors take the aged copy path it doesn't implement.)
    pub(super) fn aging_enabled(&self) -> bool {
        nursery_age() >= 1
    }

    /// Is a major collection pending? This reads the generational module's
    /// `next_gc_full_heap` flag, whose name describes SCOPE (whole-heap),
    /// not pause shape: under Bactrian the request is normally executed as
    /// an InitialMark->FinalMark cycle, and only becomes `Pause::Full` when
    /// a forced-Full condition or a slicing feasibility gate applies (see
    /// decide_pause). Read-only: the request is consumed by
    /// [`Self::take_major_request`].
    fn major_request_pending(&self) -> bool {
        self.gen.next_gc_full_heap.load(Ordering::SeqCst)
    }

    /// Consume the pending major-collection request (see
    /// [`Self::major_request_pending`] for the naming note). Returns whether
    /// one was pending. Called exactly once per pause decision, in
    /// decide_pause, below the mid-cycle and sweep gates — so a request
    /// arriving while a cycle or sweep drain is in flight stays latched.
    fn take_major_request(&self) -> bool {
        self.gen.next_gc_full_heap.swap(false, Ordering::SeqCst)
    }

    /// Request a major collection (the same flag the binding's pacing sets
    /// through `force_full_heap_collection`).
    fn set_major_request(&self) {
        self.gen.next_gc_full_heap.store(true, Ordering::SeqCst);
    }

    /// Pop one parked sweep packet (incremental sweep).
    pub(super) fn pop_sweep_packet(&self) -> Option<Box<dyn GCWork<VM>>> {
        loop {
            match self.parked_sweep.steal() {
                crossbeam::deque::Steal::Success(w) => return Some(w),
                crossbeam::deque::Steal::Retry => continue,
                crossbeam::deque::Steal::Empty => return None,
            }
        }
    }

    /// Is the parked sweep queue empty? Injector::is_empty is momentary under
    /// concurrency; this is an EXACT test only under the drain invariants
    /// documented at the budget-expiry check in `BactrianSweepQuantum`.
    pub(super) fn sweep_queue_is_empty(&self) -> bool {
        self.parked_sweep.is_empty()
    }

    /// Called by the sweep quantum when it drains the queue empty. Guarded:
    /// only the true->false TRANSITION performs completion actions, so a
    /// quantum that raced ahead of the parking (empty pop, pending still
    /// false) cannot prematurely disarm anything.
    pub(super) fn sweep_queue_emptied(&self) {
        if !self.sweep_pending.swap(false, Ordering::SeqCst) {
            return;
        }
        // Deferred half of FinalMark's end_of_gc: the sweep is complete, so
        // new allocations no longer need eager line marks (see the FinalMark
        // arm in end_of_gc).
        {
            use crate::plan::global::HasSpaces;
            self.for_each_space(&mut |space: &dyn Space<VM>| {
                space.set_allocate_as_live(false);
            });
        }
        // AUTO-COMPACTION (knob-gated, MMTK_COMPACT_UTIL_PCT; 0 = off):
        // Bactrian's cycles never defragment — defrag runs only at STW Fulls,
        // which the pacing never schedules — so a fragmented mature space
        // (kb: ~15 MiB of partially-occupied blocks for 2.5 MiB live) holds
        // its slack forever. Stock OCaml's analog is automatic compaction.
        // The post-sweep metric is the honest trigger point: if more than
        // the threshold fraction of live blocks are only partially occupied,
        // request a Full — which (given reusable blocks exist) is already a
        // defragmenting collection by mmtk's decide_whether_to_defrag law
        // (!exhausted_reusable_space). Self-limiting: the compacting Full
        // resets the fraction, so it cannot storm.
        let threshold = compact_util_threshold_pct();
        if threshold > 0 {
            let (partial, live) = self.immix_space.post_sweep_fragmentation();
            // Small-heap floor: with a handful of live blocks the fraction is
            // meaningless (spectralnorm: 2-3 blocks, all partial -> the
            // trigger stormed a compacting Full per cycle). Require at least
            // 64 live blocks (2 MiB) before the law can fire.
            if live >= 64 && partial * 100 >= live * threshold {
                if std::env::var_os("MMTK_PACE_DEBUG").is_some() {
                    eprintln!(
                        "[compact] trigger: {partial}/{live} live blocks partial (>= {threshold}%)"
                    );
                }
                self.set_major_request();
            }
        }
    }

    /// Pop one parked marking packet (sliced mode).
    pub(super) fn pop_marking_packet(&self) -> Option<Box<dyn GCWork<VM>>> {
        loop {
            match self.parked_marking.steal() {
                crossbeam::deque::Steal::Success(w) => return Some(w),
                crossbeam::deque::Steal::Retry => continue,
                crossbeam::deque::Steal::Empty => return None,
            }
        }
    }

    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &BACTRIAN_CONSTRAINTS,
            global_side_metadata_specs:
                crate::plan::generational::new_generational_global_metadata_specs::<VM>(),
        };

        let immix_space = ImmixSpace::new(
            plan_args.get_mature_space_args(
                "immix_mature",
                true,
                false,
                VMRequest::discontiguous(),
            ),
            ImmixSpaceArgs {
                // Young objects are never allocated in the ImmixSpace directly.
                mixed_age: false,
                never_move_objects: false,
            },
        );

        // These buckets are never used by this plan (no compaction/forwarding stages).
        let scheduler = &plan_args.global_args.scheduler;
        scheduler.work_buckets[WorkBucketStage::VMRefForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::CalculateForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::SecondRoots].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::RefForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::FinalizableForwarding].set_enabled(false);
        scheduler.work_buckets[WorkBucketStage::Compact].set_enabled(false);

        let aged0 = CopySpace::new(
            plan_args.get_nursery_space_args("aged0", true, false, VMRequest::discontiguous()),
            false,
        );
        let aged1 = CopySpace::new(
            plan_args.get_nursery_space_args("aged1", true, false, VMRequest::discontiguous()),
            true,
        );

        let bactrian = Bactrian {
            gen: CommonGenPlan::new(plan_args),
            immix_space,
            aged0,
            aged1,
            aged_hi: AtomicBool::new(false),
            aging_this_gc: AtomicBool::new(false),
            last_gc_was_defrag: AtomicBool::new(false),
            current_pause: Atomic::new(None),
            sliced_marking: std::env::var("MMTK_MARK_SLICED")
                .map(|v| v != "0")
                .unwrap_or(true),
            parked_marking: crossbeam::deque::Injector::new(),
            parked_sweep: crossbeam::deque::Injector::new(),
            sweep_pending: AtomicBool::new(false),
            progress_pause_requested: AtomicBool::new(false),
            mark_quantum_hint_nanos: AtomicU64::new(0),
            cycle_tick_origin: AtomicBool::new(false),
            compact_requested: AtomicBool::new(false),
            previous_pause: Atomic::new(None),
            concurrent_marking_active: AtomicBool::new(false),
        };

        bactrian.verify_side_metadata_sanity();

        bactrian
    }

    /// Decide what kind of pause this collection is. Called once per collection from
    /// `schedule_collection`, with mutators about to be (or being) stopped.
    fn decide_pause(&self) -> Pause {
        if self.concurrent_marking_in_progress() {
            // While a cycle is in flight the only legal pauses are Nursery and
            // FinalMark (upgrading to Full mid-cycle is unsafe w.r.t. defrag — same
            // restriction as ConcurrentImmix). Any pending full-heap request stays
            // set (next_gc_full_heap) and is honoured after the cycle completes.
            if self.marking_queue_drained() {
                Pause::FinalMark
            } else {
                Pause::Nursery
            }
        } else {
            // Consume the "next GC should be full heap" request. Its meaning depends
            // on who set it:
            //  - a USER request (Gc.major/full_major/compact routes through
            //    handle_user_collection_request(exhaustive=true), which sets both
            //    user_triggered_collection and next_gc_full_heap) is a full-heap
            //    collection by contract → STW Full;
            //  - a mature-pressure request (the binding's GH#5 pacing, or the
            //    available-pages heuristic at end_of_gc) starts a *concurrent* major
            //    cycle — that is what a major collection IS in this design, as in
            //    stock OCaml → InitialMark.
            // INCREMENTAL SWEEP GATE: while the previous cycle's deferred
            // sweep is undrained, its line_mark_state / defrag histograms /
            // chunk map are still being consumed — starting a new cycle or a
            // Full (both re-prepare that state) would corrupt it. Stay on
            // Nursery pauses; a pending cycle request stays latched
            // (next_gc_full_heap is consumed below this gate) and waits out
            // the BUDGETED drain — see schedule_collection: only a genuine
            // allocation emergency drains unbudgeted (eager drain-all
            // measured 8.2 -> 14.4ms minor max at bt@2M and was rejected).
            if self.sliced_marking && self.sweep_pending.load(Ordering::SeqCst) {
                return Pause::Nursery;
            }
            let cycle_requested = self.take_major_request();
            let user_triggered = self
                .gen
                .common
                .base
                .global_state
                .user_triggered_collection
                .load(Ordering::SeqCst);
            let user_full = user_triggered
                && (cycle_requested || *self.gen.common.base.options.full_heap_system_gc);
            let emergency = self.genuine_allocation_emergency();
            let vm_exhausted = ((self.get_collection_reserved_pages() as f64
                * VM::VMObjectModel::VM_WORST_CASE_COPY_EXPANSION)
                as usize)
                > self.get_mature_physical_pages_available();
            let full = crate::plan::generational::FULL_NURSERY_GC
                || user_full
                || emergency
                || vm_exhausted;
            let decision = if full {
                Pause::Full
            } else if cycle_requested {
                // Debug bisection knob: BACTRIAN_NO_CONCURRENT=1 degrades every
                // major-cycle request to a STW Full GC (GenImmix-equivalent
                // behaviour), isolating the concurrent machinery when debugging.
                if std::env::var_os("BACTRIAN_NO_CONCURRENT").is_some() {
                    Pause::Full
                } else if self.sliced_marking && {
                    // Slicing is only worth running when it can make pauses
                    // small. Otherwise a monolithic Full has the same
                    // worst-case pause and costs less total GC time
                    // (bt-def@192M: 2663ms as Fulls vs 3181ms sliced,
                    // ~103-138ms max pause either way). Two checks:
                    //  - nursery size (MMTK_SLICE_MAX_NURSERY_MB, default 4):
                    //    with a big nursery, the minor pause alone is already
                    //    tens of milliseconds, so slicing cannot make pauses
                    //    small. Tick-origin cycles are exempt below: their
                    //    pauses are near-empty minors, small at any nursery
                    //    size.
                    //  - the slice-sizing hint (MMTK_MAX_QUANTUM_MS, default
                    //    50): if each slice would need more than this much
                    //    marking time, the runway cannot be covered by small
                    //    pauses at all.
                    let nursery_big = self.gen.common.base.gc_trigger.get_max_nursery_pages()
                        > slice_max_nursery_pages();
                    // Tick-origin cycles (mature-direct pacing) progress in
                    // near-empty nursery pauses — small at any nursery cap —
                    // so the nursery gate does not apply to them.
                    let tick_origin = self.cycle_tick_origin.load(Ordering::Relaxed);
                    let hint_ms =
                        self.mark_quantum_hint_nanos.load(Ordering::Relaxed) as f64 / 1e6;
                    (nursery_big && !tick_origin) || hint_ms > max_quantum_ms()
                } {
                    Pause::Full
                } else if !self.sliced_marking
                    && self.immix_space.reserved_pages() < conc_mark_min_mature_pages()
                {
                    // Prefer a Full pause instead of concurrent GC worker +
                    // mutator working if the mature space is smaller than a
                    // given threshold. This avoids LLC contention between the
                    // GC worker and the mutator, and the mutator does not
                    // have to go through the write barrier on each write.
                    Pause::Full
                } else {
                    Pause::InitialMark
                }
            } else {
                Pause::Nursery
            };
            // COMPACT-ALL (round 30): a pending mature-compaction request
            // rides the next major. Cycles never defragment (SATB marking
            // cannot move mature objects under mutator-held references), so
            // the request upgrades a would-be InitialMark to a monolithic
            // Full and arms every-block defrag selection on the space.
            let decision = match decision {
                Pause::Full | Pause::InitialMark
                    if self.compact_requested.swap(false, Ordering::SeqCst) =>
                {
                    self.immix_space.request_compact_all();
                    Pause::Full
                }
                d => d,
            };
            if std::env::var_os("BACTRIAN_TRACE").is_some() {
                eprintln!(
                    "[bactrian] decide: cycle_req={} user={} emergency={} vm_exhausted={} -> {:?}",
                    cycle_requested, user_triggered, emergency, vm_exhausted, decision
                );
            }
            decision
        }
    }

    /// A GENUINE allocation-failure emergency (degrade to STW Full /
    /// unbudgeted quanta), as opposed to mmtk-core's raw
    /// `cur_collection_attempts > 1`. The raw counter is spuriously 2 for
    /// every binding-forced pacing trigger honored at the first poll after
    /// a minor: the allocation that triggered the minor retries, its TLAB
    /// refill polls, the pending `next_gc_full_heap` blocks it again — two
    /// GCs with no successful allocation between reads as a failed-alloc
    /// retry loop. That hijack degraded every post-minor pressure cycle to
    /// an emergency monolithic Full (bt@192M: all majors, 80-140ms pauses;
    /// round 30). A real OOM loop reaches 3 on its second failed retry, one
    /// bounded nursery-class pause later — and the mid-cycle emergency
    /// unbudgeted mark quantum completes marking so the Full/sweep can free
    /// the backlog. mmtk-core's own HeapOutOfMemory protocol reads its own
    /// flag and is unaffected.
    fn genuine_allocation_emergency(&self) -> bool {
        self.gen
            .common
            .base
            .global_state
            .cur_collection_attempts
            .load(Ordering::SeqCst)
            > 2
    }

    /// Temporary bring-up tracing (release builds strip `log`); gated on
    /// BACTRIAN_TRACE=1.
    fn trace_pause(&self, what: &str, pause: Pause) {
        if std::env::var_os("BACTRIAN_TRACE").is_some() {
            use crate::plan::concurrent::diag;
            use std::sync::atomic::Ordering as O;
            eprintln!(
                "[bactrian] {} {:?} (marking={}, mature_pages={}, seeded={}, enq={}, traced={}, skipped_young={}, satb={}, satb_young_drop={})",
                what,
                pause,
                self.concurrent_marking_in_progress(),
                self.immix_space.reserved_pages(),
                diag::SEEDED.load(O::Relaxed),
                diag::ENQUEUED.load(O::Relaxed),
                diag::TRACED.load(O::Relaxed),
                diag::SKIPPED_YOUNG.load(O::Relaxed),
                diag::SATB_ENQ.load(O::Relaxed),
                diag::SATB_YOUNG_DROP.load(O::Relaxed),
            );
        }
    }

    pub fn concurrent_marking_in_progress(&self) -> bool {
        self.concurrent_marking_active.load(Ordering::Acquire)
    }

    fn set_concurrent_marking_state(&self, active: bool) {
        use crate::plan::global::HasSpaces;

        // Tell the spaces to allocate new objects as live (eager line marks in the
        // Immix copy allocator; mature+marked LOS allocations).
        self.for_each_space(&mut |space: &dyn Space<VM>| {
            space.set_allocate_as_live(active);
        });

        self.concurrent_marking_active
            .store(active, Ordering::SeqCst);
    }

    pub(crate) fn is_concurrent_marking_active(&self) -> bool {
        self.concurrent_marking_active.load(Ordering::SeqCst)
    }

    fn previous_pause(&self) -> Option<Pause> {
        self.previous_pause.load(Ordering::SeqCst)
    }
}


impl<VM: VMBinding> Bactrian<VM> {
    fn aged_to(&self) -> &CopySpace<VM> {
        if self.aged_hi.load(Ordering::SeqCst) {
            &self.aged1
        } else {
            &self.aged0
        }
    }

    fn aged_from_mut(&mut self) -> &mut CopySpace<VM> {
        if self.aged_hi.load(Ordering::SeqCst) {
            &mut self.aged0
        } else {
            &mut self.aged1
        }
    }

    fn aged_to_mut(&mut self) -> &mut CopySpace<VM> {
        if self.aged_hi.load(Ordering::SeqCst) {
            &mut self.aged1
        } else {
            &mut self.aged0
        }
    }

    /// Both aged spaces become from-spaces (their residents evacuate to mature
    /// through the nursery/full trace). Used by every non-aging pause.
    fn prepare_aged_all_from(&mut self) {
        self.aged0.prepare(true);
        self.aged1.prepare(true);
        self.aged0
            .set_copy_for_sft_trace(Some(CopySemantics::PromoteToMature));
        self.aged1
            .set_copy_for_sft_trace(Some(CopySemantics::PromoteToMature));
    }
}

/// MMTK_NURSERY_AGE: >= 1 enables survivor aging (one extra minor to die before
/// promotion — the semispace pair gives exactly one age step today; values > 1
/// are accepted but behave as 1 until per-object age bits exist). Default 0
/// (off): behaviour is identical to the pre-aging plan.
fn nursery_age() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MMTK_NURSERY_AGE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Largest nursery for which sliced major cycles pay (pages;
/// MMTK_SLICE_MAX_NURSERY_MB overrides, default 4MB — see the feasibility
/// escape in decide_pause). Above it, minor pauses are promotion-bound and
/// already dwarf any quantum, so majors run monolithic.
fn slice_max_nursery_pages() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let mb = std::env::var("MMTK_SLICE_MAX_NURSERY_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|m| *m > 0)
            .unwrap_or(4);
        mb * 1024 * 1024 / crate::util::constants::BYTES_IN_PAGE
    })
}

/// Largest per-pause mark quantum worth slicing for, ms
/// (MMTK_MAX_QUANTUM_MS overrides; see the feasibility escape in
/// decide_pause). Above this the monolithic Full wins on both axes.
fn max_quantum_ms() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MMTK_MAX_QUANTUM_MS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|ms| *ms > 0.0)
            .unwrap_or(50.0)
    })
}

/// Mature-size floor (in pages) below which a requested major cycle runs as a
/// STW Full GC instead of concurrent marking. MMTK_CONC_MARK_MIN_MATURE_MB
/// overrides; 256 MB is the default (0 = always concurrent).
fn conc_mark_min_mature_pages() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let mb = std::env::var("MMTK_CONC_MARK_MIN_MATURE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256);
        mb * 1024 * 1024 / crate::util::constants::BYTES_IN_PAGE
    })
}
