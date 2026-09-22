use super::global::Bactrian;
use crate::plan::concurrent::concurrent_marking_work::ConcurrentTraceObjects;
use crate::plan::concurrent::global::ConcurrentPlan;
use crate::plan::concurrent::Pause;
use crate::plan::generational::global::GenerationalPlan;
use crate::plan::generational::global::GenerationalPlanExt;
use crate::plan::global::PlanTraceObject;
use crate::plan::VectorObjectQueue;
use crate::policy::gc_work::TraceKind;
use crate::policy::gc_work::DEFAULT_TRACE;
use crate::policy::immix::TRACE_KIND_FAST;
use crate::policy::space::Space;
use crate::scheduler::gc_work::PlanProcessEdges;
use crate::scheduler::gc_work::PlanScanObjects;
use crate::scheduler::gc_work::ProcessEdgesBase;
use crate::scheduler::gc_work::SlotOf;
use crate::scheduler::gc_work::UnsupportedProcessEdges;
use crate::scheduler::ProcessEdgesWork;
use crate::scheduler::WorkBucketStage;
use crate::util::ObjectReference;
use crate::vm::slot::Slot;
use crate::vm::Scanning;
use crate::vm::VMBinding;

/// MMTK_UP_OLDIFY=1: opt-in stock-oldify fast path for plain nursery pauses
/// under UP (see Scanning::up_oldify_packet). Default OFF.
fn up_oldify_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MMTK_UP_OLDIFY")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}
use crate::MMTK;
use std::ops::{Deref, DerefMut};

/// Work context for every nursery-anchored pause (`Nursery`, `InitialMark`,
/// `FinalMark`); the trace type is pause-aware.
pub(in crate::plan) struct BactrianNurseryGCWorkContext<VM: VMBinding>(
    std::marker::PhantomData<VM>,
);
impl<VM: VMBinding> crate::scheduler::GCWorkContext for BactrianNurseryGCWorkContext<VM> {
    type VM = VM;
    type PlanType = Bactrian<VM>;
    type DefaultProcessEdges = BactrianNurseryProcessEdges<VM>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// Work context for `Pause::Full`: GenImmix's STW full-heap collection.
pub(in crate::plan) struct BactrianSTWGCWorkContext<VM: VMBinding, const KIND: TraceKind>(
    std::marker::PhantomData<VM>,
);
impl<VM: VMBinding, const KIND: TraceKind> crate::scheduler::GCWorkContext
    for BactrianSTWGCWorkContext<VM, KIND>
{
    type VM = VM;
    type PlanType = Bactrian<VM>;
    type DefaultProcessEdges = PlanProcessEdges<VM, Bactrian<VM>, KIND>;
    type PinningProcessEdges = UnsupportedProcessEdges<VM>;
}

/// The trace for all nursery-anchored Bactrian pauses. It is the generational
/// nursery trace (young objects are promoted, transitively), extended per pause:
///
/// - `Pause::Nursery`: exactly the generational nursery trace; mature objects are
///   left to the concurrent marker (mid-cycle) or the next cycle.
///
/// - `Pause::InitialMark`: additionally *seeds* the concurrent marking queue with
///   every mature object the trace encounters (root targets, remembered-set
///   targets, and the mature children of promoted objects). Together with
///   emptying the nursery this establishes the SATB snapshot; the mature-to-
///   mature closure then runs concurrently.
///
/// - `Pause::FinalMark`: additionally *marks* (non-moving, with transitive
///   scanning via this same trace) any mature object it reaches that concurrent
///   marking missed. This makes FinalMark a true remark pause: marking is
///   complete at its end regardless of concurrent coverage — the same safety
///   structure as other SATB collectors' final remark. In the common case
///   everything reachable is already marked and this degenerates to mark-bit
///   checks. It is also what lets weak-reference processing at FinalMark
///   resurrect ("retain") mature objects correctly.
pub(in crate::plan) struct BactrianNurseryProcessEdges<VM: VMBinding> {
    plan: &'static Bactrian<VM>,
    base: ProcessEdgesBase<VM>,
    pause: Pause,
    /// InitialMark only: mature objects encountered by this trace, handed to the
    /// concurrent marker as marking roots.
    mark_seed: Vec<ObjectReference>,
}

/// Slot visitor that just collects slots into a scratch buffer, for the UP
/// direct-trace drain below.
struct SlotCollector<'a, S: Slot>(&'a mut Vec<S>);
impl<S: Slot> crate::vm::SlotVisitor<S> for SlotCollector<'_, S> {
    fn visit_slot(&mut self, slot: S) {
        self.0.push(slot);
    }
}

/// Core services for the binding's oldify loop (MMTK_UP_OLDIFY): young test
/// on the copying nursery + aged pair, mature bump-alloc through the
/// worker's copy context, and the full promotion post-copy protocol.
struct BactrianOldifyOps<'w, VM: VMBinding> {
    plan: &'static Bactrian<VM>,
    worker: &'w mut crate::scheduler::GCWorker<VM>,
}

impl<VM: VMBinding> crate::vm::UpOldifyOps<VM> for BactrianOldifyOps<'_, VM> {
    #[inline(always)]
    fn young_range(&self) -> (crate::util::Address, crate::util::Address) {
        use crate::policy::space::Space;
        let c = self.plan.gen.nursery.common();
        (c.start, c.start + c.extent)
    }

    #[inline(always)]
    fn in_young(&self, addr: crate::util::Address) -> bool {
        self.plan.is_address_in_nursery(addr)
    }

    fn alloc_mature(&mut self, bytes: usize) -> crate::util::Address {
        self.worker.get_copy_context_mut().alloc_copy(
            unsafe {
                crate::util::ObjectReference::from_raw_address_unchecked(
                    crate::util::Address::from_usize(8),
                )
            },
            bytes,
            crate::util::constants::BYTES_IN_WORD,
            0,
            crate::util::copy::CopySemantics::PromoteToMature,
        )
    }

    fn post_copy(&mut self, object: crate::util::ObjectReference, bytes: usize) {
        self.worker.get_copy_context_mut().post_copy(
            object,
            bytes,
            crate::util::copy::CopySemantics::PromoteToMature,
        );
        self.plan.post_scan_object(object);
    }

    fn is_young_los(&self, object: crate::util::ObjectReference) -> bool {
        use crate::policy::space::Space;
        self.plan.gen.common.los.in_space(object) && self.plan.gen.common.los.is_in_nursery(object)
    }

    fn promote_young_los(&mut self, object: crate::util::ObjectReference) -> bool {
        // LOS in-place promotion: test_and_mark clears the nursery bit and
        // moves the treadmill entry; a newly-promoted object is enqueued for
        // scanning — intercepted with a local queue so the caller scans it
        // via the oldify walk instead.
        let mut q = crate::plan::VectorObjectQueue::default();
        self.plan.gen.common.los.trace_object(&mut q, object);
        !q.is_empty()
    }
}

/// Sliced-STW mark quantum: pops parked marking packets (see
/// `Bactrian::parked_marking`) and executes them on this worker, world
/// stopped, until the queue empties or the budget expires. Scheduled in the
/// Release stage of mid-cycle Nursery pauses (budgeted — stock OCaml's
/// allocation-paced mark slice, one per minor) and in the Closure stage of
/// FinalMark (unbudgeted — drain everything, including SATB flushes parked
/// during StopMutators).
pub(in crate::plan) struct BactrianMarkQuantum<VM: VMBinding> {
    plan: &'static Bactrian<VM>,
    budget: Option<std::time::Duration>,
}

/// Per-quantum budget. MMTK_MARK_SLICE_MS overrides (fractional ok);
/// default 2ms — comparable to a nursery pause at the stock-parity 2 MiB
/// nursery, so mid-cycle pauses stay in vanilla's slice-pause class.
fn mark_slice_budget() -> std::time::Duration {
    static V: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let ms = std::env::var("MMTK_MARK_SLICE_MS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|ms| *ms > 0.0 && *ms <= 1000.0)
            .unwrap_or(2.0);
        std::time::Duration::from_secs_f64(ms / 1e3)
    })
}

impl<VM: VMBinding> BactrianMarkQuantum<VM> {
    pub(in crate::plan) fn budgeted(plan: &'static Bactrian<VM>) -> Self {
        // The time budget only tops up the work floors (inflow, then the
        // runway share); MMTK_MARK_SLICE_MS, default 2 ms.
        Self {
            plan,
            budget: Some(mark_slice_budget()),
        }
    }
    pub(in crate::plan) fn unbudgeted(plan: &'static Bactrian<VM>) -> Self {
        Self { plan, budget: None }
    }
}

impl<VM: VMBinding> crate::scheduler::GCWork<VM> for BactrianMarkQuantum<VM> {
    fn do_work(&mut self, worker: &mut crate::scheduler::GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        // Work floor (budgeted quanta only): first drain at least as many
        // objects as were handed to marking packets since the previous quantum
        // finished — SATB old values the barrier parked between pauses and at
        // mutator flush, plus the nursery closure's seeds — and only THEN run
        // the time budget. The time budget alone is a rate guess (debt over
        // runway pauses at an assumed mark rate, re-derived every minor against
        // a heap that grows while the cycle is open); if the inflow outruns it
        // the parked queue only grows and the cycle never reaches FinalMark
        // (eio_conc: 60-125k SATB entries per minor vs ~50k traced in a 2-5 ms
        // quantum; the queue hovered at ~1M objects for 1600 minors, RSS 3 ->
        // 24 GB). Inflow and budget ADD: a max() of the two only holds the
        // queue steady and never drains it. The quantum's own child packets are
        // not inflow: the snapshot is taken after the drain.
        use crate::plan::concurrent::diag::{ENQUEUED, SATB_ENQ, TRACED};
        use std::sync::atomic::Ordering::Relaxed;
        // SATB old values are counted at enqueue time (SATB_ENQ, mutator side):
        // their ProcessModBufSATB packets only feed ENQUEUED when they execute
        // inside a quantum, which is after this quota is read.
        let inflow_mark = || ENQUEUED.load(Relaxed) + SATB_ENQ.load(Relaxed);
        let quota = self.budget.map(|_| {
            let inflow = inflow_mark()
                .saturating_sub(self.plan.enqueued_at_last_quantum.load(Relaxed) as usize);
            // Runway floor: besides its own inflow, each slice retires enough
            // of the backlog for the cycle to finish before promotion consumes
            // the runway latched at InitialMark — stock's "slice work
            // proportional to what must be done before the heap is full".
            // With no runway left, drain everything now.
            let backlog = self.plan.mark_backlog_objects();
            let (promo, runway) = self.plan.promotion_and_runway();
            // Minors the runway allows; never fewer than the slices needed to
            // keep each slice within the pause target at the measured mark
            // rate. Past the runway the heap overshoots the frozen limit by
            // promotion x the slices left — bounded — instead of one slice
            // draining everything (sedlex: 16-38 s).
            let avail_runway = if promo == 0 {
                u64::MAX
            } else {
                (runway / promo).max(1)
            };
            let rate = self.plan.mark_rate_x256();
            let min_slices = if rate == 0 {
                1
            } else {
                let backlog_ms = backlog as f64 * 256.0 / rate as f64;
                (backlog_ms / self.plan.slice_target_ms()).ceil().max(1.0) as u64
            };
            let avail =
                std::cmp::max(std::cmp::min(avail_runway, u64::MAX / 2), min_slices) as usize;
            let share = if promo == 0 && rate == 0 {
                0
            } else {
                backlog.div_ceil(avail)
            };
            inflow + share
        });
        let traced_at_start = TRACED.load(Relaxed);
        let started = std::time::Instant::now();
        // Armed once the quota is met; None while the floor is being worked.
        let mut deadline: Option<std::time::Instant> = None;
        let mut packets = 0usize;
        // Run the drain under full-heap LOS semantics (mid-cycle marking must
        // mark mature LOS objects even when the enclosing pause latched
        // nursery semantics); restore the previous mode after.
        let was_full = mmtk
            .get_plan()
            .common()
            .los
            .set_marking_full_semantics(true);
        while let Some(mut w) = self.plan.pop_marking_packet() {
            w.do_work(worker, mmtk);
            packets += 1;
            if let Some(b) = self.budget {
                match deadline {
                    None => {
                        if TRACED.load(Relaxed) - traced_at_start >= quota.unwrap_or(0) {
                            deadline = Some(std::time::Instant::now() + b);
                        }
                    }
                    Some(d) => {
                        if std::time::Instant::now() >= d {
                            break;
                        }
                    }
                }
            }
        }
        mmtk.get_plan()
            .common()
            .los
            .set_marking_full_semantics(was_full);
        self.plan
            .enqueued_at_last_quantum
            .store(inflow_mark() as u64, Relaxed);
        self.plan.note_mark_rate(
            (TRACED.load(Relaxed) - traced_at_start) as u64,
            started.elapsed().as_nanos() as u64,
        );
        self.plan
            .note_quantum_nanos(started.elapsed().as_nanos() as u64);
        if let Some(q) = quota {
            // Projection guard: at the measured net rate (objects of backlog
            // retired per slice, EWMA), does the backlog finish before
            // promotion (pages per minor, EWMA) consumes the runway? If not,
            // the next quantum runs unbudgeted — drain in one pause, FinalMark
            // next — the legal mid-cycle equivalent of giving up on slicing (a
            // Full cannot start with a cycle in flight). A short warm-up keeps
            // a bursty first few slices from firing it. A queue that is not
            // shrinking at all is the infinite case of the same test.
            let traced = TRACED.load(Relaxed) - traced_at_start;
            let net = traced.saturating_sub(q) as u64;
            let (net_ewma, slices) = self.plan.note_mark_slice(net);
            let (promo, runway) = self.plan.sample_promotion();
            let backlog = self.plan.mark_backlog_objects();
            let drained = self.plan.marking_queue_drained();
            let need = if net_ewma == 0 {
                f64::INFINITY
            } else {
                backlog as f64 / net_ewma as f64
            };
            let avail = if promo == 0 {
                f64::INFINITY
            } else {
                runway as f64 / promo as f64
            };
            // With the runway-paced, target-capped share above, the slices
            // already finish the cycle inside the runway or overshoot it by a
            // bounded amount; an unbudgeted drain here would just be the
            // multi-second pause the cap exists to avoid (v6e eio: 14
            // escalations = 11 pauses > 500 ms). Report, don't escalate.
            let escalate = !drained && slices >= 4 && need > avail;
            if std::env::var_os("MMTK_PACE_DEBUG").is_some() {
                eprintln!(
                    "[pace] quantum: budget={:.1}ms quota={} traced={} packets={} took={:.1}ms drained={} | guard: backlog={} net={}/slice promo={}p runway={}MB need={:.0} avail={:.0} slices={}{}",
                    self.budget.map(|b| b.as_secs_f64() * 1e3).unwrap_or(0.0),
                    q,
                    traced,
                    packets,
                    started.elapsed().as_secs_f64() * 1e3,
                    drained,
                    backlog,
                    net_ewma,
                    promo,
                    runway * 4096 / (1 << 20),
                    need,
                    avail,
                    slices,
                    if escalate { " -> ESCALATE" } else { "" }
                );
            }
        }
        probe!(mmtk, bactrian_mark_quantum, packets);
    }
}

/// Incremental-sweep quantum: pops deferred chunk-sweep packets (see
/// `Bactrian::parked_sweep`) and executes them, world stopped, until the
/// queue empties or the budget expires — stock OCaml's sweep slices,
/// scheduled in the Release stage of nursery pauses after FinalMark.
/// Unbudgeted when the pacing wants the next cycle (drain-to-completion).
/// The freed blocks flow to the page resource per packet, so RSS falls
/// incrementally across the minors instead of at one FinalMark cliff.
pub(in crate::plan) struct BactrianSweepQuantum<VM: VMBinding> {
    plan: &'static Bactrian<VM>,
    budget: Option<std::time::Duration>,
}

/// Per-quantum sweep budget. MMTK_SWEEP_SLICE_MS overrides; default 2ms
/// (a chunk-sweep packet is ~fast: line-mark scans over 4MB of blocks).
fn sweep_slice_budget() -> std::time::Duration {
    static V: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let ms = std::env::var("MMTK_SWEEP_SLICE_MS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|ms| *ms > 0.0 && *ms <= 1000.0)
            .unwrap_or(2.0);
        std::time::Duration::from_secs_f64(ms / 1e3)
    })
}

impl<VM: VMBinding> BactrianSweepQuantum<VM> {
    pub(in crate::plan) fn budgeted(plan: &'static Bactrian<VM>) -> Self {
        Self {
            plan,
            budget: Some(sweep_slice_budget()),
        }
    }
    pub(in crate::plan) fn unbudgeted(plan: &'static Bactrian<VM>) -> Self {
        Self { plan, budget: None }
    }
}

impl<VM: VMBinding> crate::scheduler::GCWork<VM> for BactrianSweepQuantum<VM> {
    fn do_work(&mut self, worker: &mut crate::scheduler::GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        let deadline = self.budget.map(|b| std::time::Instant::now() + b);
        // Sample this pause's promotion BEFORE sweeping (the sweep frees
        // mature pages, which would hide it) — see sample_promotion.
        let promo_runway = self.budget.map(|_| self.plan.sample_promotion());
        // Runway floor for the sweep: a pending cycle request waits on this
        // drain, so each slice sweeps at least its share of the remaining
        // chunks for the drain to finish before promotion consumes the frozen
        // runway; with no runway left, drain everything now. (A 2 ms slice
        // was tuned at 192 MB; at 8-15 GB it took 146-432 minors.)
        let sweep_quota = promo_runway.map(|(promo, runway)| {
            let remaining = self.plan.sweep_packets_remaining();
            if promo == 0 {
                0
            } else {
                remaining.div_ceil((runway / promo).max(1) as usize)
            }
        });
        // Soft time cap on the share (MMTK_SWEEP_SLICE_CAP_MS, default 20):
        // past the runway the next cycle waits a little longer rather than
        // one slice sweeping gigabytes.
        let cap = std::time::Duration::from_secs_f64(
            std::env::var("MMTK_SWEEP_SLICE_CAP_MS")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(20.0)
                / 1e3,
        );
        let started = std::time::Instant::now();
        let mut packets = 0usize;
        loop {
            let Some(mut w) = self.plan.pop_sweep_packet() else {
                // Queue empty: the cycle's sweep is COMPLETE. (Single quantum
                // per pause and quanta only run world-stopped, so this edge
                // cannot race a concurrent producer — packets are only parked
                // by FinalMark, which is gated on the previous drain.)
                self.plan.sweep_queue_emptied();
                break;
            };
            w.do_work(worker, mmtk);
            packets += 1;
            if let Some(d) = deadline {
                let now = std::time::Instant::now();
                if now >= d && (packets >= sweep_quota.unwrap_or(0) || now >= started + cap) {
                    // Budget expired. If the queue emptied on this very
                    // packet, still flip the flag in THIS pause — a one-pause
                    // delay would hold the sweep gate (no new cycle or Full)
                    // and the post-sweep baseline latch for one extra minor.
                    //
                    // is_empty() is an exact test here, but ONLY under two
                    // invariants of the drain design: (a) no producers — all
                    // packets are parked by FinalMark before the first
                    // quantum is scheduled (see the schedule_collection
                    // ordering note); (b) no packet in flight — one quantum
                    // per pause, and every popped packet was executed to
                    // completion above. If either breaks, an empty-looking
                    // queue can coexist with an unswept packet and this
                    // reverts to the false-sweep-complete class (fragmed T4,
                    // NOTES 2026-08-12) — switch back to a steal-based
                    // emptiness test in that world.
                    if self.plan.sweep_queue_is_empty() {
                        self.plan.sweep_queue_emptied();
                    }
                    break;
                }
            }
        }
        self.plan
            .note_quantum_nanos(started.elapsed().as_nanos() as u64);
        if let Some((promo, runway)) = promo_runway {
            // Projection guard for the sweep: a pending cycle request waits on
            // this drain, so at the measured packets-per-slice rate the
            // remaining chunks must be swept before promotion consumes the
            // runway; otherwise the next sweep quantum runs unbudgeted.
            let (ewma_x256, slices) = self.plan.note_sweep_slice(packets as u64);
            let remaining = self.plan.sweep_packets_remaining();
            let need = if ewma_x256 == 0 {
                f64::INFINITY
            } else {
                remaining as f64 * 256.0 / ewma_x256 as f64
            };
            let avail = if promo == 0 {
                f64::INFINITY
            } else {
                runway as f64 / promo as f64
            };
            let escalate = remaining > 0 && slices >= 2 && need > avail;
            if escalate {
                self.plan.request_escalate_sweep();
            }
            if std::env::var_os("MMTK_PACE_DEBUG").is_some() {
                eprintln!(
                    "[pace] sweep quantum: packets={} took={:.1}ms remaining={} rate={:.1}/slice promo={}p runway={}MB need={:.0} avail={:.0} slices={}{}",
                    packets,
                    started.elapsed().as_secs_f64() * 1e3,
                    remaining,
                    ewma_x256 as f64 / 256.0,
                    promo,
                    runway * 4096 / (1 << 20),
                    need,
                    avail,
                    slices,
                    if escalate { " -> ESCALATE" } else { "" }
                );
            }
        }
        probe!(mmtk, bactrian_sweep_quantum, packets);
    }
}

impl<VM: VMBinding> BactrianNurseryProcessEdges<VM> {
    /// Match ProcessEdgesWork's own buffer sizing for the seed packets.
    const SEED_CAPACITY: usize = 4096;

    /// UP direct-trace closure: with a single tracer inside a stopped-world
    /// pause, consume the whole transitive closure inside THIS packet with an
    /// explicit work list — stock oldify's todo-list discipline — instead of
    /// bouncing every generation of the BFS through packet creation, bucket
    /// scheduling and a fresh ProcessEdges instance. Per object this performs
    /// exactly the packet path's protocol (support_slot_enqueuing → scan_object
    /// → post_scan_object → process each slot), so trace semantics, line
    /// marking at scan time, InitialMark seed collection and the FinalMark
    /// remark all behave identically; only the scheduling round-trips go away.
    fn drain_closure_locally(&mut self) {
        use crate::vm::Scanning;
        let tls = self.worker().tls;
        let mut scratch: Vec<SlotOf<Self>> = Vec::new();
        // Objects the VM cannot expose as slots: Scanning::support_slot_enqueuing
        // may answer false per object, and the packet path then scans them with
        // scan_object_and_trace_edges. This drain cannot do that inline (the
        // tracer context needs the worker while we hold self), so such objects
        // are diverted to the normal packet path below instead of being
        // asserted away. The OCaml binding answers true for every object (trait
        // default), so for it this vector stays empty.
        let mut fallback: Vec<ObjectReference> = Vec::new();
        loop {
            let nodes = self.pop_nodes();
            if nodes.is_empty() {
                break;
            }
            for object in nodes {
                if !<VM as VMBinding>::VMScanning::support_slot_enqueuing(tls, object) {
                    fallback.push(object);
                    continue;
                }
                {
                    let mut collector = SlotCollector(&mut scratch);
                    <VM as VMBinding>::VMScanning::scan_object(tls, object, &mut collector);
                }
                self.plan.post_scan_object(object);
                for slot in scratch.iter().copied() {
                    self.process_slot(slot);
                }
                scratch.clear();
            }
        }
        if !fallback.is_empty() {
            // Same packet flush() uses for un-drained nodes: PlanScanObjects with
            // the plan's post_scan hook. Its scan_object_and_trace_edges path
            // traces these objects' fields through a fresh ProcessEdges instance,
            // whose own flush drains locally again, so closure completeness is
            // unchanged; only these objects skip the local work list.
            self.start_or_dispatch_scan_work(self.create_scan_work(fallback));
        }
    }

    fn flush_mark_seed(&mut self) {
        if !self.mark_seed.is_empty() {
            let objects = std::mem::take(&mut self.mark_seed);
            let w = ConcurrentTraceObjects::<VM, Bactrian<VM>, TRACE_KIND_FAST>::new(
                objects,
                self.base.mmtk(),
            );
            // Route via the plan: worker-concurrent mode parks in the Concurrent
            // bucket without notifying (the scheduler opens it when the pause
            // ends); sliced mode parks in the plan queue for in-pause quanta.
            self.plan.schedule_marking_packet(Box::new(w));
        }
    }
}

impl<VM: VMBinding> ProcessEdgesWork for BactrianNurseryProcessEdges<VM> {
    type VM = VM;
    type ScanObjectsWorkType = PlanScanObjects<Self, Bactrian<VM>>;

    fn new(
        slots: Vec<SlotOf<Self>>,
        roots: bool,
        mmtk: &'static MMTK<VM>,
        bucket: WorkBucketStage,
    ) -> Self {
        let base = ProcessEdgesBase::new(slots, roots, mmtk, bucket);
        let plan = base.plan().downcast_ref::<Bactrian<VM>>().unwrap();
        // The pause kind is fixed for the duration of a collection; latch it here.
        // (Packets of this type only ever run inside a pause.)
        let pause = plan.current_pause().unwrap_or(Pause::Nursery);
        Self {
            plan,
            base,
            pause,
            mark_seed: Vec::new(),
        }
    }

    fn trace_object(&mut self, object: ObjectReference) -> ObjectReference {
        // We cannot borrow `self` twice in a call, so extract `worker` first.
        let worker = self.worker();
        let new_object = self
            .plan
            .trace_object_nursery::<VectorObjectQueue, DEFAULT_TRACE>(
                &mut self.base.nodes,
                object,
                worker,
            );
        // `new_object == object` means the object was already mature (promotion
        // returns the new copy, which post_copy born-black-marked and which this
        // trace scans transitively; an LOS "promotion" is in place and is idempotent
        // under both treatments below).
        //
        // Pause gate FIRST: the seed/remark treatments below only exist for the
        // two marking-fused pauses. A plain mid-cycle Nursery pause ran the
        // young-check chain (4 space lookups per traced object since the
        // young-LOS fix) for a match arm that does nothing — measured ~2-4%
        // of bt's whole-process cycles. The chain is also SOUND to skip for
        // the LOS side here: any LOS object this trace reaches was in-place
        // promoted (nursery bit cleared) before this check runs.
        if matches!(self.pause, Pause::InitialMark | Pause::FinalMark)
            && new_object == object
            && !self.plan.is_object_in_nursery(object)
        {
            match self.pause {
                Pause::InitialMark => {
                    crate::plan::concurrent::diag::SEEDED
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.mark_seed.push(object);
                    if self.mark_seed.len() >= Self::SEED_CAPACITY {
                        self.flush_mark_seed();
                    }
                }
                Pause::FinalMark => {
                    // Remark: mark (non-moving) anything concurrent marking missed;
                    // newly-marked objects are enqueued to base.nodes and scanned
                    // with this same trace. Already-marked objects are a mark-bit
                    // check only.
                    let marked = self
                        .plan
                        .trace_object::<VectorObjectQueue, TRACE_KIND_FAST>(
                            &mut self.base.nodes,
                            object,
                            worker,
                        );
                    debug_assert_eq!(marked, object, "FinalMark remark must not move");
                }
                _ => (),
            }
        }
        new_object
    }

    fn process_slot(&mut self, slot: SlotOf<Self>) {
        let Some(object) = slot.load() else {
            return;
        };
        let new_object = self.trace_object(object);
        // With survivor aging, a trace result may legitimately be YOUNG (in the
        // aged to-space); it must only never remain in the nursery proper.
        debug_assert!(!self.plan.gen.nursery.in_space(new_object));
        if new_object != object {
            slot.store(new_object);
        }
    }

    fn process_slots(&mut self) {
        // OPT-IN oldify fast path (MMTK_UP_OLDIFY=1): plain nursery pauses
        // only (no seed/remark logic), single tracer, world stopped. The
        // binding walks this packet's slots and their transitive closure
        // natively — stock minor_gc.c's structure — leaving nothing to trace
        // or schedule for this packet. Aging must be off (young survivors
        // would need the aged copy path, which the oldify loop doesn't know).
        if self.pause == Pause::Nursery
            && up_oldify_enabled()
            && crate::util::up_trace::up()
            && !self.plan.aging_enabled()
            && !*self.base.mmtk().get_options().count_live_bytes_in_gc
            && !self.base.slots.is_empty()
        {
            let slots = std::mem::take(&mut self.base.slots);
            let plan = self.plan;
            let tls = self.worker().tls;
            let consumed = {
                let worker = self.worker();
                let mut ops = BactrianOldifyOps { plan, worker };
                <VM as VMBinding>::VMScanning::up_oldify_packet::<BactrianOldifyOps<VM>>(
                    tls, &slots, &mut ops,
                )
            };
            if consumed {
                return;
            }
            // Binding declined: restore and take the generic path.
            self.base.slots = slots;
        }
        for i in 0..self.base.slots.len() {
            self.process_slot(self.base.slots[i])
        }
    }

    fn flush(&mut self) {
        // Single tracer: finish the whole closure here (see drain_closure_locally).
        // Gated off when live-bytes stats are requested — the packet path is the
        // one that accounts them.
        if crate::util::up_trace::up() && !*self.base.mmtk().get_options().count_live_bytes_in_gc {
            self.drain_closure_locally();
        }
        self.flush_mark_seed();
        // Default flush behaviour: hand accumulated nodes to a scan-objects packet.
        let nodes = self.pop_nodes();
        if !nodes.is_empty() {
            self.start_or_dispatch_scan_work(self.create_scan_work(nodes));
        }
    }

    fn create_scan_work(&self, nodes: Vec<ObjectReference>) -> Self::ScanObjectsWorkType {
        PlanScanObjects::new(self.plan, nodes, false, self.bucket)
    }
}

impl<VM: VMBinding> Drop for BactrianNurseryProcessEdges<VM> {
    fn drop(&mut self) {
        // Safety net: flush any buffered marking seeds when the instance is dropped.
        // Most callers go through the blanket `GCWork for E` do_work (which flushes),
        // but a few construct a ProcessEdgesWork and drop it without calling flush()
        // — e.g. `ProcessEdgesWorkTracerContext::with_tracer` (weak-ref "retain"
        // processing). Losing seeds means live mature objects escape the snapshot and
        // get swept at FinalMark.
        self.flush_mark_seed();
    }
}

impl<VM: VMBinding> Deref for BactrianNurseryProcessEdges<VM> {
    type Target = ProcessEdgesBase<VM>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<VM: VMBinding> DerefMut for BactrianNurseryProcessEdges<VM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
