use super::gc_work::FastRCPrepare;
use super::gc_work::LXRRCWorkContext;
use super::mutator::ALLOCATOR_MAPPING;
use super::rc::ProcessDecs;
use super::rc::RCImmixCollectRootEdges;
use crate::plan::global::BasePlan;
use crate::plan::global::CommonPlan;
use crate::plan::global::CreateGeneralPlanArgs;
use crate::plan::global::CreateSpecificPlanArgs;
use crate::plan::AllocationSemantics;
use crate::plan::Plan;
use crate::plan::PlanConstraints;
use crate::policy::immix::ImmixSpaceArgs;
use crate::policy::space::Space;
use crate::scheduler::gc_work::Release;
use crate::scheduler::gc_work::StopMutators;
use crate::scheduler::gc_work::UnsupportedProcessEdges;
use crate::scheduler::*;
use crate::util::alloc::allocators::AllocatorSelector;
use crate::util::copy::*;
use crate::util::heap::gc_trigger::SpaceStats;
use crate::util::heap::VMRequest;
use crate::util::metadata::side_metadata::SideMetadataContext;
use crate::util::rc::RefCountHelper;
use crate::util::ObjectReference;
use crate::vm::ObjectModel;
use crate::vm::VMBinding;
use crate::LazySweepingJobsCounter;
use crate::{policy::immix::ImmixSpace, util::opaque_pointer::VMWorkerThread};
use crossbeam::queue::SegQueue;
use std::sync::atomic::AtomicBool;
use std::sync::RwLock;

use super::Pause;
use atomic::Atomic;
use atomic::Ordering;
use enum_map::EnumMap;

use mmtk_macros::{HasSpaces, PlanTraceObject};

// ── Backup-trace (cycle collector) trigger state + knobs (P5) ──────────────────────────────────
//
// The trigger is RC-EFFECTIVENESS-based, NOT occupancy-based. The key distinction: heap occupancy
// at pause start cannot tell "full of RECLAIMABLE nursery garbage" (binarytrees — the imminent RC
// pause will free it, no trace needed) from "full of UN-reclaimable cycles" (kb — RC frees little,
// the trace IS needed); both look ">90% used". So instead we measure how much the RC pause's
// nursery+mature sweeps ACTUALLY freed (in `end_of_gc`, after the sweeps) and only schedule the
// NEXT pause as Full when an RC pause UNDER-reclaimed: it left the heap still near-full AND freed
// little. For binarytrees the RC sweep frees ~1500 blocks → occupancy drops → no backup; for kb the
// RC sweep frees little → occupancy stays high → backup. An absolute pressure backstop forces a
// Full just before OOM so we never run out even if the heuristic is fooled.

/// Set by `end_of_gc` when the just-finished RC pause under-reclaimed; consumed (→ Full) by the
/// next `select_collection_kind`.
static BACKUP_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// `get_used_pages()` captured at the START of the current pause (before the RC sweeps), so
/// `end_of_gc` can compute how much the RC pause's nursery+mature sweeps freed.
static USED_AT_PAUSE_START: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
/// Disable the backup trace entirely (pure RC) for A/B. `MMTK_RC_NO_CM` / `MMTK_RC_NO_BACKUP_TRACE`.
fn backup_trace_disabled() -> bool {
    std::env::var_os("MMTK_RC_NO_CM").is_some()
        || std::env::var_os("MMTK_RC_NO_BACKUP_TRACE").is_some()
}
/// "Under-reclaim" low-water mark: only consider a backup when post-RC-sweep occupancy is still ≥
/// this % of total (`MMTK_RC_BACKUP_LO_PCT`, default 80). Below this the heap has headroom; no trace.
fn backup_lo_pct() -> usize {
    env_usize("MMTK_RC_BACKUP_LO_PCT", 80)
}
/// The RC pause counts as "effective" (→ no backup) iff its nursery+mature sweeps freed at least
/// this % of total pages (`MMTK_RC_BACKUP_MIN_RECLAIM_PCT`, default 5). If it freed less AND the heap
/// is still ≥ lo-water, the heap is filling with RC-unreclaimable garbage → schedule Full.
fn backup_min_reclaim_pct() -> usize {
    env_usize("MMTK_RC_BACKUP_MIN_RECLAIM_PCT", 5)
}
/// Absolute pressure backstop: arm a backup when used ≥ this % of total AFTER the RC pause's
/// sweeps complete (`MMTK_RC_BACKUP_HI_PCT`, default 98), evaluated POST-SWEEP in
/// `evaluate_rc_effectiveness` — NOT at pause start. A GC fires because the heap filled, so
/// pause-start occupancy is always ~total and a pause-start backstop fires every pause (the bug
/// this replaces). Post-sweep, a still-critically-full heap genuinely means RC could not reclaim
/// it (cycles), so trace. binarytrees drops to ~13% post-sweep → never fires; kb stays full → fires
/// (and the rate signal usually fires first). `0` disables the backstop.
fn backup_hi_pct() -> usize {
    env_usize("MMTK_RC_BACKUP_HI_PCT", 98)
}

// LXR (reference counting on a hierarchical Immix heap) — P3.5 skeleton.
//
// Built UP from our base Immix plan (plan/immix/global.rs), NOT down from the
// reference's 1268-line global.rs (see LXR_PORT_PLAN.md). At this stage the LXR
// plan is a structural clone of Immix with `rc_enabled = false`, so selecting
// `MMTK_PLAN=LXR` runs identically to Immix and trips none of the P2
// `debug_assert(!rc_enabled)` guards. The reference-counting behaviour (field
// barrier install, RefCountHelper field, Pause::RefCount, ProcessIncs/Decs, the
// RC sweep machinery) is layered on incrementally in later P3 steps, only flipping
// `rc_enabled = true` once the machinery the asserts guard is actually present.
#[derive(HasSpaces, PlanTraceObject)]
pub struct LXR<VM: VMBinding> {
    #[post_scan]
    #[space]
    #[copy_semantics(CopySemantics::DefaultCopy)]
    pub immix_space: ImmixSpace<VM>,
    #[parent]
    pub common: CommonPlan<VM>,
    last_gc_was_defrag: AtomicBool,
    /// Reference-counting helper (RC_TABLE access + promote/dead bookkeeping). Inert
    /// until the RC trace (`ProcessIncs`/`ProcessDecs`) and the field barrier are wired
    /// and `rc_enabled` is flipped on; present now as the foundation those steps build on.
    #[allow(dead_code)]
    // read by the RC trace (ProcessIncs/ProcessDecs), wired in a later P3 step
    pub rc: RefCountHelper<VM>,
    /// The kind of the in-progress GC pause (`None` outside a pause). Set in `schedule_collection`,
    /// read by the RC trace + scheduling. Mirrors the same field on the ConcurrentImmix plan.
    current_pause: Atomic<Option<Pause>>,
    /// The kind of the previous GC pause. Deferred / always `None` in the minimal RC cut.
    #[allow(dead_code)]
    previous_pause: Atomic<Option<Pause>>,
    /// Roots collected in the PREVIOUS GC, kept alive by an extra reference count, to be
    /// decremented at the start of THIS GC (`process_prev_roots`). Each entry is one
    /// root-edge-scan's worth of root targets. `RwLock<SegQueue<..>>` matches the reference so the
    /// `release`-time swap (`mem::swap(prev, curr)`) is a cheap pointer swap.
    pub prev_roots: RwLock<SegQueue<Vec<ObjectReference>>>,
    /// Roots collected in THIS GC (pushed by the RC root scan); become `prev_roots` at release.
    pub curr_roots: RwLock<SegQueue<Vec<ObjectReference>>>,
}

/// The plan constraints for the LXR plan. **RC ACTIVATED**: `rc_enabled = true`,
/// `needs_field_log_bit = true`, `barrier = FieldBarrier`. `needs_log_bit` is also true (the
/// per-field unlog metadata the barrier indexes lives in the global field-unlog spec, which the
/// log-bit machinery maps).
///
/// `moves_objects = false`: the minimal single-domain RC cut promotes nursery objects **in place**
/// and never evacuates (mature evac / nursery copying are deferred), so the GC is non-moving. This
/// is intentionally stricter than the reference LXR (a moving GC) and keeps `sanity` honest (no
/// forwarding to validate).
pub const LXR_CONSTRAINTS: PlanConstraints = PlanConstraints {
    moves_objects: false,
    max_non_los_default_alloc_bytes: crate::policy::immix::MAX_IMMIX_OBJECT_SIZE,
    needs_log_bit: true,
    needs_field_log_bit: true,
    rc_enabled: true,
    barrier: crate::plan::barriers::BarrierSelector::FieldBarrier,
    ..PlanConstraints::default()
};

impl<VM: VMBinding> Plan for LXR<VM> {
    fn collection_required(&self, space_full: bool, _space: Option<SpaceStats<Self::VM>>) -> bool {
        self.base().collection_required(self, space_full)
    }

    fn last_collection_was_exhaustive(&self) -> bool {
        self.immix_space
            .is_last_gc_exhaustive(self.last_gc_was_defrag.load(Ordering::Relaxed))
    }

    fn constraints(&self) -> &'static PlanConstraints {
        &LXR_CONSTRAINTS
    }

    fn create_copy_config(&'static self) -> CopyConfig<Self::VM> {
        use enum_map::enum_map;
        CopyConfig {
            copy_mapping: enum_map! {
                CopySemantics::DefaultCopy => CopySelector::Immix(0),
                _ => CopySelector::Unused,
            },
            space_mapping: vec![(CopySelector::Immix(0), &self.immix_space)],
            constraints: &LXR_CONSTRAINTS,
        }
    }

    fn schedule_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        // Lazily wire `block_allocation`'s self-reference (idempotent).
        self.ensure_block_allocation_initialized();
        // NOTE: the pause-START phase-epoch bump is done in `notify_mutators_paused` (after all
        // mutators are stopped), NOT here — `schedule_collection` runs while mutators are still
        // allocating, and bumping to the even (GC-phase) epoch here would let a still-running
        // mutator stamp a freshly-allocated block with an EVEN epoch, violating the odd=mutator
        // invariant (`init_rc` asserts it) and mis-classifying that live block as non-nursery →
        // the nursery sweep could then free a LIVE block (SIGSEGV).
        // Snapshot pre-sweep occupancy so `end_of_gc` can measure how much THIS pause's RC sweeps
        // freed (the RC-effectiveness backup trigger).
        USED_AT_PAUSE_START.store(self.get_used_pages(), Ordering::Relaxed);
        // RefCount = steady-state in-place RC. Full = the periodic STW backup mark/sweep that
        // reclaims cyclic garbage RC misses (no concurrent marking; the in-place STW cut).
        let pause = self.select_collection_kind();
        self.current_pause.store(Some(pause), Ordering::SeqCst);
        match pause {
            Pause::RefCount => self.schedule_rc_collection(scheduler),
            Pause::Full => self.schedule_full_collection(scheduler),
            _ => unreachable!(
                "LXR cut only schedules RefCount + Full pauses, got {:?}",
                pause
            ),
        }
    }

    // NOTE: there is intentionally NO pause-start phase-epoch bump. We use the SINGLE-bump scheme
    // (one `update_global_phase_epoch` per GC, at the END of the epilogue). The two-bumps-per-GC
    // scheme I tried — adding a pause-start bump in `notify_mutators_paused` — REGRESSED the pause:
    // with freeing fully off it SIGSEGV'd (h64) while single-bump was clean. The pause-start bump
    // makes the epoch even DURING the inc phase, which flips `is_nursery_or_reusing()` while
    // `promote()`/`set_as_in_place_promoted()` are writing per-block metadata, corrupting block
    // state. Commit 1711b5e48f was single-bump AND sanity-clean (zero Invalid reference), so the
    // single-bump scheme is the provably-correct one. (Trade-off: a clean nursery block allocated in
    // the *next* mutator phase carries an even epoch under single-bump, so cross-GC nursery
    // classification is imperfect — but `rc_dead()` still gates every free, so this can only under-
    // reclaim (leak), never free a live block.)

    fn get_allocator_mapping(&self) -> &'static EnumMap<AllocationSemantics, AllocatorSelector> {
        &ALLOCATOR_MAPPING
    }

    fn prepare(&mut self, tls: VMWorkerThread) {
        // RC pause prepare (run from FastRCPrepare in the RCProcessIncs bucket — the normal Prepare
        // bucket is disabled for a RefCount pause; for a Full pause it runs in the Prepare bucket).
        let pause = self.current_pause().unwrap();
        debug_assert!(pause == Pause::RefCount || pause == Pause::Full);
        // `false` = not a full/major-heap *mark-based* prepare for the common spaces (immortal/LOS
        // are RC-managed). The Full backup trace marks the immix graph; the common spaces don't need
        // the mark-based prepare.
        self.common.prepare(tls, false);
        self.immix_space.prepare_rc(pause);
    }

    fn release(&mut self, tls: VMWorkerThread) {
        let pause = self.current_pause().unwrap();
        debug_assert!(pause == Pause::RefCount || pause == Pause::Full);
        self.common.release(tls, false);
        self.immix_space.release_rc(pause);
        // Swap roots: this GC's collected roots become next GC's prev_roots (to be decremented).
        {
            let mut prev_roots = self.prev_roots.write().unwrap();
            let mut curr_roots = self.curr_roots.write().unwrap();
            std::mem::swap::<SegQueue<_>>(&mut prev_roots, &mut curr_roots);
            debug_assert!(curr_roots.is_empty());
        }
        // NOTE: the release-end phase-epoch bump (GC→mutator) is done at the END of
        // `RCBlockSweepEpilogue`, NOT here. `release` runs in the `Release` bucket, which is BEFORE
        // `STWRCDecsAndSweep` (the decs) and the epilogue (the nursery sweep). The nursery sweep
        // classifies blocks by the GC-phase (even) epoch, so the epoch must stay un-bumped until
        // after that sweep runs; bumping here (to the next odd mutator epoch) would make the
        // epilogue's `is_nursery()` mis-classify every block.
    }

    fn end_of_gc(&mut self, tls: VMWorkerThread) {
        self.evaluate_rc_effectiveness();
        self.dump_rc_stats();
        self.previous_pause
            .store(self.current_pause(), Ordering::SeqCst);
        self.current_pause.store(None, Ordering::SeqCst);
        self.last_gc_was_defrag.store(false, Ordering::Relaxed);
        self.common.end_of_gc(tls);
    }

    fn current_gc_may_move_object(&self) -> bool {
        // Minimal RC cut: in-place promotion only, never moves objects.
        false
    }

    fn get_collection_reserved_pages(&self) -> usize {
        self.immix_space.defrag_headroom_pages()
    }

    fn get_used_pages(&self) -> usize {
        self.immix_space.reserved_pages() + self.common.get_used_pages()
    }

    fn base(&self) -> &BasePlan<VM> {
        &self.common.base
    }

    fn base_mut(&mut self) -> &mut BasePlan<Self::VM> {
        &mut self.common.base
    }

    fn common(&self) -> &CommonPlan<VM> {
        &self.common
    }
}

impl<VM: VMBinding> LXR<VM> {
    pub fn new(args: CreateGeneralPlanArgs<VM>) -> Self {
        // P4: map the VM-side global log bit (needs_log_bit) + the per-field unlog bit
        // (needs_field_log_bit) the coalescing field barrier uses. The binding lays the
        // field-unlog spec `side_after` the log bit, so they occupy disjoint regions.
        let mut spec = crate::util::metadata::extract_side_metadata(&[
            *VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC,
            *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC,
        ]);
        // RC_TABLE is a global side-metadata spec (whole-heap reference counts), but it is only
        // DEFINED in spec_defs — no plan/space registered it in its global metadata context, so its
        // per-chunk pages were never mapped/committed. Without this, rc.count/rc.inc fault reading
        // uncommitted metadata on the first RC pause (the object is valid; only its RC metadata is
        // uncommitted). Register it so the whole heap's RC counts are backed.
        spec.push(crate::util::rc::RC_TABLE);
        let mut plan_args = CreateSpecificPlanArgs {
            global_args: args,
            constraints: &LXR_CONSTRAINTS,
            global_side_metadata_specs: SideMetadataContext::new_global_specs(&spec),
        };
        let lxr = LXR {
            immix_space: ImmixSpace::new(
                plan_args.get_normal_space_args("immix", true, false, VMRequest::discontiguous()),
                ImmixSpaceArgs {
                    mixed_age: false,
                    // Minimal RC cut: in-place promotion only, never evacuate. Matches
                    // `LXR_CONSTRAINTS.moves_objects = false`.
                    never_move_objects: true,
                },
            ),
            common: CommonPlan::new(plan_args),
            last_gc_was_defrag: AtomicBool::new(false),
            rc: RefCountHelper::NEW,
            current_pause: Atomic::new(None),
            previous_pause: Atomic::new(None),
            prev_roots: RwLock::new(SegQueue::new()),
            curr_roots: RwLock::new(SegQueue::new()),
        };

        lxr.verify_side_metadata_sanity();

        lxr
    }

    /// One-time hook to give `block_allocation` its self-reference + space pointer. Idempotent;
    /// run lazily at the top of `schedule_collection` (the first `&'static self` entry point). The
    /// reference does this in a `gc_init` hook, which our base plan-creation flow lacks.
    fn ensure_block_allocation_initialized(&'static self) {
        if self.immix_space.block_allocation.lxr.is_none() {
            // # Safety: `self` is `&'static` (the plan is boxed-and-leaked into MMTK), and the
            // `block_allocation` lives inside `self.immix_space`, so these pointers are stable for
            // the plan's whole lifetime. We mutate through a shared ref because `block_allocation`'s
            // `lxr`/`space` are set-once; no concurrent writer (this runs at the start of a STW
            // collection, single-threaded).
            #[allow(invalid_reference_casting)]
            let ba = unsafe {
                &mut *(&self.immix_space.block_allocation
                    as *const crate::policy::immix::block_allocation::BlockAllocation<VM>
                    as *mut crate::policy::immix::block_allocation::BlockAllocation<VM>)
            };
            ba.init(&self.immix_space);
            ba.lxr = Some(self);
        }
    }
}

// ── LXR RC-trace support (P3) ─────────────────────────────────────────────────────────────────
// The reference-counting work packets (`ProcessIncs`/`ProcessDecs`/`RCImmixCollectRootEdges`),
// the field barrier, and `block_allocation.rs` query the plan through this small surface. In the
// minimal single-domain RC cut the CM/defrag-flavoured queries are hard `false`/`None` (concurrent
// marking, mature evacuation, and lazy decrements are deferred). RC is now ACTIVE (`rc_enabled =
// true`): the RefCount pause runs these on `MMTK_PLAN=LXR`.
#[allow(dead_code)]
impl<VM: VMBinding> LXR<VM> {
    /// The large-object space (RC dec'd objects that overflow immix live here). Returns the shared
    /// CommonPlan LOS — note the *RC-aware* LOS nursery tracking / `rc_free` is deferred, so this is
    /// the standard mark-swept LOS for now.
    pub fn los(&self) -> &crate::policy::largeobjectspace::LargeObjectSpace<VM> {
        self.common.get_los()
    }

    /// Is a concurrent-marking (SATB) cycle in progress? Deferred under `lxr_no_cm` → always false.
    pub fn cm_in_progress(&self) -> bool {
        false
    }

    /// Is concurrent marking enabled at all? Deferred (`lxr_no_cm`) → always false.
    pub fn cm_enabled(&self) -> bool {
        false
    }

    /// Convenience used by the reference's `block_allocation` cm-gate. Deferred → false.
    pub fn cm_in_progress_or_final_mark(&self) -> bool {
        self.cm_in_progress() || self.current_pause() == Some(Pause::FinalMark)
    }

    /// The kind of the in-progress pause, or `None` outside a pause. Set in `schedule_collection`,
    /// cleared in `end_of_gc`.
    pub fn current_pause(&self) -> Option<Pause> {
        self.current_pause.load(Ordering::Relaxed)
    }

    /// The kind of the previous pause. Deferred / always `None` in the minimal RC cut.
    pub fn previous_pause(&self) -> Option<Pause> {
        self.previous_pause.load(Ordering::Relaxed)
    }

    /// Is `object` marked (live) by the mark bit? Forwarded to the immix space's RC read-side.
    pub fn is_marked(&self, object: ObjectReference) -> bool {
        self.immix_space.is_marked(object)
    }

    /// Atomically mark `object` (0→1); returns true iff this call did the marking. Only used by the
    /// CM/SATB dec path (deferred), so inert in the minimal cut.
    pub fn mark(&self, object: ObjectReference) -> bool {
        self.immix_space.attempt_mark_rc(object)
    }

    /// Is `object` in a block selected for mature defrag-evacuation? Deferred (no mature evac) → false.
    pub fn in_defrag(&self, _object: ObjectReference) -> bool {
        false
    }

    /// Is `addr` in a defrag-source block? Deferred (no mature evac) → false.
    pub fn address_in_defrag(&self, _addr: crate::util::Address) -> bool {
        false
    }
}

// ── LXR RC pause scheduling (P3 activation) ───────────────────────────────────────────────────
impl<VM: VMBinding> LXR<VM> {
    /// Decide the pause kind. Steady state is `RefCount` (in-place RC). Every `K` RC pauses (or when
    /// the previous RC pause failed to free enough and the heap is near-full) we run a `Pause::Full`
    /// — a STW backup mark/sweep that reclaims the CYCLIC garbage pure RC cannot. `K` =
    /// `MMTK_RC_BACKUP_EVERY` (default 16); `MMTK_RC_NO_CM`/`MMTK_RC_NO_BACKUP_TRACE` disables it
    /// entirely (pure RC, for A/B). The pressure trigger fires a Full when the heap is > ~90% used
    /// (so kb-style cyclic accumulation gets collected before OOM instead of after).
    fn select_collection_kind(&self) -> Pause {
        if backup_trace_disabled() {
            return Pause::RefCount;
        }
        // The RC-effectiveness signal (armed in `end_of_gc`, AFTER a pause's sweeps — including the
        // post-sweep pressure backstop) is the SOLE trigger. We must NOT test pause-start occupancy
        // here: a GC fires precisely because the heap filled, so `used` is always ~total at pause
        // start, and a pause-start backstop would fire every pause and defeat the signal (the bug
        // that made binarytrees trace on 115/115 pauses).
        if BACKUP_PENDING.swap(false, Ordering::Relaxed) {
            Pause::Full
        } else {
            Pause::RefCount
        }
    }

    /// Disable the work buckets a pause does not use. For `RefCount` we additionally disable
    /// `Closure` (the RC root packets route to `RCProcessIncs`, so `Closure` carries no RC work);
    /// for `Full` we LEAVE `Closure` ENABLED — the backup trace's transitive MARK closure runs
    /// there. The ref-closure / forwarding / compact stages are unused by either and stay disabled.
    fn disable_unnecessary_buckets(&self, scheduler: &GCWorkScheduler<VM>, pause: Pause) {
        use WorkBucketStage::*;
        for stage in [
            SoftRefClosure,
            WeakRefClosure,
            FinalRefClosure,
            PhantomRefClosure,
            CalculateForwarding,
            SecondRoots,
            RefForwarding,
            FinalizableForwarding,
            Compact,
        ] {
            scheduler.work_buckets[stage].set_enabled(false);
        }
        // Closure: disabled for RefCount (no RC work there), ENABLED for Full (the mark closure).
        scheduler.work_buckets[Closure].set_enabled(pause == Pause::Full);
    }

    /// Wrap the previous GC's roots into `ProcessDecs` packets (their extra root reference count is
    /// dropped now). The minimal cut runs decs stop-the-world in `STWRCDecsAndSweep` (lazy decrements
    /// are deferred). Always schedules at least one (possibly empty) packet so the bucket opens.
    fn process_prev_roots(&self, scheduler: &GCWorkScheduler<VM>) {
        let prev_roots = self.prev_roots.write().unwrap();
        if super::rc::rc_debug_on() {
            let n_pkts = prev_roots.len();
            // SegQueue::len is O(1); count total root targets across packets non-destructively is
            // not cheap, so just report packet count (matches the [RC-STATS] prev/curr).
            eprintln!("[RC-PREV-ROOTS] process_prev_roots: prev_root_packets={n_pkts}");
        }
        let mut work_packets: Vec<Box<dyn GCWork<VM>>> = Vec::with_capacity(prev_roots.len());
        while let Some(decs) = prev_roots.pop() {
            work_packets.push(Box::new(ProcessDecs::new(
                decs,
                LazySweepingJobsCounter::new_decs(),
            )));
        }
        if work_packets.is_empty() {
            work_packets.push(Box::new(ProcessDecs::new(
                vec![],
                LazySweepingJobsCounter::new_decs(),
            )));
        }
        scheduler.work_buckets[WorkBucketStage::STWRCDecsAndSweep].bulk_add(work_packets);
    }

    /// Schedule a RefCount pause. Adapted from lxr-v0.32.0 `schedule_rc_collection`:
    /// (1) disable unused buckets, (2) decrement the previous GC's roots, (3) StopMutators (whose
    /// root scan flows through `RCImmixCollectRootEdges` → `ProcessIncs<ROOT>`), (4) `FastRCPrepare`
    /// (runs `prepare_rc`) in `RCProcessIncs`, (5) `Release` in `Release`. CM/mature-evac packets are
    /// omitted (deferred).
    /// Dump per-pause RC reclamation stats + reset them (MMTK_RC_DEBUG). Lets the reclamation
    /// balance be tracked across GCs: incs vs decs, objects promoted vs reaching 0, blocks freed,
    /// and the prev/curr root-set sizes (the root-dec balance).
    /// RC-effectiveness backup trigger (runs in `end_of_gc`, AFTER the RC sweeps). If the just-
    /// finished RC pause UNDER-reclaimed — it freed < `MMTK_RC_BACKUP_MIN_RECLAIM_PCT`% of total
    /// pages AND left the heap still ≥ `MMTK_RC_BACKUP_LO_PCT`% used — the heap is filling with
    /// RC-unreclaimable garbage (cycles), so arm `BACKUP_PENDING` for the next pause. A Full pause is
    /// NOT re-evaluated (it just traced; don't chain backups). For binarytrees the RC sweep frees a
    /// large fraction → effective → never arms; for kb it frees little → arms → cycles get traced.
    fn evaluate_rc_effectiveness(&self) {
        if backup_trace_disabled() || self.current_pause() != Some(Pause::RefCount) {
            return;
        }
        let total = self.get_total_pages();
        if total == 0 {
            return;
        }
        let used_after = self.get_used_pages();
        let used_before = USED_AT_PAUSE_START.load(Ordering::Relaxed);
        let freed = used_before.saturating_sub(used_after);
        let still_full = used_after * 100 >= total * backup_lo_pct();
        let under_reclaimed = freed * 100 < total * backup_min_reclaim_pct();
        // Post-sweep pressure backstop: even if the pause reclaimed a fair amount, a heap still
        // critically full AFTER the RC sweep needs a trace (cycles that slipped past the rate
        // signal). Checked here (post-sweep), where occupancy is meaningful — never at pause start.
        let hi = backup_hi_pct();
        let critical = hi > 0 && used_after * 100 >= total * hi;
        if (still_full && under_reclaimed) || critical {
            BACKUP_PENDING.store(true, Ordering::Relaxed);
        }
        if super::rc::rc_debug_on() {
            eprintln!(
                "[RC-EFFECT] used_before={used_before} used_after={used_after} freed={freed} \
                 total={total} still_full={still_full} under_reclaimed={under_reclaimed} \
                 backup_next={}",
                still_full && under_reclaimed
            );
        }
    }

    fn dump_rc_stats(&self) {
        use super::rc::{
            rc_debug_on, RC_DECS_TOTAL, RC_DECS_TO_ZERO, RC_INCS_PROMOTED, RC_INCS_TOTAL,
        };
        if !rc_debug_on() {
            return;
        }
        let load = |c: &std::sync::atomic::AtomicUsize| c.swap(0, Ordering::Relaxed);
        let incs = load(&RC_INCS_TOTAL);
        let promoted = load(&RC_INCS_PROMOTED);
        let decs = load(&RC_DECS_TOTAL);
        let dead = load(&RC_DECS_TO_ZERO);
        let young = self
            .immix_space
            .num_clean_blocks_released_young
            .swap(0, Ordering::Relaxed);
        let mature = self
            .immix_space
            .num_clean_blocks_released_mature
            .swap(0, Ordering::Relaxed);
        let prev_roots = self.prev_roots.read().unwrap().len();
        let curr_roots = self.curr_roots.read().unwrap().len();
        // `dump_rc_stats` runs in `end_of_gc` BEFORE `current_pause` is cleared, so it still reflects
        // THIS GC's pause kind.
        let kind = match self.current_pause() {
            Some(Pause::Full) => "FULL(backup-trace)",
            _ => "RC",
        };
        eprintln!(
            "[RC-STATS] {kind} incs={incs} (promoted={promoted}) decs={decs} (dead={dead}) \
             blocks_freed: young={young} mature={mature}  root-packets: prev={prev_roots} curr={curr_roots}"
        );
    }

    fn schedule_rc_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        self.disable_unnecessary_buckets(scheduler, Pause::RefCount);
        self.process_prev_roots(scheduler);
        type RootEdges<VM> = RCImmixCollectRootEdges<VM>;
        // Plain `add` (not `add_prioritized`): our base's `Unconstrained` bucket has no
        // prioritized queue (the reference's does), and the base schedules StopMutators the
        // same way (scheduler.rs). The prioritization was only a latency optimisation.
        scheduler.work_buckets[WorkBucketStage::Unconstrained]
            .add(StopMutators::<LXRRCWorkContext<RootEdges<VM>>>::new());
        scheduler.work_buckets[WorkBucketStage::RCProcessIncs].add(FastRCPrepare);
        scheduler.work_buckets[WorkBucketStage::Release]
            .add(Release::<LXRRCWorkContext<UnsupportedProcessEdges<VM>>>::new(self));
        // After ALL decrements (incl. the recursive cascade) drain, sweep the now-dead MATURE
        // blocks that the decs queued into `possibly_dead_mature_blocks`. A bucket sentinel runs
        // exactly once the bucket has emptied (the STW equivalent of the reference's
        // `LazySweepingJobsCounter::end_of_decs` Drop callback, which we deferred). Without this,
        // dead mature blocks are queued but never returned to the free list → the reclamation leak.
        scheduler.work_buckets[WorkBucketStage::STWRCDecsAndSweep]
            .set_sentinel(Box::new(super::gc_work::RCBlockSweepEpilogue));
    }

    /// Schedule a Full (backup-trace) pause. This is the steady-state RC pause PLUS a STW backup
    /// mark/sweep that reclaims cyclic garbage:
    ///   - mark-table zeroing (Unconstrained) so the trace marks into a clean table;
    ///   - `Closure` re-enabled — the root scan (`RCImmixCollectRootEdges`) both INCREMENTS roots
    ///     (RC, preserving the root-dec balance) AND seeds a transitive MARK closure in `Closure`;
    ///   - `prepare_rc(Full)` sets `is_end_of_satb_or_full_gc` so liveness consults the mark bit;
    ///   - after the closure + decs drain, `RCBlockSweepEpilogue` runs the nursery + mature sweeps
    ///     and (for Full) the `SweepDeadCycles` dead-cycle sweep, then bumps the epoch.
    /// No concurrent marking, no copying — pure STW, matching the in-place cut.
    fn schedule_full_collection(&'static self, scheduler: &GCWorkScheduler<VM>) {
        self.disable_unnecessary_buckets(scheduler, Pause::Full);
        // Clean the object mark table before the trace marks into it.
        self.immix_space.schedule_mark_table_zeroing();
        self.process_prev_roots(scheduler);
        type RootEdges<VM> = RCImmixCollectRootEdges<VM>;
        scheduler.work_buckets[WorkBucketStage::Unconstrained]
            .add(StopMutators::<LXRRCWorkContext<RootEdges<VM>>>::new());
        scheduler.work_buckets[WorkBucketStage::RCProcessIncs].add(FastRCPrepare);
        scheduler.work_buckets[WorkBucketStage::Release]
            .add(Release::<LXRRCWorkContext<UnsupportedProcessEdges<VM>>>::new(self));
        // Same post-decs epilogue; it runs the dead-cycle sweep too because current_pause == Full.
        scheduler.work_buckets[WorkBucketStage::STWRCDecsAndSweep]
            .set_sentinel(Box::new(super::gc_work::RCBlockSweepEpilogue));
    }
}
