//! LXR reference-counting work packets — minimal single-domain port (P3).
//!
//! Vendored and **heavily adapted** (streamlined to the minimal subset) from the LXR research
//! fork's `plan/lxr/rc.rs`. This is the "RC heart": the increment processor (`ProcessIncs`), the
//! decrement processor (`ProcessDecs`), and the root-edge → root-increment bridge
//! (`RCImmixCollectRootEdges`). Everything is **present + compiling but INERT** — the LXR plan ships
//! with `rc_enabled = false`, so the field barrier that would enqueue these packets is not installed
//! and `current_pause()` stays `None`; nothing here runs until the main loop flips `rc_enabled`.
//!
//! ## What was kept vs the reference (single-domain, in-place-promotion-only cut)
//!
//! The reference interleaves four concerns we are **deferring**: nursery/mature **evacuation**
//! (copying), **concurrent marking** (CM/SATB), **lazy decrements** (the global sweeping-jobs
//! registry), and a large body of **instrumentation**. The minimal cut here is the
//! `RC_NURSERY_EVACUATION = false`, `lxr_no_cm`, `lxr_no_mature_evac` configuration, which collapses:
//!
//! * `process_inc_and_evacuate` → **inc, and if the object was freshly promoted (RC 0→1), promote it
//!   in place**. No forwarding, no copy context, no `NO_EVAC` throttle. Since objects never move,
//!   the slot is never written back.
//! * `scan_nursery_object` → set the freshly-promoted object's per-field unlog bits (so the field
//!   barrier won't re-log them) and generate recursive increments for its pointer fields. The
//!   compressed-pointer / val-array / huge-obj-array special cases are dropped (OCaml is 64-bit,
//!   uncompressed; field iteration goes through the base `SlotIterator`).
//! * `process_dead_object` → recursively decrement fields, clear straddle-line metadata, and hand
//!   the now-(maybe-)dead block to the lazy mature sweep. The CM/SATB mark push is dropped.
//!
//! ## Base-API adaptations (vs the LXR fork)
//!
//! | LXR call | base replacement |
//! |---|---|
//! | `o.get_size::<VM>()` | `VM::VMObjectModel::get_current_size(o)` |
//! | `o.verify::<VM>()` | dropped (no such method on our `ObjectReference`) |
//! | `o.iterate_fields::<VM,_>(CLDScanPolicy, RefScanPolicy, |slot, out_of_heap| …)` | `crate::plan::tracing::SlotIterator::<VM>::iterate_fields(o, tls, |slot| …)`; `out_of_heap` is derived as `!immix_space.in_space(target)` |
//! | `Block::containing(o)` | same, but via the `Region` trait import |
//! | `GCWorker::current()` | a `*mut GCWorker` captured at `do_work` start (our base has no thread-local current-worker accessor; matches the `ProcessEdgesBase` pattern) |
//! | `rc.fetch_update(o, closure)` for the dec | a manual CAS loop — our `RefCountHelper::fetch_update` bounds the closure `+ Copy`, which a `&mut self`-capturing closure is not |
//! | `s.store(Some(new))` | dropped — the in-place cut never moves objects |
//! | CM (`super::cm::*`), forwarding, copy context, survival predictor, counters/prefetch, `RootKind`/`RC_ROOTS`, `curr_roots` | dropped |

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::LXR;
use crate::plan::tracing::SlotIterator;
use crate::plan::VectorQueue;
use crate::policy::immix::block::Block;
use crate::policy::space::Space;
use crate::scheduler::gc_work::{ProcessEdgesBase, ScanObjects, SlotOf};
use crate::scheduler::{GCWork, GCWorker, ProcessEdgesWork, WorkBucketStage};
use crate::util::linear_scan::Region;
use crate::util::rc::{RefCountHelper, MAX_REF_COUNT};
use crate::util::{ObjectReference, VMThread};
use crate::vm::slot::Slot;
use crate::vm::*;
use crate::LazySweepingJobsCounter;
use crate::MMTK;

// ── RC reclamation instrumentation (MMTK_RC_DEBUG) ────────────────────────────────────────────
// Per-pause counters, dumped by `LXR::dump_rc_stats` at end_of_gc when MMTK_RC_DEBUG is set, so the
// reclamation balance can be tracked across GCs. Cheap relaxed atomics; reset each pause.
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

pub static RC_INCS_PROMOTED: AtomicUsize = AtomicUsize::new(0); // objects whose RC went 0->1 (promoted)
pub static RC_INCS_TOTAL: AtomicUsize = AtomicUsize::new(0); // total inc() calls
pub static RC_DECS_TOTAL: AtomicUsize = AtomicUsize::new(0); // total dec attempts processed
pub static RC_DECS_TO_ZERO: AtomicUsize = AtomicUsize::new(0); // objects whose RC reached 0 (dead)

#[inline(always)]
pub(super) fn rc_stat_inc(counter: &AtomicUsize) {
    if cfg!(debug_assertions) || crate::plan::lxr::rc::rc_debug_on() {
        counter.fetch_add(1, AtomicOrdering::Relaxed);
    }
}

/// Cached `MMTK_RC_DEBUG` flag (env read once).
pub(crate) fn rc_debug_on() -> bool {
    use std::sync::atomic::AtomicU8;
    static STATE: AtomicU8 = AtomicU8::new(0); // 0=unknown, 1=off, 2=on
    match STATE.load(AtomicOrdering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("MMTK_RC_DEBUG").is_some();
            STATE.store(if on { 2 } else { 1 }, AtomicOrdering::Relaxed);
            on
        }
    }
}

/// The classification of an increment's source slot.
pub type EdgeKind = u8;
/// A root slot (scanned from the stack/registers). Roots are never written back. (Used by
/// `RCImmixCollectRootEdges` once the RC root scan is wired — main loop.)
#[allow(dead_code)]
pub const EDGE_KIND_ROOT: u8 = 0;
/// A slot of a freshly-promoted nursery object (recursive increment).
pub const EDGE_KIND_NURSERY: u8 = 1;
/// A mature heap slot (logged by the field barrier). Mature slots are unlogged on load.
pub const EDGE_KIND_MATURE: u8 = 2;

/// A fake TLS for the base `SlotIterator` (it ignores the tls; see the FIXME there).
#[inline(always)]
fn fake_tls() -> VMThread {
    VMThread::UNINITIALIZED
}

/// MMTK_RC_DEBUG diagnostic: before doing RC metadata work on `o`, verify it is a real, mapped,
/// in-space heap object. If not, print the bogus reference + the call site and abort cleanly
/// (instead of a raw SIGSEGV deep in `atomic_load`), so the offending path is pinpointed.
#[inline(always)]
fn debug_rc_validate(site: &str, o: ObjectReference) {
    if !cfg!(debug_assertions) && std::env::var_os("MMTK_RC_DEBUG").is_none() {
        return;
    }
    let a = o.to_raw_address();
    let ok = a.is_mapped() && crate::memory_manager::is_in_mmtk_spaces(o);
    if !ok {
        eprintln!(
            "[RC-BOGUS] {site}: object {:?} (addr {:#x}) is NOT a valid in-space mapped object \
             (is_mapped={}, in_spaces={})",
            o,
            a.as_usize(),
            a.is_mapped(),
            crate::memory_manager::is_in_mmtk_spaces(o),
        );
        std::process::abort();
    }
}

// ───────────────────────────────────── ProcessIncs ──────────────────────────────────────────────

/// Process a buffer of reference-count increments. `KIND` distinguishes root / nursery / mature
/// slots (it only changes whether the slot is unlogged on load).
pub struct ProcessIncs<VM: VMBinding, const KIND: EdgeKind> {
    /// Increments (slots) to process.
    incs: Vec<VM::VMSlot>,
    /// Recursively-generated new increments (fields of freshly-promoted objects).
    new_incs: VectorQueue<VM::VMSlot>,
    new_incs_count: u32,
    lxr: &'static LXR<VM>,
    rc: RefCountHelper<VM>,
    /// The worker running this packet (captured at `do_work` start; null until then). Used to
    /// enqueue recursively-generated nursery-inc packets.
    worker: *mut GCWorker<VM>,
    /// For `KIND == EDGE_KIND_ROOT` only: the root *targets* (the objects loaded from the root
    /// slots). These get an extra "root" reference count this GC; they are stashed into
    /// `lxr.curr_roots` so the NEXT GC decrements them (`process_prev_roots`). Without this the root
    /// increment is never matched by a decrement and root-reachable objects leak forever.
    root_targets: Vec<ObjectReference>,
}

unsafe impl<VM: VMBinding, const KIND: EdgeKind> Send for ProcessIncs<VM, KIND> {}

#[allow(dead_code)]
impl<VM: VMBinding, const KIND: EdgeKind> ProcessIncs<VM, KIND> {
    const CAPACITY: usize = crate::args::BUFFER_SIZE;

    fn worker(&self) -> &'static mut GCWorker<VM> {
        unsafe { &mut *self.worker }
    }

    pub fn new(incs: Vec<VM::VMSlot>, lxr: &'static LXR<VM>) -> Self {
        Self {
            incs,
            new_incs: VectorQueue::default(),
            new_incs_count: 0,
            lxr,
            rc: RefCountHelper::NEW,
            worker: std::ptr::null_mut(),
            root_targets: Vec::new(),
        }
    }

    /// Increment `o`'s RC. Returns true iff this call promoted it (0 → 1).
    fn inc(&self, o: ObjectReference) -> bool {
        rc_stat_inc(&RC_INCS_TOTAL);
        let promoted = self.rc.inc(o) == Ok(0);
        if promoted {
            rc_stat_inc(&RC_INCS_PROMOTED);
        }
        promoted
    }

    /// Promote a freshly-incremented object to mature: mark its block as in-place-promoted (if it
    /// is a fresh nursery block), set its straddle-line metadata, and scan it to set field unlog
    /// bits + generate recursive increments. No copying (in-place-only cut).
    ///
    /// `in_immix` = the object is in the immix space. The per-block / straddle-line bookkeeping
    /// (`is_nursery`, `set_as_in_place_promoted`, `promote_with_size`) reads LOCAL side metadata
    /// that is mapped ONLY for the immix space's chunks, so it must be skipped for objects in the
    /// LOS / immortal / non-moving spaces (whose addresses index unmapped immix-block metadata →
    /// SIGSEGV). RC_TABLE is global (whole-heap mapped), so the inc itself is always safe.
    fn promote(&mut self, o: ObjectReference, in_immix: bool) {
        let size = VM::VMObjectModel::get_current_size(o);
        if in_immix {
            let block = Block::containing(o);
            // Flip a not-yet-promoted nursery block to mature (Unmarked). Key on the block STATE
            // (`Unallocated` = a clean nursery block this phase that hasn't been promoted yet), NOT
            // `is_nursery()` — under the single-bump scheme `is_nursery()`'s epoch comparison
            // mis-classifies blocks from the 2nd mutator phase onward, which would leave surviving
            // blocks stuck in `Unallocated` and untracked (a leak). `set_as_in_place_promoted` is
            // idempotent (its own `is_in_place_promoted` guard), so re-entry is harmless.
            if block.get_state() == crate::policy::immix::block::BlockState::Unallocated {
                block.set_as_in_place_promoted(&self.lxr.immix_space);
            }
            self.rc.promote_with_size(o, size);
        }
        self.scan_nursery_object(o, in_immix);
    }

    /// Scan a freshly-promoted object: set its per-field unlog bits (so the field write barrier
    /// won't re-log them — they are now mature) and generate a recursive increment for each pointer
    /// field, bumping already-live children directly. `in_immix` distinguishes the immix path from
    /// the LOS/immortal/non-moving path (the latter unlogs the header, not the per-field bulk path).
    fn scan_nursery_object(&mut self, o: ObjectReference, in_immix: bool) {
        if !in_immix {
            o.to_raw_address().unlog_field_relaxed::<VM>();
        }
        SlotIterator::<VM>::iterate_fields(o, fake_tls(), |slot| {
            // Unlog this field (it now belongs to a mature object) — but ONLY if the slot address is
            // in an MMTk space. A `Cont_tag` object's `iterate_fields` yields fiber-stack slot
            // addresses (mmap'd/caml_stat_alloc'd, outside MMTk spaces; runtime/fiber.c); their
            // GLOBAL_FIELD_UNLOG side-metadata page is unmapped, so unlogging them faults (the
            // chameneos-under-LXR SIGSEGV). Stacks are not field-barrier-tracked, so skipping the
            // unlog is correct. Same cheap no-deref SFT guard used for objects at :252/:276/:596
            // (cf. the GH#15 FieldSlot::load re-check).
            let sa = slot.to_address();
            if sa.is_mapped()
                && crate::memory_manager::is_in_mmtk_spaces(unsafe {
                    ObjectReference::from_raw_address_unchecked(sa)
                })
            {
                sa.unlog_field_relaxed::<VM>();
            }
            let Some(target) = slot.load() else {
                return;
            };
            debug_rc_validate("scan_nursery_object.field", target);
            let rc = self.rc.count(target);
            if rc == 0 {
                // Fresh nursery child — defer a recursive increment (it will itself promote).
                self.new_incs.push(slot);
                self.new_incs_count += 1;
            } else if rc != MAX_REF_COUNT {
                // Already-live child — just bump its RC.
                let _ = self.rc.inc(target);
            }
        });
        if self.new_incs_count as usize >= Self::CAPACITY {
            self.flush();
        }
    }

    /// The minimal in-place-promotion inc: increment, and promote on the 0 → 1 transition.
    fn process_inc(&mut self, o: ObjectReference) {
        debug_rc_validate("process_inc", o);
        // Validate `o` is a real heap object in some MMTk space BEFORE indexing its per-object
        // RC_TABLE metadata. `inc` -> RC_TABLE.fetch_update_atomic(o.to_raw_address()) does NO
        // mapped-address check, so a garbage ObjectReference faults on the metadata atomic. Under
        // the multidomain spawn/terminate root-scan window a freed/reused global-root slot can load
        // as an even garbage value that passes FieldSlot::load's immediate-only filter (the GH#15
        // variant-1 residual). The tracing plans survive the same garbage because trace_object's
        // in_space dispatch falls through for a non-in-space ref; RC has no such guard, so add it.
        // is_in_mmtk_spaces is a cheap SFT lookup that does NOT dereference `o` (safe on garbage).
        if !crate::memory_manager::is_in_mmtk_spaces(o) {
            return;
        }
        // Immix membership gates the per-block / straddle-line metadata in `promote` (LOS /
        // immortal / non-moving objects index unmapped immix-block side metadata otherwise).
        let in_immix = self.lxr.immix_space.in_space(o);
        if self.inc(o) {
            self.promote(o, in_immix);
        }
    }

    /// Load the (possibly-null) target of slot `s`, unlogging the slot first for mature edges,
    /// then increment + maybe-promote it.
    fn process_slot(&mut self, s: VM::VMSlot) {
        if KIND == EDGE_KIND_MATURE {
            // Guard the slot-unlog against non-MMTk (fiber-stack) slot addresses: a Cont_tag's
            // stack slots reach here as mature edges; their UNLOG side-metadata page is unmapped, so
            // unlogging them faults (the multidomain chameneos-under-LXR SIGSEGV — rr-confirmed at
            // rc.rs:280). Same is_in_mmtk_spaces guard as scan_nursery_object / the keep-alive scan.
            let sa = s.to_address();
            if sa.is_mapped()
                && crate::memory_manager::is_in_mmtk_spaces(unsafe {
                    ObjectReference::from_raw_address_unchecked(sa)
                })
            {
                sa.unlog_field_relaxed::<VM>();
            }
        }
        let Some(o) = s.load() else {
            return;
        };
        // Filter a garbage root value here too — BEFORE pushing to root_targets — so a freed/reused
        // root slot (the multidomain GH#15 residual) cannot poison curr_roots/prev_roots and fault
        // next GC's ProcessDecs (which reads rc.count(o) and would index garbage metadata). Same
        // cheap, no-deref SFT check as process_inc; keeps the root inc/dec sets garbage-free.
        if !crate::memory_manager::is_in_mmtk_spaces(o) {
            return;
        }
        // Root targets get an extra "root" reference count this GC; remember them so next GC's
        // `process_prev_roots` decrements them (otherwise root-reachable objects never die).
        if KIND == EDGE_KIND_ROOT {
            self.root_targets.push(o);
        }
        self.process_inc(o);
        // In-place cut: the object never moves, so the slot is never written back.
    }

    fn process_incs(&mut self, incs: &[VM::VMSlot]) {
        for s in incs {
            self.process_slot(*s);
        }
    }

    #[cold]
    fn flush(&mut self) {
        if !self.new_incs.is_empty() {
            let new_incs = self.new_incs.take();
            let w = ProcessIncs::<VM, EDGE_KIND_NURSERY>::new(new_incs, self.lxr);
            self.worker().add_work(WorkBucketStage::Unconstrained, w);
        }
        self.new_incs_count = 0;
    }
}

impl<VM: VMBinding, const KIND: EdgeKind> GCWork<VM> for ProcessIncs<VM, KIND> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        self.lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        self.worker = worker as *mut GCWorker<VM>;
        // Process the main buffer.
        let incs = std::mem::take(&mut self.incs);
        self.process_incs(&incs);
        // Stash this packet's root targets for next GC's decrement (root edges only).
        if KIND == EDGE_KIND_ROOT && !self.root_targets.is_empty() {
            let roots = std::mem::take(&mut self.root_targets);
            self.lxr.curr_roots.read().unwrap().push(roots);
        }
        // Drain the recursively-generated buffer.
        let mut buf = vec![];
        while !self.new_incs.is_empty() {
            self.new_incs_count = 0;
            buf.clear();
            self.new_incs.swap(&mut buf);
            self.process_incs(&buf);
        }
    }
}

// ───────────────────────────────────── ProcessDecs ──────────────────────────────────────────────

/// Process a buffer of reference-count decrements. A 1 → 0 transition makes the object dead, which
/// triggers a recursive decrement of its fields, straddle-line clearing, and a lazy block sweep.
pub struct ProcessDecs<VM: VMBinding> {
    decs: Option<Vec<ObjectReference>>,
    decs_arc: Option<Arc<Vec<ObjectReference>>>,
    /// Recursively-generated new decrements (fields of dead objects).
    new_decs: VectorQueue<ObjectReference>,
    counter: LazySweepingJobsCounter,
    rc: RefCountHelper<VM>,
    /// The worker running this packet (captured at `do_work` start; null until then).
    worker: *mut GCWorker<VM>,
}

unsafe impl<VM: VMBinding> Send for ProcessDecs<VM> {}

#[allow(dead_code)]
impl<VM: VMBinding> ProcessDecs<VM> {
    pub const CAPACITY: usize = crate::args::BUFFER_SIZE;

    fn worker(&self) -> &'static mut GCWorker<VM> {
        unsafe { &mut *self.worker }
    }

    pub fn new(decs: Vec<ObjectReference>, counter: LazySweepingJobsCounter) -> Self {
        Self {
            decs: Some(decs),
            decs_arc: None,
            new_decs: VectorQueue::default(),
            counter,
            rc: RefCountHelper::NEW,
            worker: std::ptr::null_mut(),
        }
    }

    pub fn new_arc(decs: Arc<Vec<ObjectReference>>, counter: LazySweepingJobsCounter) -> Self {
        Self {
            decs: None,
            decs_arc: Some(decs),
            new_decs: VectorQueue::default(),
            counter,
            rc: RefCountHelper::NEW,
            worker: std::ptr::null_mut(),
        }
    }

    fn recursive_dec(&mut self, o: ObjectReference) {
        self.new_decs.push(o);
        if self.new_decs.is_full() {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if !self.new_decs.is_empty() {
            let mmtk = self.worker().mmtk;
            let new_decs = self.new_decs.take();
            let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
            let w = ProcessDecs::new(new_decs, self.counter.clone_with_decs());
            if lxr.current_pause().is_none() {
                self.worker()
                    .add_work_prioritized(WorkBucketStage::Unconstrained, w);
            } else {
                self.worker().add_work(WorkBucketStage::Unconstrained, w);
            }
        }
    }

    /// An object's RC reached 0. Recursively decrement its fields, clear its straddle-line
    /// metadata, and hand its block to the lazy mature sweep.
    #[cold]
    fn process_dead_object(&mut self, o: ObjectReference, lxr: &LXR<VM>) {
        let in_ix_space = lxr.immix_space.in_space(o);
        // Recursively decrement the dead object's pointer fields. Symmetric with the inc side
        // (`scan_nursery_object` increments EVERY traced child, immix or not, since RC_TABLE is
        // global), so we decrement every traced child here too — otherwise non-immix children
        // (LOS / immortal / non-moving) would be inc'd but never dec'd (RC leak / never reclaimed).
        // The per-block bookkeeping in the recursively-scheduled `process_dead_object` is itself
        // gated on immix membership, so decrementing a non-immix child is safe.
        if !cfg!(feature = "lxr_no_recursive_dec") {
            SlotIterator::<VM>::iterate_fields(o, fake_tls(), |slot| {
                if let Some(x) = slot.load() {
                    let rc = self.rc.count(x);
                    if rc != MAX_REF_COUNT && rc != 0 {
                        self.recursive_dec(x);
                    }
                }
            });
        }
        if !crate::args::BLOCK_ONLY && in_ix_space {
            self.rc.unmark_straddle_object(o);
        }
        #[cfg(feature = "sanity")]
        unsafe {
            o.to_raw_address().store(0xdeadusize)
        };
        if in_ix_space {
            let block = Block::containing(o);
            lxr.immix_space
                .add_to_possibly_dead_mature_blocks(block, false);
        }
        // LOS path (`!in_ix_space`): the RC-aware LOS free (`los().rc_free`) is deferred; the
        // standard LOS sweep reclaims the object.
    }

    fn process_decs(&mut self, decs: &[ObjectReference], lxr: &LXR<VM>) {
        for o in decs {
            let o = *o;
            rc_stat_inc(&RC_DECS_TOTAL);
            // ATOMIC decrement (item #1 fix). `RefCountHelper::dec` is a single CAS that goes
            // x -> x-1 (and refuses on 0 / sticky-MAX). It returns `Ok(old)`. So `Ok(1)` means THIS
            // caller is the unique thread that drove the count 1 -> 0 — the winner runs
            // `process_dead_object` exactly once. A previous read-then-set(0) was a double-free hazard
            // under PARALLEL GC workers: two could both read count==1 and both kill the object.
            // (The object's fields are still readable here — its memory is not freed until the block
            // sweep — so reading them after the atomic kill is safe.)
            match self.rc.dec(o) {
                Ok(1) => {
                    // We won the 1 -> 0 kill.
                    rc_stat_inc(&RC_DECS_TO_ZERO);
                    self.process_dead_object(o, lxr);
                }
                _ => {
                    // Either decremented to a still-positive count, or refused (already 0 / sticky).
                    // Nothing more to do.
                }
            }
        }
    }
}

impl<VM: VMBinding> GCWork<VM> for ProcessDecs<VM> {
    fn do_work(&mut self, worker: &mut GCWorker<VM>, mmtk: &'static MMTK<VM>) {
        if cfg!(feature = "lxr_no_decs") {
            return;
        }
        self.worker = worker as *mut GCWorker<VM>;
        let lxr = mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap();
        if let Some(decs) = std::mem::take(&mut self.decs) {
            self.process_decs(&decs, lxr);
        } else if let Some(decs) = std::mem::take(&mut self.decs_arc) {
            self.process_decs(&decs, lxr);
        }
        let mut decs = vec![];
        while !self.new_decs.is_empty() {
            decs.clear();
            self.new_decs.swap(&mut decs);
            self.process_decs(&decs, lxr);
        }
        self.flush();
    }
}

// ─────────────────────────────── RCImmixCollectRootEdges ─────────────────────────────────────────

/// Converts a root-edge buffer into a root-increment packet. This is a `ProcessEdgesWork` purely so
/// the existing root-scanning machinery (which produces `ProcessEdgesWork` packets) can drive the RC
/// roots: its `process_slots` turns the root slots into a `ProcessIncs<_, EDGE_KIND_ROOT>` and runs
/// it inline. `trace_object`/`create_scan_work` are unreachable — it never traces. (Wired into the
/// RC root scan when `rc_enabled` is flipped — main loop.)
#[allow(dead_code)]
pub struct RCImmixCollectRootEdges<VM: VMBinding> {
    base: ProcessEdgesBase<VM>,
}

impl<VM: VMBinding> ProcessEdgesWork for RCImmixCollectRootEdges<VM> {
    type VM = VM;
    type ScanObjectsWorkType = ScanObjects<Self>;
    const OVERWRITE_REFERENCE: bool = false;
    const SCAN_OBJECTS_IMMEDIATELY: bool = true;
    const RC_ROOTS: bool = true;

    fn new(
        slots: Vec<SlotOf<Self>>,
        roots: bool,
        mmtk: &'static MMTK<VM>,
        bucket: WorkBucketStage,
    ) -> Self {
        debug_assert!(roots);
        let base = ProcessEdgesBase::new(slots, roots, mmtk, bucket);
        Self { base }
    }

    fn trace_object(&mut self, _object: ObjectReference) -> ObjectReference {
        unreachable!()
    }

    fn process_slots(&mut self) {
        if !self.slots.is_empty() {
            let lxr = self.mmtk().get_plan().downcast_ref::<LXR<VM>>().unwrap();
            // FULL (backup-trace) pause: ALSO drive a transitive MARK closure from these same root
            // slots (in the `Closure` bucket). The RC inc below preserves the root-dec balance; the
            // mark closure sets the mark bit on the reachable graph so the dead-cycle sweep can tell
            // reachable cyclic objects (marked) from dead cyclic garbage (rc>0 but unmarked).
            if lxr.current_pause() == Some(crate::plan::lxr::Pause::Full) {
                let mark_roots = self.slots.clone();
                crate::memory_manager::add_work_packet(
                    self.mmtk(),
                    WorkBucketStage::Closure,
                    crate::scheduler::gc_work::PlanProcessEdges::<
                        VM,
                        LXR<VM>,
                        { crate::policy::immix::TRACE_KIND_FAST },
                    >::new(
                        mark_roots, true, self.mmtk(), WorkBucketStage::Closure
                    ),
                );
            }
            let roots = std::mem::take(&mut self.slots);
            let mut w = ProcessIncs::<_, EDGE_KIND_ROOT>::new(roots, lxr);
            GCWork::do_work(&mut w, self.worker(), self.mmtk());
        }
    }

    fn create_scan_work(&self, _nodes: Vec<ObjectReference>) -> Self::ScanObjectsWorkType {
        unimplemented!()
    }
}

impl<VM: VMBinding> Deref for RCImmixCollectRootEdges<VM> {
    type Target = ProcessEdgesBase<VM>;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<VM: VMBinding> DerefMut for RCImmixCollectRootEdges<VM> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

/// Synchronously -- OUTSIDE any collection, with NO GC-worker/closure context -- durably keep
/// `root` and its transitive pointer-reachable children alive under the LXR reference-counting
/// plan. Used by the OCaml domain-termination path (issue #31): the just-built `Finished(Ok
/// result)` chain (three heap blocks) starts at RC 0 -- nothing has incremented it -- and under
/// LXR it is NOT saved by the tracing-plan machinery in `sync_and_terminate`:
///   * the `caml_mmtk_is_young`-retry loop is a dead no-op (LXR is non-generational, so
///     `mmtk_ocaml_is_in_nursery` always returns false), and
///   * `caml_mmtk_collect()`'s global-root scan of `ml_values->result` coalesces onto a peer GC
///     that already ran its root scan, so it never promotes the result, and
///   * the `term_sync.state` field-barrier increment is buffered but the terminating mutator's
///     barrier buffer is dropped un-flushed at deregister.
///
/// So the result's clean nursery block (`BlockState::Unallocated`, all-RC-zero) is reclaimed by
/// `sweep_nursery_blocks` and reused before the joiner dereferences `term_sync.state` -- SIGSEGV
/// in `Domain.join` (rr-confirmed).
///
/// This gives `root` and every transitive child RC >= 1 (so `Block::rc_dead()` is false and
/// `sweep_nursery_blocks`'s `state==Unallocated && rc_dead()` predicate spares each block), and
/// flips each fresh nursery block to `Unmarked` (`set_as_in_place_promoted`) for correct mature
/// bookkeeping. It mirrors `ProcessIncs::process_inc` / `promote` / `scan_nursery_object` but is
/// driven by an explicit worklist instead of scheduled `ProcessIncs` packets, so it needs no
/// worker and no open work bucket -- exactly like `LXRFieldBarrierSemantics::drain_terminating`
/// calls `rc.inc` directly. Unlike that bare single inc, it recurses so the WHOLE `Finished(Ok
/// v)` chain survives (a lone `rc.inc(root)` leaves the inner `Ok` and payload blocks at RC 0,
/// which get swept -- the reason a plain terminating-drain did not fix the crash).
///
/// This is an intentional root increment with NO matching decrement while the result is published
/// only through the terminating domain: the result stays reachably alive via `term_sync.state`
/// once the joiner links it, and is reclaimed normally when `term_sync` itself dies (its field
/// dec cascade decrements the chain). The un-paired inc is a bounded over-retention (one result
/// chain per terminated domain), never an unbounded leak and never a crash.
pub(crate) fn lxr_keep_alive_recursive<VM: VMBinding>(lxr: &LXR<VM>, root: ObjectReference) {
    use crate::policy::immix::block::BlockState;
    let rc = RefCountHelper::<VM>::NEW;
    let mut worklist: Vec<ObjectReference> = vec![root];
    while let Some(o) = worklist.pop() {
        // Cheap, no-deref SFT guard (same as `process_inc`): never index RC_TABLE for a
        // non-in-space / garbage reference.
        if !crate::memory_manager::is_in_mmtk_spaces(o) {
            continue;
        }
        // `inc` returns `Ok(0)` exactly on the 0 -> 1 (fresh-promote) transition. If the object
        // was already live (RC >= 1) its block is already spared and its subgraph already carries
        // counts -- do not re-scan (this also terminates on shared children and cycles).
        if rc.inc(o) != Ok(0) {
            continue;
        }
        // Fresh promote: block-state flip + straddle-line metadata, but ONLY for immix-space
        // objects (LOS / immortal / non-moving objects index unmapped immix-block side metadata
        // otherwise). RC_TABLE is whole-heap-mapped, so the inc above is always safe.
        let in_immix = lxr.immix_space.in_space(o);
        if in_immix {
            let block = Block::containing(o);
            if block.get_state() == BlockState::Unallocated {
                block.set_as_in_place_promoted(&lxr.immix_space);
            }
            rc.promote_with_size(o, VM::VMObjectModel::get_current_size(o));
        } else {
            o.to_raw_address().unlog_field_relaxed::<VM>();
        }
        // Enqueue every pointer child for a recursive keep-alive; unlog each field (it now
        // belongs to a mature object, so the field barrier will not re-log it).
        SlotIterator::<VM>::iterate_fields(o, fake_tls(), |slot| {
            // Skip fiber-stack slots (Cont_tag): outside MMTk spaces, so their UNLOG side-metadata
            // is unmapped and unlogging faults — see scan_nursery_object above (chameneos SIGSEGV).
            let sa = slot.to_address();
            if sa.is_mapped()
                && crate::memory_manager::is_in_mmtk_spaces(unsafe {
                    ObjectReference::from_raw_address_unchecked(sa)
                })
            {
                sa.unlog_field_relaxed::<VM>();
            }
            if let Some(target) = slot.load() {
                worklist.push(target);
            }
        });
    }
}
