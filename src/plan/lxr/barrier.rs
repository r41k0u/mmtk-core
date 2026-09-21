//! LXR field-logging write barrier (P3, additive — NOT yet installed).
//!
//! Vendored and adapted from the LXR research fork's `plan/lxr/barrier.rs`. `LXRFieldBarrierSemantics`
//! is the slow-path of the reference-counting field barrier: on the first write to a field it logs
//! the field (so duplicate writes are ignored), buffers a **decrement** of the old referent and an
//! **increment** of the slot, and on `flush` enqueues `ProcessDecs` / `ProcessIncs` packets.
//!
//! It is **present + compiling but INERT and NOT installed**. The LXR mutator (`mutator.rs`) does
//! not wire this barrier — that is part of the `rc_enabled` flip (the main loop, after review +
//! `sanity` setup). With `rc_enabled = false` the constraints select `BarrierSelector::NoBarrier`,
//! so this type is never constructed.
//!
//! ## Base-API adaptations (vs the LXR fork)
//!
//! * The base `BarrierSemantics::object_reference_write_slow` takes `src: ObjectReference` (not
//!   `Option<ObjectReference>`); signature matched here.
//! * Dropped: concurrent-marking (`ProcessModBufSATB`, the `should_create_satb_packets` /
//!   `flush_weak_refs` SATB paths), the takerate/precise-incs instrumentation, and the
//!   `lxr_precise_incs_counter` stat field — all deferred. The minimal barrier only does the
//!   inc/dec buffering + flush.
//! * `slot.to_address()` is the `Slot::to_address` method added to the base `Slot` trait.
//! * Decrements go straight to `WorkBucketStage::STWRCDecsAndSweep` (we don't take the
//!   `LAZY_DECREMENTS` postpone path — lazy decrements are deferred).

use std::sync::atomic::AtomicUsize;

use atomic::Ordering;

use super::rc::ProcessDecs;
use super::rc::ProcessIncs;
use super::rc::EDGE_KIND_MATURE;
use super::LXR;
use crate::plan::barriers::BarrierSemantics;
use crate::plan::barriers::LOGGED_VALUE;
use crate::plan::barriers::UNLOGGED_VALUE;
use crate::plan::VectorQueue;
use crate::scheduler::WorkBucketStage;
use crate::util::metadata::side_metadata::SideMetadataSpec;
use crate::util::*;
use crate::vm::slot::MemorySlice;
use crate::vm::slot::Slot;
use crate::vm::*;
use crate::LazySweepingJobsCounter;
use crate::MMTK;

// Barrier-takerate instrumentation (counted only under `TAKERATE_MEASUREMENT`, off by default).
// Inert / unused until the barrier is installed and the measurement feature is on.
#[allow(dead_code)]
pub const TAKERATE_MEASUREMENT: bool = crate::args::TAKERATE_MEASUREMENT;
#[allow(dead_code)]
pub static FAST_COUNT: AtomicUsize = AtomicUsize::new(0);
#[allow(dead_code)]
pub static SLOW_COUNT: AtomicUsize = AtomicUsize::new(0);

pub struct LXRFieldBarrierSemantics<VM: VMBinding> {
    mmtk: &'static MMTK<VM>,
    incs: VectorQueue<VM::VMSlot>,
    decs: VectorQueue<ObjectReference>,
    lxr: &'static LXR<VM>,
}

#[allow(dead_code)]
impl<VM: VMBinding> LXRFieldBarrierSemantics<VM> {
    const UNLOG_BITS: SideMetadataSpec = *VM::VMObjectModel::GLOBAL_FIELD_UNLOG_BIT_SPEC
        .as_spec()
        .extract_side_spec();

    pub fn new(mmtk: &'static MMTK<VM>) -> Self {
        Self {
            mmtk,
            incs: VectorQueue::default(),
            decs: VectorQueue::default(),
            lxr: mmtk.get_plan().downcast_ref::<LXR<VM>>().unwrap(),
        }
    }

    fn get_slot_logging_state(&self, slot: VM::VMSlot) -> u8 {
        unsafe { Self::UNLOG_BITS.load(slot.to_address()) }
    }

    /// CAS the slot's unlog bit UNLOGGED → LOGGED. Returns true iff this call did the logging (so the
    /// caller performs the once-per-field inc/dec buffering).
    fn attempt_to_log_field(&self, slot: VM::VMSlot) -> bool {
        loop {
            if self.get_slot_logging_state(slot) == LOGGED_VALUE {
                return false;
            }
            match Self::UNLOG_BITS.compare_exchange_atomic(
                slot.to_address(),
                UNLOGGED_VALUE,
                LOGGED_VALUE,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(current) => {
                    if current == LOGGED_VALUE {
                        return false;
                    }
                }
            }
            std::hint::spin_loop();
        }
    }

    /// Log the slot and return its old target, or `Err(())` if it was already logged.
    fn log_slot_and_get_old_target(&self, slot: VM::VMSlot) -> Result<Option<ObjectReference>, ()> {
        if self.get_slot_logging_state(slot) == LOGGED_VALUE {
            return Err(());
        }
        let old = slot.load();
        if self.attempt_to_log_field(slot) {
            Ok(old)
        } else {
            Err(())
        }
    }

    /// The slow path: buffer a decrement of the old referent and an increment of the slot.
    fn slow(
        &mut self,
        _src: Option<ObjectReference>,
        slot: VM::VMSlot,
        old: Option<ObjectReference>,
    ) {
        if let Some(old) = old {
            self.decs.push(old);
            if self.decs.is_full() {
                self.flush_decs();
            }
        }
        self.incs.push(slot);
        if self.incs.is_full() {
            self.flush_incs();
        }
    }

    fn enqueue_node(
        &mut self,
        src: Option<ObjectReference>,
        slot: VM::VMSlot,
        _new: Option<ObjectReference>,
    ) -> bool {
        if let Ok(old) = self.log_slot_and_get_old_target(slot) {
            self.slow(src, slot, old);
            true
        } else {
            false
        }
    }

    #[cold]
    fn flush_incs(&mut self) {
        if !self.incs.is_empty() {
            let incs = self.incs.take();
            self.lxr.rc.increase_inc_buffer_size(incs.len());
            self.mmtk.scheduler.work_buckets[WorkBucketStage::RCProcessIncs].add(ProcessIncs::<
                _,
                EDGE_KIND_MATURE,
            >::new(
                incs, self.lxr
            ));
        }
    }

    #[cold]
    fn flush_decs(&mut self) {
        if !self.decs.is_empty() {
            let decs = self.decs.take();
            let w = ProcessDecs::new(decs, LazySweepingJobsCounter::new_decs());
            self.mmtk.scheduler.work_buckets[WorkBucketStage::STWRCDecsAndSweep].add(w);
        }
    }

    /// SYNCHRONOUS drain for a TERMINATING domain (`Domain.join`), called OUTSIDE any collection /
    /// GC-worker context. The normal `flush` enqueues `ProcessIncs`/`ProcessDecs` work packets — but
    /// those need a running collection (open buckets, a worker, a sweep to drain them); enqueuing
    /// them from the terminate path (no GC active) corrupts scheduler state and the packets carry raw
    /// inc/dec slots into a domain that's about to be torn down → a GC worker later faults on freed
    /// memory. So instead we apply the buffered work DIRECTLY here:
    ///
    /// - INCREMENTS (the load-bearing ones): for each buffered slot, load its current target and do
    ///   a direct `rc.inc` into the global, whole-heap-mapped RC_TABLE. No GC context, no promotion,
    ///   no tracing needed — just bump the count so the referent is NOT prematurely freed (a lost inc
    ///   = refcount too low = premature free = the joining domain dereferences freed memory =
    ///   `EXC_BAD_ACCESS` in `Domain.join`). The dying domain's initialising/mutating writes that the
    ///   normal per-GC `mutator.flush()` would have inc'd are thus captured.
    /// - DECREMENTS: DISCARDED. A dropped dec leaves a refcount too HIGH = a small, bounded LEAK,
    ///   never a crash; running `process_dead_object`/freeing needs sweep context we don't have here.
    ///   (`rc.dec` alone — without the kill/sweep — would also be sound, but discarding is strictly
    ///   safer: no chance of driving a count to 0 and stranding a dead object whose block never gets
    ///   swept.)
    ///
    /// Bounded over-retention: a bare `rc.inc` that promotes a NURSERY object (0→1) here does NOT
    /// flip its block to mature / scan its fields (that needs the worker/closure context), so such an
    /// object can linger as a low (rc>0) entry. This is bounded by the count of un-flushed mutations
    /// at the dying domain (tiny for the functional/immutable workloads we target — binarytrees has
    /// almost no field writes) and is a retention/leak, NEVER a crash. If it ever matters, promote
    /// inline here mirroring `ProcessIncs::promote` (sans the recursive field scan).
    fn drain_terminating(&mut self) {
        let incs = self.incs.take();
        for slot in incs {
            if let Some(o) = slot.load() {
                let _ = self.lxr.rc.inc(o);
            }
        }
        // Discard the buffered decrements (bounded over-retention; never a crash).
        let _dropped = self.decs.take();
    }
}

impl<VM: VMBinding> BarrierSemantics for LXRFieldBarrierSemantics<VM> {
    type VM = VM;

    #[cold]
    fn flush(&mut self) {
        self.flush_incs();
        self.flush_decs();
    }

    /// Terminating-domain drain (no GC context): apply incs directly to RC_TABLE, discard decs.
    /// Overrides the default (which would call `flush` → enqueue work packets that can't run here).
    #[cold]
    fn flush_terminating(&mut self) {
        self.drain_terminating();
    }

    fn object_reference_write_slow(
        &mut self,
        src: ObjectReference,
        slot: VM::VMSlot,
        target: Option<ObjectReference>,
    ) {
        self.enqueue_node(Some(src), slot, target);
    }

    fn memory_region_copy_slow(&mut self, _src: VM::VMMemorySlice, dst: VM::VMMemorySlice) {
        for s in dst.iter_slots() {
            self.enqueue_node(None, s, None);
        }
    }
}
