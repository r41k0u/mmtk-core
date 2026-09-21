//! Bactrian's combined write barrier: generational object/region remembering
//! (post-write) + slot-granularity SATB deletion barrier (pre-write).
//!
//! The generational half is byte-identical in behaviour to GenImmix's
//! `ObjectBarrier<GenObjectBarrierSemantics>`: it owns the per-object unlog bit
//! and feeds `ProcessModBuf`/`ProcessRegionModBuf` packets into the Closure
//! bucket for the next nursery-collecting pause.
//!
//! The SATB half mirrors stock OCaml's deletion barrier (`caml_darken(old)` in
//! `caml_modify`): it is slot-granular, has no dedup bit, is active only while
//! concurrent marking is in progress, and ignores young referents (young objects
//! are post-snapshot; see the plan's module docs). Old values are buffered and
//! flushed as `ProcessModBufSATB` packets into the Concurrent bucket.

use super::gc_work::BactrianNurseryProcessEdges;
use super::global::Bactrian;
use crate::plan::barriers::Barrier;
use crate::plan::concurrent::concurrent_marking_work::ProcessModBufSATB;
use crate::plan::concurrent::global::ConcurrentPlan;
use crate::plan::concurrent::Pause;
use crate::plan::generational::global::GenerationalPlan;
use crate::plan::VectorQueue;
use crate::policy::immix::TRACE_KIND_FAST;
use crate::scheduler::WorkBucketStage;
use crate::util::constants::BYTES_IN_INT;
use crate::util::ObjectReference;
use crate::util::VMMutatorThread;
use crate::vm::slot::{MemorySlice, Slot};
use crate::vm::ObjectModel;
use crate::vm::VMBinding;
use crate::MMTK;
use atomic::Ordering;

pub struct BactrianBarrier<VM: VMBinding> {
    mmtk: &'static MMTK<VM>,
    plan: &'static Bactrian<VM>,
    tls: VMMutatorThread,
    /// Fast-path gate for the SATB half, toggled at InitialMark (on) and
    /// FinalMark/Full (off) in the mutator prepare/release hooks.
    satb_enabled: bool,
    /// Generational: objects in mature space(s) that may point to the nursery.
    modbuf: VectorQueue<ObjectReference>,
    /// Generational: mature slot regions that may point to the nursery.
    region_modbuf: VectorQueue<VM::VMMemorySlice>,
    /// SATB: overwritten old values (the snapshot's deleted edges).
    satb: VectorQueue<ObjectReference>,
    /// SATB: referents loaded from weak references during marking.
    refs: VectorQueue<ObjectReference>,
}

impl<VM: VMBinding> BactrianBarrier<VM> {
    pub fn new(mmtk: &'static MMTK<VM>, tls: VMMutatorThread) -> Self {
        Self {
            mmtk,
            plan: mmtk.get_plan().downcast_ref::<Bactrian<VM>>().unwrap(),
            tls,
            satb_enabled: false,
            modbuf: VectorQueue::new(),
            region_modbuf: VectorQueue::new(),
            satb: VectorQueue::default(),
            refs: VectorQueue::default(),
        }
    }

    pub(crate) fn set_satb_enabled(&mut self, value: bool) {
        self.satb_enabled = value;
    }

    // ---- Generational (post-write) half — mirrors GenObjectBarrierSemantics ----

    fn flush_modbuf(&mut self) {
        let buf = self.modbuf.take();
        if !buf.is_empty() {
            // The pause-aware Bactrian trace type: at InitialMark the remembered-set
            // scan must also seed the concurrent marker; at FinalMark it must remark.
            self.mmtk.scheduler.work_buckets[WorkBucketStage::Closure]
                .add(crate::plan::generational::gc_work::ProcessModBuf::<
                BactrianNurseryProcessEdges<VM>,
            >::new(buf));
        }
    }

    fn flush_region_modbuf(&mut self) {
        let buf = self.region_modbuf.take();
        if !buf.is_empty() {
            self.mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(
                crate::plan::generational::gc_work::ProcessRegionModBuf::<
                    BactrianNurseryProcessEdges<VM>,
                >::new(buf),
            );
        }
    }

    /// Per-object unlog-bit logging, as in `ObjectBarrier` (owned by the
    /// generational half).
    fn object_is_unlogged(&self, object: ObjectReference) -> bool {
        unsafe {
            VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC
                .as_spec()
                .load::<VM, u8>(object, None)
                != 0
        }
    }

    fn log_object(&self, object: ObjectReference) -> bool {
        let spec = VM::VMObjectModel::GLOBAL_LOG_BIT_SPEC.as_spec();
        loop {
            let old_value = spec.load_atomic::<VM, u8>(object, None, Ordering::SeqCst);
            if old_value == 0 {
                return false;
            }
            if spec
                .compare_exchange_metadata::<VM, u8>(
                    object,
                    1,
                    0,
                    None,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return true;
            }
        }
    }

    // ---- SATB (pre-write) half — slot-granular, marking-gated, young-filtered ----

    fn satb_enqueue(&mut self, old: ObjectReference) {
        // Young referents are post-snapshot (the nursery is emptied at InitialMark):
        // skipping them is sound, and it keeps young references out of the marking
        // queues, which must never hold them across a (moving) nursery pause.
        if self.plan.is_object_in_nursery(old) {
            crate::plan::concurrent::diag::SATB_YOUNG_DROP
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        crate::plan::concurrent::diag::SATB_ENQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.satb.push(old);
        if self.satb.is_full() {
            self.flush_satb();
        }
    }

    fn should_create_satb_packets(&self) -> bool {
        self.plan.concurrent_work_in_progress()
            || self.plan.current_pause() == Some(Pause::FinalMark)
    }

    fn dispatch_satb_packet(&self, w: ProcessModBufSATB<VM, Bactrian<VM>, TRACE_KIND_FAST>) {
        if self.plan.concurrent_work_in_progress() {
            self.plan.schedule_marking_packet(Box::new(w));
        } else {
            self.mmtk.scheduler.work_buckets[WorkBucketStage::Closure].add(w);
        }
    }

    fn flush_satb(&mut self) {
        if !self.satb.is_empty() {
            if self.should_create_satb_packets() {
                let satb = self.satb.take();
                self.dispatch_satb_packet(ProcessModBufSATB::new(satb));
            } else {
                let _ = self.satb.take();
            }
        }
    }

    #[cold]
    fn flush_weak_refs(&mut self) {
        if !self.refs.is_empty() {
            if self.should_create_satb_packets() {
                let refs = self.refs.take();
                self.dispatch_satb_packet(ProcessModBufSATB::new(refs));
            } else {
                let _ = self.refs.take();
            }
        }
    }
}

impl<VM: VMBinding> Barrier<VM> for BactrianBarrier<VM> {
    fn flush(&mut self) {
        self.flush_modbuf();
        self.flush_region_modbuf();
        self.flush_satb();
        self.flush_weak_refs();
    }

    fn load_weak_reference(&mut self, o: ObjectReference) {
        // See SATBBarrierSemantics::load_weak_reference: a referent loaded from a
        // weak reference during marking becomes strongly reachable but may not be in
        // the snapshot; conservatively keep it live. Young referents are
        // post-snapshot and are handled by the nursery (skip, as everywhere).
        if !self.satb_enabled || !self.plan.concurrent_work_in_progress() {
            return;
        }
        if self.plan.is_object_in_nursery(o) {
            return;
        }
        self.refs.push(o);
        if self.refs.is_full() {
            self.flush_weak_refs();
        }
    }

    // -- Object-granularity write paths --

    fn object_reference_write_pre(
        &mut self,
        _src: ObjectReference,
        slot: VM::VMSlot,
        _target: Option<ObjectReference>,
    ) {
        // SATB deletion barrier: snapshot the old value of this slot.
        if self.satb_enabled {
            if let Some(old) = slot.load() {
                self.satb_enqueue(old);
            }
        }
    }

    fn object_reference_write_post(
        &mut self,
        src: ObjectReference,
        slot: VM::VMSlot,
        target: Option<ObjectReference>,
    ) {
        // Generational object-remembering barrier.
        if self.object_is_unlogged(src) {
            self.object_reference_write_slow(src, slot, target);
        }
    }

    fn object_reference_write_slow(
        &mut self,
        src: ObjectReference,
        _slot: VM::VMSlot,
        _target: Option<ObjectReference>,
    ) {
        if self.log_object(src) {
            self.modbuf.push(src);
            self.modbuf.is_full().then(|| self.flush_modbuf());
        }
    }

    // -- Region (slot-range) write paths: the OCaml binding's entry points --

    /// SATB half. The binding calls this BEFORE the store(s), while the slots still
    /// hold the old values (`mmtk_ocaml_satb_barrier`).
    fn memory_region_copy_pre(&mut self, _src: VM::VMMemorySlice, dst: VM::VMMemorySlice) {
        if !self.satb_enabled {
            return;
        }
        for s in dst.iter_slots() {
            if let Some(old) = s.load() {
                self.satb_enqueue(old);
            }
        }
    }

    /// Generational half. The binding calls this for every modified slot region
    /// (`mmtk_ocaml_region_barrier`); mirrors GenObjectBarrierSemantics.
    fn memory_region_copy_post(&mut self, _src: VM::VMMemorySlice, dst: VM::VMMemorySlice) {
        let dst_in_nursery = match dst.object() {
            Some(obj) => self.plan.is_object_in_nursery(obj),
            None => self.plan.is_address_in_nursery(dst.start()),
        };
        // Only remember slots in mature spaces.
        if !dst_in_nursery {
            debug_assert_eq!(
                dst.bytes() & (BYTES_IN_INT - 1),
                0,
                "bytes should be a multiple of 32-bit words"
            );
            self.region_modbuf.push(dst);
            self.region_modbuf
                .is_full()
                .then(|| self.flush_region_modbuf());
        }
    }

    fn object_probable_write(&mut self, obj: ObjectReference) {
        // SATB half: snapshot all fields (object-granularity pre-write).
        if self.satb_enabled && !self.plan.is_object_in_nursery(obj) {
            crate::plan::tracing::SlotIterator::<VM>::iterate_fields(obj, self.tls.0, |s| {
                if let Some(old) = s.load() {
                    self.satb_enqueue(old);
                }
            });
        }
        // Generational half: remember the object.
        if self.object_is_unlogged(obj) && self.log_object(obj) {
            self.modbuf.push(obj);
            self.modbuf.is_full().then(|| self.flush_modbuf());
        }
    }
}
