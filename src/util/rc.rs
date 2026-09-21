//! Reference-counting helpers for the LXR plan (P1 scaffolding — additive).
//!
//! This vendors (and adapts) the LXR research fork's `src/util/rc.rs`: the per-object
//! reference-count table accessors plus the immix straddle-line bookkeeping. It is **purely
//! additive** — `RefCountHelper` is only instantiated by the LXR plan/gc_work (P3), so the whole
//! module is `#[allow(dead_code)]`.
//!
//! ## Adaptation notes (LXR `lxr/lxr` @ +1690 commits  vs  our 0.32.0)
//!
//! The LXR original leans on a cluster of *LXR-added* convenience methods that do not exist on our
//! 0.32.0 `ObjectReference` / `Address` / `Line` / `SideMetadataSpec`. Each was bridged here:
//!
//! | LXR call (does not exist in 0.32.0)        | 0.32.0 replacement used here                                  |
//! |--------------------------------------------|---------------------------------------------------------------|
//! | `a.to_object_reference::<VM>()`            | `ObjectReference::from_raw_address(a).unwrap()` (local `oref`) |
//! | `Line::containing::<VM>(o)`                | `Line::from_aligned_address(Line::align(o.to_raw_address()))`  |
//! | `Line::from(addr)`                         | `Line::from_aligned_address(addr)`                            |
//! | `SideMetadataSpec::load_byte(a)`           | `RC_TABLE.load_atomic::<u8>(a, Relaxed)`                       |
//! | `SideMetadataSpec::prefetch_{read,write}`  | dropped (pure perf hint; not needed for P1 table accessors)   |
//! | `o.log_start_address::<VM>()` (`promote*`) | dropped (field-unlog logging is a P3 barrier concern)         |
//!
//! Methods that genuinely require P3 (`promote`, `promote_with_size`, `prefetch_*`,
//! `rc_table_range`) are deferred rather than half-implemented.

use std::marker::PhantomData;
use std::sync::atomic::AtomicUsize;

use crate::policy::immix::line::Line;
use crate::util::linear_scan::Region;
use crate::util::{metadata::side_metadata::SideMetadataSpec, Address, ObjectReference};
use crate::vm::*;
use atomic::Ordering;

/// log2 of the number of RC bits per object. LXR default = 1 (=> 2 bits, values 0..3).
pub const LOG_REF_COUNT_BITS: usize = {
    if cfg!(feature = "lxr_rc_bits_2") {
        1
    } else if cfg!(feature = "lxr_rc_bits_4") {
        2
    } else if cfg!(feature = "lxr_rc_bits_8") {
        3
    } else {
        1
    }
};
pub const REF_COUNT_BITS: u8 = 1 << LOG_REF_COUNT_BITS;
pub const REF_COUNT_MASK: u8 = (((1u16 << REF_COUNT_BITS) - 1) & 0xff) as u8;
/// The saturated ("sticky") count. Once an object reaches this it is never decremented;
/// it can only be reclaimed by the concurrent backup (SATB) trace.
pub const MAX_REF_COUNT: u8 = REF_COUNT_MASK;

pub const LOG_MIN_OBJECT_SIZE: usize = crate::util::constants::LOG_MIN_OBJECT_SIZE as _;
pub const MIN_OBJECT_SIZE: usize = 1 << LOG_MIN_OBJECT_SIZE;

/// Straddle-line marks (per immix line, side metadata).
pub const RC_STRADDLE_LINES: SideMetadataSpec =
    crate::util::metadata::side_metadata::spec_defs::RC_STRADDLE_LINES;

/// Per-object reference count table (global side metadata).
pub const RC_TABLE: SideMetadataSpec = crate::util::metadata::side_metadata::spec_defs::RC_TABLE;

#[allow(dead_code)]
static INC_BUFFER_SIZE: AtomicUsize = AtomicUsize::new(0);

/// Reconstruct an `ObjectReference` from a raw address. Bridges LXR's
/// `Address::to_object_reference::<VM>()` (which does not exist on our 0.32.0 `Address`).
#[inline(always)]
fn oref(a: Address) -> ObjectReference {
    ObjectReference::from_raw_address(a).unwrap()
}

#[repr(transparent)]
#[derive(Debug, Copy)]
#[allow(dead_code)]
pub struct RefCountHelper<VM: VMBinding>(PhantomData<VM>);

#[allow(dead_code)]
impl<VM: VMBinding> RefCountHelper<VM> {
    pub const NEW: Self = Self(PhantomData);

    pub fn inc_buffer_size(&self) -> usize {
        INC_BUFFER_SIZE.load(Ordering::Relaxed)
    }

    pub fn increase_inc_buffer_size(&self, delta: usize) {
        INC_BUFFER_SIZE.store(
            INC_BUFFER_SIZE
                .load(Ordering::Relaxed)
                .saturating_add(delta),
            Ordering::Relaxed,
        )
    }

    pub fn reset_inc_buffer_size(&self) {
        INC_BUFFER_SIZE.store(0, Ordering::Relaxed)
    }

    pub fn fetch_update(
        &self,
        o: ObjectReference,
        // NOTE (API delta): our 0.32.0 `SideMetadataSpec::fetch_update_atomic` bounds its closure
        // `F: FnMut(T) -> Option<T> + Copy`. LXR's signature had no `+ Copy`. We add `+ Copy` here
        // so the closure satisfies our tree's bound; all call sites below pass `Copy` closures.
        f: impl FnMut(u8) -> Option<u8> + Copy,
    ) -> Result<u8, u8> {
        RC_TABLE.fetch_update_atomic(o.to_raw_address(), Ordering::Relaxed, Ordering::Relaxed, f)
    }

    pub fn is_stuck(&self, o: ObjectReference) -> bool {
        self.count(o) == MAX_REF_COUNT
    }

    pub fn stick(&self, o: ObjectReference) -> Result<u8, u8> {
        self.fetch_update(o, |x| {
            debug_assert!(x <= MAX_REF_COUNT);
            if x == MAX_REF_COUNT {
                None
            } else {
                Some(MAX_REF_COUNT)
            }
        })
    }

    pub fn inc(&self, o: ObjectReference) -> Result<u8, u8> {
        self.fetch_update(o, |x| {
            debug_assert!(x <= MAX_REF_COUNT);
            if x == MAX_REF_COUNT {
                None
            } else {
                Some(x + 1)
            }
        })
    }

    pub fn dec(&self, o: ObjectReference) -> Result<u8, u8> {
        self.fetch_update(o, |x| {
            debug_assert!(x <= MAX_REF_COUNT);
            if x == 0 || x == MAX_REF_COUNT
            /* sticky */
            {
                None
            } else {
                Some(x - 1)
            }
        })
    }

    pub fn set(&self, o: ObjectReference, count: u8) {
        RC_TABLE.store_atomic(o.to_raw_address(), count, Ordering::Relaxed)
    }

    pub fn set_relaxed(&self, o: ObjectReference, count: u8) {
        unsafe { RC_TABLE.store(o.to_raw_address(), count) }
    }

    pub fn count(&self, o: ObjectReference) -> u8 {
        RC_TABLE.load_atomic(o.to_raw_address(), Ordering::Relaxed)
    }

    pub fn object_or_line_is_dead(&self, o: ObjectReference) -> bool {
        // LXR used `RC_TABLE.load_byte(addr)`; our 0.32.0 spec has no `load_byte`, so we read
        // the RC entry atomically as a `u8` instead (the RC entry is < 8 bits, so this reads the
        // whole byte's worth of the entry's value).
        RC_TABLE.load_atomic::<u8>(o.to_raw_address(), Ordering::Relaxed) == 0
    }

    pub fn is_dead(&self, o: ObjectReference) -> bool {
        let v: u8 = RC_TABLE.load_atomic(o.to_raw_address(), Ordering::Relaxed);
        v == 0
    }

    pub fn is_dead_or_stuck(&self, o: ObjectReference) -> bool {
        let v: u8 = RC_TABLE.load_atomic(o.to_raw_address(), Ordering::Relaxed);
        v == 0 || v == MAX_REF_COUNT
    }

    pub fn is_straddle_line(&self, line: Line) -> bool {
        let v: u8 = unsafe { RC_STRADDLE_LINES.load::<u8>(line.start()) };
        v != 0
    }

    pub fn address_is_in_straddle_line(&self, a: Address) -> bool {
        // LXR: `Line::from(Line::align(a))`. Our `Line::align` is the static `Region::align`,
        // and we build the line from the aligned address.
        let line = Line::from_aligned_address(Line::align(a));
        self.count(oref(a)) != 0 && self.is_straddle_line(line)
    }

    fn mark_straddle_object_with_size(&self, o: ObjectReference, size: usize) {
        debug_assert!(size > Line::BYTES);
        // LXR: `Line::containing::<VM>(o).next()`. We align the object's address down to a line
        // and take the following line as the first continuation line.
        let start_line = Line::from_aligned_address(Line::align(o.to_raw_address())).next();
        let end_line = Line::from_aligned_address(Line::align(o.to_raw_address() + size));
        let mut line = start_line;
        while line != end_line {
            unsafe { RC_STRADDLE_LINES.store(line.start(), 1u8) };
            self.set_relaxed(oref(line.start()), 1);
            line = line.next();
        }
    }

    pub fn mark_straddle_object(&self, o: ObjectReference) {
        let size = VM::VMObjectModel::get_current_size(o);
        self.mark_straddle_object_with_size(o, size)
    }

    pub fn unmark_straddle_object(&self, o: ObjectReference) {
        let size = VM::VMObjectModel::get_current_size(o);
        if size > Line::BYTES {
            let start_line = Line::from_aligned_address(Line::align(o.to_raw_address())).next();
            let end_line = Line::from_aligned_address(Line::align(o.to_raw_address() + size));
            let mut line = start_line;
            while line != end_line {
                self.set_relaxed(oref(line.start()), 0);
                unsafe { RC_STRADDLE_LINES.store(line.start(), 0u8) };
                line = line.next();
            }
        }
    }

    pub fn assert_zero_ref_count(&self, o: ObjectReference) {
        let size = VM::VMObjectModel::get_current_size(o);
        for i in (0..size).step_by(MIN_OBJECT_SIZE) {
            let a = o.to_raw_address() + i;
            assert_eq!(0, self.count(oref(a)));
        }
    }

    /// Promote a freshly-incremented nursery object to mature: mark the straddle-line metadata
    /// for any object spanning more than one immix line. Vendored from lxr-v0.32.0 `util/rc.rs`.
    ///
    /// Adaptation: the reference also calls `o.log_start_address::<VM>()`, but that method is a
    /// no-op `{}` in the reference (the per-object start-address log is unused under the default
    /// config), so it is dropped here. The per-field unlog logging that actually matters for the
    /// barrier is done by `scan_nursery_object` in the RC trace, not here.
    pub fn promote(&self, o: ObjectReference) {
        let size = VM::VMObjectModel::get_current_size(o);
        if size > Line::BYTES {
            self.mark_straddle_object_with_size(o, size);
        }
    }

    /// As [`Self::promote`], but the object's size is already known (saves a re-read).
    pub fn promote_with_size(&self, o: ObjectReference, size: usize) {
        if size > Line::BYTES {
            self.mark_straddle_object_with_size(o, size);
        }
    }

    // NOTE (deferred to P3+): `prefetch_read`, `prefetch_write`, and `rc_table_range` are
    // intentionally not vendored. The prefetch hints need LXR's `SideMetadataSpec::prefetch_*`
    // (not in our 0.32.0); `rc_table_range` is only used by the immix-policy RC sweep.
}

impl<VM: VMBinding> Clone for RefCountHelper<VM> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}
