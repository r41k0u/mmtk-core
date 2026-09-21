use crate::util::copy::*;
use crate::util::metadata::MetadataSpec;
use crate::util::{constants, ObjectReference};
use crate::vm::ObjectModel;
use crate::vm::VMBinding;
use std::sync::atomic::Ordering;

const FORWARDING_NOT_TRIGGERED_YET: u8 = 0b00;
const BEING_FORWARDED: u8 = 0b10;
const FORWARDED: u8 = 0b11;
const FORWARDING_MASK: u8 = 0b11;
#[allow(unused)]
const FORWARDING_BITS: usize = 2;

// copy address mask
#[cfg(target_pointer_width = "64")]
const FORWARDING_POINTER_MASK: usize = 0x00ff_ffff_ffff_fff8;
#[cfg(target_pointer_width = "32")]
const FORWARDING_POINTER_MASK: usize = 0xffff_fffc;

/// Is the value-range forwarding discriminator active? Requires BOTH the
/// binding's guarantee (headers below heap start, see
/// [`crate::vm::ObjectModel::HEADER_FORWARDING_SENTINEL`]) and a
/// stopped-world single-tracer pause (`up_trace::up()`): with one tracer the
/// BEING_FORWARDED claim state never exists, so "the header word is a heap
/// pointer" is a complete two-state encoding and the side FORWARDING_BITS
/// table is never touched. Multi-tracer pauses fall back to the side bits.
#[inline(always)]
fn header_sentinel_active<VM: VMBinding>() -> bool {
    VM::VMObjectModel::HEADER_FORWARDING_SENTINEL && crate::util::up_trace::up()
}

/// Under the sentinel: FORWARDED iff the header word now holds a heap pointer.
#[inline(always)]
fn sentinel_status<VM: VMBinding>(object: ObjectReference) -> u8 {
    let word = VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC.load_atomic::<VM, usize>(
        object,
        None,
        Ordering::Relaxed,
    );
    if word
        >= crate::util::heap::layout::vm_layout::vm_layout()
            .heap_start
            .as_usize()
    {
        FORWARDED
    } else {
        FORWARDING_NOT_TRIGGERED_YET
    }
}

/// Attempt to become the worker thread who will forward the object.
/// The successful worker will set the object forwarding bits to BEING_FORWARDED, preventing other workers from forwarding the same object.
pub fn attempt_to_forward<VM: VMBinding>(object: ObjectReference) -> u8 {
    // UP-trace: a single tracer cannot race itself — the claim CAS and the
    // BEING_FORWARDED intermediate state exist only to exclude other workers.
    // Return the current status; NOT_TRIGGERED sends the caller straight to
    // forward_object (which under UP writes the final state plainly).
    if crate::util::up_trace::up() {
        return get_forwarding_status::<VM>(object);
    }
    loop {
        let old_value = get_forwarding_status::<VM>(object);
        if old_value != FORWARDING_NOT_TRIGGERED_YET
            || VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC
                .compare_exchange_metadata::<VM, u8>(
                    object,
                    old_value,
                    BEING_FORWARDED,
                    None,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_ok()
        {
            return old_value;
        }
    }
}

/// Spin-wait for the object's forwarding to become complete and then read the forwarding pointer to the new object.
///
/// # Arguments:
///
/// * `object`: the forwarded/being_forwarded object.
/// * `forwarding_bits`: the last state of the forwarding bits before calling this function.
///
/// Returns a reference to the new object.
///
pub fn spin_and_get_forwarded_object<VM: VMBinding>(
    object: ObjectReference,
    forwarding_bits: u8,
) -> ObjectReference {
    let mut forwarding_bits = forwarding_bits;
    while forwarding_bits == BEING_FORWARDED {
        forwarding_bits = get_forwarding_status::<VM>(object);
    }

    if forwarding_bits == FORWARDED {
        read_forwarding_pointer::<VM>(object)
    } else {
        // For some policies (such as Immix), we can have interleaving such that one thread clears
        // the forwarding word while another thread was stuck spinning in the above loop.
        // See: https://github.com/mmtk/mmtk-core/issues/579
        debug_assert!(
            forwarding_bits == FORWARDING_NOT_TRIGGERED_YET,
            "Invalid/Corrupted forwarding word {:x} for object {}",
            forwarding_bits,
            object,
        );
        object
    }
}

/// Copy an object and set the forwarding state.
///
/// The caller can use `on_after_forwarding` to set extra metadata (including VO bits, mark bits,
/// etc.) after the object is copied, but before the forwarding state is changed to `FORWARDED`. The
/// atomic memory operation that sets the forwarding bits to `FORWARDED` has the `SeqCst` order.  It
/// will guarantee that if another GC worker thread that attempts to forward the same object sees
/// the forwarding bits being `FORWARDED`, it is guaranteed to see those extra metadata set.
///
/// Arguments:
///
/// *   `object`: The object to copy.
/// *   `semantics`: The copy semantics.
/// *   `copy_context`: A reference ot the `CopyContext` instance of the current GC worker.
/// *   `on_after_forwarding`: A callback function that is called after `object` is copied, but
///     before the forwarding bits are set.  Its argument is a reference to the new copy of
///     `object`.
pub fn forward_object<VM: VMBinding>(
    object: ObjectReference,
    semantics: CopySemantics,
    copy_context: &mut GCWorkerCopyContext<VM>,
    on_after_forwarding: impl FnOnce(ObjectReference),
) -> ObjectReference {
    let new_object = VM::VMObjectModel::copy(object, semantics, copy_context);
    on_after_forwarding(new_object);
    // UP-trace: SeqCst stores compile to locked XCHG on x86; with a single
    // tracer Relaxed (a plain MOV) is sufficient — the pause-ending barrier
    // publishes before any other thread can look.
    let ord = if crate::util::up_trace::up() {
        Ordering::Relaxed
    } else {
        Ordering::SeqCst
    };
    if let Some(shift) = forwarding_bits_offset_in_forwarding_pointer::<VM>() {
        VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC.store_atomic::<VM, usize>(
            object,
            new_object.to_raw_address().as_usize() | ((FORWARDED as usize) << shift),
            None,
            ord,
        )
    } else {
        write_forwarding_pointer::<VM>(object, new_object);
        // Sentinel mode: a clever trick to know whether the object has been
        // forwarded yet — the header-pointer store above IS the state change
        // (a word >= heap start now reads FORWARDED); no side bits to set.
        if !header_sentinel_active::<VM>() {
            VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC
                .store_atomic::<VM, u8>(object, FORWARDED, None, ord);
        }
    }
    new_object
}

/// Return the forwarding bits for a given `ObjectReference`.
pub fn get_forwarding_status<VM: VMBinding>(object: ObjectReference) -> u8 {
    if header_sentinel_active::<VM>() {
        return sentinel_status::<VM>(object);
    }
    VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC.load_atomic::<VM, u8>(
        object,
        None,
        Ordering::SeqCst,
    )
}

pub fn is_forwarded<VM: VMBinding>(object: ObjectReference) -> bool {
    get_forwarding_status::<VM>(object) == FORWARDED
}

fn is_being_forwarded<VM: VMBinding>(object: ObjectReference) -> bool {
    get_forwarding_status::<VM>(object) == BEING_FORWARDED
}

pub fn is_forwarded_or_being_forwarded<VM: VMBinding>(object: ObjectReference) -> bool {
    get_forwarding_status::<VM>(object) != FORWARDING_NOT_TRIGGERED_YET
}

pub fn state_is_forwarded_or_being_forwarded(forwarding_bits: u8) -> bool {
    forwarding_bits != FORWARDING_NOT_TRIGGERED_YET
}

pub fn state_is_being_forwarded(forwarding_bits: u8) -> bool {
    forwarding_bits == BEING_FORWARDED
}

/// Zero the forwarding bits of an object.
/// This function is used on new objects.
pub fn clear_forwarding_bits<VM: VMBinding>(object: ObjectReference) {
    // Sentinel mode never writes the side bits, and a fresh copy's real
    // header (below heap start) already reads NOT_TRIGGERED — nothing to do.
    if header_sentinel_active::<VM>() {
        return;
    }
    VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC.store_atomic::<VM, u8>(
        object,
        0,
        None,
        Ordering::SeqCst,
    )
}

/// Read the forwarding pointer of an object.
/// This function is called on forwarded/being_forwarded objects.
pub fn read_forwarding_pointer<VM: VMBinding>(object: ObjectReference) -> ObjectReference {
    debug_assert!(
        is_forwarded_or_being_forwarded::<VM>(object),
        "read_forwarding_pointer called for object {:?} that has not started forwarding!",
        object,
    );

    // We write the forwarding poiner. We know it is an object reference.
    unsafe {
        // We use "unchecked" convertion becasue we guarantee the forwarding pointer we stored
        // previously is from a valid `ObjectReference` which is never zero.
        ObjectReference::from_raw_address_unchecked(crate::util::Address::from_usize(
            VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC.load_atomic::<VM, usize>(
                object,
                Some(FORWARDING_POINTER_MASK),
                Ordering::SeqCst,
            ),
        ))
    }
}

/// Write the forwarding pointer of an object.
/// This function is called on being_forwarded objects.
pub fn write_forwarding_pointer<VM: VMBinding>(
    object: ObjectReference,
    new_object: ObjectReference,
) {
    debug_assert!(
        // Sentinel mode has no BEING_FORWARDED claim state: the single tracer
        // goes straight from unforwarded to this store.
        header_sentinel_active::<VM>() || is_being_forwarded::<VM>(object),
        "write_forwarding_pointer called for object {:?} that is not being forwarded! Forwarding state = 0x{:x}",
        object,
        get_forwarding_status::<VM>(object),
    );

    trace!("write_forwarding_pointer({}, {})", object, new_object);
    VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC.store_atomic::<VM, usize>(
        object,
        new_object.to_raw_address().as_usize(),
        Some(FORWARDING_POINTER_MASK),
        Ordering::SeqCst,
    )
}

/// (This function is only used internal to the `util` module)
///
/// This function checks whether the forwarding pointer and forwarding bits can be written in the same atomic operation.
///
/// Returns `None` if this is not possible.
/// Otherwise, returns `Some(shift)`, where `shift` is the left shift needed on forwarding bits.
///
#[cfg(target_endian = "little")]
pub(super) fn forwarding_bits_offset_in_forwarding_pointer<VM: VMBinding>() -> Option<isize> {
    use std::ops::Deref;
    // if both forwarding bits and forwarding pointer are in-header
    match (
        VM::VMObjectModel::LOCAL_FORWARDING_POINTER_SPEC.deref(),
        VM::VMObjectModel::LOCAL_FORWARDING_BITS_SPEC.deref(),
    ) {
        (MetadataSpec::InHeader(fp), MetadataSpec::InHeader(fb)) => {
            let maybe_shift = fb.bit_offset - fp.bit_offset;
            if maybe_shift >= 0 && maybe_shift < constants::BITS_IN_WORD as isize {
                Some(maybe_shift)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(target_endian = "big")]
pub(super) fn forwarding_bits_offset_in_forwarding_pointer<VM: VMBinding>() -> Option<isize> {
    unimplemented!()
}

pub(crate) fn debug_print_object_forwarding_info<VM: VMBinding>(object: ObjectReference) {
    let forwarding_bits = get_forwarding_status::<VM>(object);
    println!(
        "forwarding bits = {:?}, forwarding pointer = {:?}",
        forwarding_bits,
        if state_is_forwarded_or_being_forwarded(forwarding_bits) {
            Some(read_forwarding_pointer::<VM>(object))
        } else {
            None
        }
    )
}
