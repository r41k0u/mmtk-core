//! UP-trace mode: single-tracer plain-operation fast paths.
//!
//! When EXACTLY ONE GC thread traces objects in a fully stopped world with no
//! concurrent marking, every synchronization primitive on the trace hot path
//! is redundant: there is no second thread to race, exclude, or publish to
//! before the pause-ending barrier. The OCaml binding measured 5.7 locked
//! RMW operations per copied object (forwarding claim CAS + SeqCst stores +
//! mark/unlog metadata ops), ~170-230 cycles/object of the 415-vs-80
//! cycles/object gap against stock OCaml's domain-private minor collector
//! (SHAPE.md 2026-08-10).
//!
//! The flag is per MMTk instance (`MMTK::set_up_trace`), set by the VM binding
//! at stop-the-world begin when it can prove the conditions (worker count == 1,
//! mutators quiesced, no concurrent marking window in flight) and cleared
//! before any mutator resumes; the pause-ending lock/unlock provides the
//! publication barrier for the plain writes. Off by default; all paths keep
//! their atomic behaviour unless the binding opts in per pause. LXR/RC paths
//! never consult this flag.
//!
//! Hot paths read a thread-local mirror (`up()`), refreshed by each GC worker
//! before every work packet, rather than a process-global: the soundness
//! argument is per tracer, so the state is per tracer. A process-wide bit
//! would leak one instance's single-worker mode into the workers of another
//! instance collecting concurrently (MMTk permits independent instances),
//! switching their metadata operations to the non-atomic paths.

use std::cell::Cell;

use crate::mmtk::MMTK;
use crate::vm::VMBinding;

thread_local! {
    /// This thread's view of the instance flag. Only GC worker threads ever
    /// set it (mirrored from `MMTK::up_trace` before each work packet, and by
    /// `MMTK::set_up_trace` on the calling thread), so mutator threads and the
    /// workers of any other MMTk instance always read `false` and keep the
    /// atomic paths.
    static UP_LOCAL: Cell<bool> = const { Cell::new(false) };
}

/// Is single-tracer mode active for the current thread's pause?
#[inline(always)]
pub fn up() -> bool {
    UP_LOCAL.with(|c| c.get())
}

/// Mirror the instance's flag into this worker thread's view. Called by
/// `GCWorker::run` before every work packet; one relaxed load.
#[inline]
pub(crate) fn sync_worker<VM: VMBinding>(mmtk: &MMTK<VM>) {
    UP_LOCAL.with(|c| c.set(mmtk.up_trace_enabled()));
    #[cfg(target_os = "linux")]
    if SOLO_PINNED.with(|c| c.get())
        && mmtk.up_solo_ordinal() != crate::scheduler::current_worker_ordinal()
    {
        solo_unpin();
    }
}

#[cfg(target_os = "linux")]
thread_local! {
    /// Solo mode: this worker's affinity mask before `solo_pin`, if pinned.
    static SOLO_SAVED: std::cell::RefCell<Option<libc::cpu_set_t>> = const { std::cell::RefCell::new(None) };
    static SOLO_PINNED: Cell<bool> = const { Cell::new(false) };
}

/// Pin the calling thread to `cpu`, saving its current mask the first time.
/// Linux only; a no-op elsewhere.
#[cfg(target_os = "linux")]
pub(crate) fn solo_pin(cpu: usize) {
    unsafe {
        SOLO_SAVED.with(|s| {
            let mut s = s.borrow_mut();
            if s.is_none() {
                let mut old: libc::cpu_set_t = std::mem::zeroed();
                if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut old) != 0
                {
                    return;
                }
                *s = Some(old);
            }
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0 {
                SOLO_PINNED.with(|c| c.set(true));
            }
        });
    }
}

/// Restore the mask saved by `solo_pin`.
#[cfg(target_os = "linux")]
fn solo_unpin() {
    SOLO_SAVED.with(|s| {
        if let Some(old) = s.borrow_mut().take() {
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &old) };
        }
    });
    SOLO_PINNED.with(|c| c.set(false));
}

/// Thread affinity is only implemented for Linux.
#[cfg(not(target_os = "linux"))]
pub(crate) fn solo_pin(_cpu: usize) {}

/// Set the calling thread's own view (see `MMTK::set_up_trace`).
pub(crate) fn set_local(enabled: bool) {
    UP_LOCAL.with(|c| c.set(enabled));
}
