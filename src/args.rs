//! LXR runtime configuration.
//!
//! This module vendors (and adapts) the LXR research fork's `src/args.rs`. It exposes the
//! compile-time and runtime knobs that LXR's reference-counting-on-Immix collector reads. It is
//! **purely additive scaffolding** (P1): nothing here is wired into any existing plan's behaviour.
//! The values are only consumed once the `MMTK_PLAN=LXR` plan + its gc_work land (P3).
//!
#![allow(missing_docs)] // vendored LXR knobs; documented upstream
//!
//! ## Adaptation notes (LXR `lxr/lxr` @ +1690 commits  vs  our 0.32.0)
//!
//! * LXR stored `RuntimeArgs` in a `static mut MaybeUninit<RuntimeArgs>` requiring an explicit
//!   `RuntimeArgs::init()` call at boot. That is UB-adjacent under the 2024 `static_mut_refs`
//!   lint and needs a boot-time hook we don't have a place for in P1. We use `lazy_static!`
//!   (already a crate dependency) so `crate::args()` is self-initialising on first use.
//! * LXR's `dump_features()` reaches into JVM-specific `Options` fields (`conc_threads`,
//!   `no_finalizer`, `no_reference_types`) and into `Block`/`Line` policy constants. Those are
//!   trimmed here to the subset that compiles against our tree; the full feature dump is a P3
//!   nicety, not load-bearing.
//! * The `lxr_*` cargo features LXR uses for compile-time A/B switches are not declared in our
//!   `Cargo.toml`, so `cfg!(feature = "lxr_*")` evaluates to `false` — i.e. we vendor LXR's
//!   *default* configuration. Declaring the features is deferred to whenever we want to sweep
//!   them (P4/P5 tuning).

use crate::BarrierSelector;
use std::env;
use std::fmt::Debug;
use std::str::FromStr;

/// Runtime (env-var-driven) LXR knobs. Mirrors LXR's `RuntimeArgs`.
#[derive(Debug)]
#[allow(dead_code)] // consumed by the LXR plan/gc_work (P3), not by any existing plan.
pub struct RuntimeArgs {
    pub incs_limit: Option<usize>,
    pub no_mutator_line_recycling: bool,
    pub no_line_recycling: bool,
    pub nursery_blocks: Option<usize>,
    pub young_limit_mb: Option<usize>,
    pub nursery_ratio: Option<usize>,
    pub max_pause_millis: Option<usize>,
    pub max_young_evac_size: usize,
    /// Terminate the CM or RC loop if the available heap after an RC pause is still small.
    pub rc_stop_percent: usize,
    pub max_survival_mb: usize,
    pub survival_predictor_harmonic_mean: bool,
    pub survival_predictor_weighted: bool,
    pub trace_threshold: usize,
    pub chunk_defarg_percent: usize,
    pub transparent_hugepage: bool,
}

impl Default for RuntimeArgs {
    fn default() -> Self {
        fn env_arg<T: FromStr + Debug>(name: &str) -> Option<T>
        where
            T::Err: Debug,
        {
            env::var(name).map(|x| T::from_str(&x).unwrap()).ok()
        }
        fn env_bool_arg(name: &str) -> Option<bool> {
            env::var(name)
                .map(|x| x == "1" || x == "true" || x == "TRUE")
                .ok()
        }
        Self {
            incs_limit: env_arg("INCS_LIMIT"),
            no_mutator_line_recycling: env_bool_arg("NO_MUTATOR_LINE_RECYCLING").unwrap_or(false),
            no_line_recycling: env_bool_arg("NO_LINE_RECYCLING").unwrap_or(false),
            nursery_blocks: env_arg("NURSERY_BLOCKS"),
            young_limit_mb: env_arg("YOUNG_LIMIT").or_else(|| env_arg("YOUNG_LIMIT_MB")),
            nursery_ratio: env_arg("NURSERY_RATIO"),
            max_pause_millis: env_arg("MAX_PAUSE_MILLIS"),
            max_young_evac_size: env_arg("MAX_YOUNG_EVAC_SIZE").unwrap_or(usize::MAX),
            rc_stop_percent: env_arg("RC_STOP_PERCENT").unwrap_or(15),
            max_survival_mb: env_arg::<usize>("MAX_SURVIVAL_MB").unwrap_or(128),
            survival_predictor_harmonic_mean: env_bool_arg("SURVIVAL_PREDICTOR_HARMONIC_MEAN")
                .unwrap_or(false),
            survival_predictor_weighted: env_bool_arg("SURVIVAL_PREDICTOR_WEIGHTED")
                .unwrap_or(false),
            trace_threshold: env_arg("TRACE_THRESHOLD2")
                .or_else(|| env_arg("TRACE_THRESHOLD"))
                .or_else(|| env_arg("CM_THRESHOLD"))
                .unwrap_or(20),
            chunk_defarg_percent: env_arg::<usize>("CHUNK_DEFARG_THRESHOLD").unwrap_or(32),
            transparent_hugepage: env_bool_arg("TRANSPARENT_HUGEPAGE")
                .or_else(|| env_bool_arg("HUGEPAGE"))
                .unwrap_or(true),
        }
    }
}

lazy_static! {
    static ref ARGS: RuntimeArgs = RuntimeArgs::default();
}

/// Access the (lazily-initialised) global LXR runtime args.
///
/// Adapts LXR's `RuntimeArgs::get()` to a `lazy_static` so callers don't need an explicit
/// boot-time `init()`.
#[allow(dead_code)]
pub fn args() -> &'static RuntimeArgs {
    &ARGS
}

// ---------- Compile-time LXR flags ---------- //
// These mirror LXR's compile-time constants. The `lxr_*` cargo features are not declared in our
// Cargo.toml yet, so every `cfg!(feature = "lxr_*")` is `false` here — i.e. we get LXR's defaults.

#[allow(dead_code)]
pub const CM_LARGE_ARRAY_OPTIMIZATION: bool = false;

#[allow(dead_code)]
pub const BUFFER_SIZE: usize = {
    if cfg!(feature = "lxr_buf_2048") {
        2048
    } else if cfg!(feature = "lxr_buf_1024") {
        1024
    } else if cfg!(feature = "lxr_buf_512") {
        512
    } else if cfg!(feature = "lxr_buf_256") {
        256
    } else {
        1024
    }
};

#[allow(dead_code)]
pub const NO_LAZY_SWEEP_WHEN_STW_CANNOT_RELEASE_ENOUGH_MEMORY: bool = false;

// ---------- Immix flags ---------- //
#[allow(dead_code)]
pub const CYCLE_TRIGGER_THRESHOLD: usize = 1024;
/// Mark lines when scanning objects. Otherwise, do it at mark time.
#[allow(dead_code)]
pub const MARK_LINE_AT_SCAN_TIME: bool = true;

// ---------- CM/RC Immix flags ---------- //
#[allow(dead_code)]
pub const EAGER_INCREMENTS: bool = false;
#[allow(dead_code)]
pub const LAZY_DECREMENTS: bool = !cfg!(feature = "lxr_no_lazy");
#[allow(dead_code)]
pub const NO_LAZY_DEC_THRESHOLD: usize = 100;
#[allow(dead_code)]
pub const RC_NURSERY_EVACUATION: bool = !cfg!(feature = "lxr_no_nursery_evac");
#[allow(dead_code)]
pub const RC_MATURE_EVACUATION: bool = !cfg!(feature = "lxr_no_mature_evac");

// Additional compile-time flags referenced by the RC work packets / block-allocation / barrier
// (P3). All default to LXR's defaults (the `lxr_*` cargo features are undeclared here, so each
// `cfg!(feature=...)` is `false`). Inert until `MMTK_PLAN=LXR` flips `rc_enabled`.
/// Sweep/scan whole blocks rather than lines. Mirrors `crate::policy::immix::BLOCK_ONLY`.
#[allow(dead_code)]
pub const BLOCK_ONLY: bool = crate::policy::immix::BLOCK_ONLY;
/// Don't nursery-evacuate objects that live in recycled (reused) lines. LXR default = false.
#[allow(dead_code)]
pub const RC_DONT_EVACUATE_NURSERY_IN_RECYCLED_LINES: bool =
    cfg!(feature = "lxr_dont_evacuate_nursery_in_recycled_lines");
/// Barrier-takerate measurement build (counts fast/slow barrier hits). LXR default = false.
#[allow(dead_code)]
pub const TAKERATE_MEASUREMENT: bool = cfg!(feature = "lxr_measure_barrier_takerate");
/// Barrier-cost measurement build (forces the slow path / disables the fast path). Default false.
#[allow(dead_code)]
pub const BARRIER_MEASUREMENT: bool = cfg!(feature = "barrier_measurement");
/// Companion to `BARRIER_MEASUREMENT`: measure the fast path only (skip the slow path). Default false.
#[allow(dead_code)]
pub const BARRIER_MEASUREMENT_NO_SLOW: bool = cfg!(feature = "barrier_measurement_no_slow");

/// One more atomic-store per barrier slow-path if this value is smaller than 6.
#[allow(dead_code)]
pub const LOG_BYTES_PER_RC_LOCK_BIT: usize = {
    if cfg!(feature = "lxr_lock_3") {
        3
    } else if cfg!(feature = "lxr_lock_4") {
        4
    } else if cfg!(feature = "lxr_lock_5") {
        5
    } else if cfg!(feature = "lxr_lock_6") {
        6
    } else if cfg!(feature = "lxr_lock_7") {
        7
    } else if cfg!(feature = "lxr_lock_8") {
        8
    } else {
        // lxr_lock_9 and the default agree.
        9
    }
};

// ---------- Debugging flags ---------- //
#[allow(dead_code)]
pub const SLOW_CONCURRENT_MARKING: bool = false;
#[allow(dead_code)]
pub const INC_MAX_COPY_DEPTH: bool = false;
#[allow(dead_code)]
pub const PREFETCH: bool = !cfg!(feature = "lxr_no_prefetch");
#[allow(dead_code)]
pub const PREFETCH_HEADER: bool = PREFETCH;
#[allow(dead_code)]
pub const PREFETCH_MARK: bool = PREFETCH;
#[allow(dead_code)]
pub const PREFETCH_RC: bool = false;
#[allow(dead_code)]
pub const PREFETCH_STEP: usize = 8;

/// Dump the active LXR runtime configuration (only when verbose). Adapted from LXR's
/// `validate_features`; the JVM-specific feature dump is trimmed to what compiles in our tree.
#[allow(dead_code)]
pub fn validate_features(active_barrier: BarrierSelector, verbose: usize) {
    if verbose == 0 {
        return;
    }
    eprintln!("-------------------- LXR Args --------------------");
    eprintln!(" * barrier: {:?}", active_barrier);
    eprintln!(
        " * log_bytes_per_rc_lock_bit: {:?}",
        LOG_BYTES_PER_RC_LOCK_BIT
    );
    eprintln!(" * buffer_size: {:?}", BUFFER_SIZE);
    eprintln!(" * lazy_decrements: {:?}", LAZY_DECREMENTS);
    eprintln!(" * rc_nursery_evacuation: {:?}", RC_NURSERY_EVACUATION);
    eprintln!(" * rc_mature_evacuation: {:?}", RC_MATURE_EVACUATION);
    eprintln!("\n{:#?}", args());
    eprintln!("--------------------------------------------------");
}
