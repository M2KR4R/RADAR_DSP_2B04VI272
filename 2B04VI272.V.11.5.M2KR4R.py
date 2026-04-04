#!/usr/bin/env python3

# ⚡ M2KR4R ⚡

"""

--                                                          --

M2KR4R — Radar :: DSP :: SiMuLaTioN.

-- CODENAME :: 2B04VI272 --

-- 2B04VI272.V.11.5.M2KR4R --

-- ITERATION :: 34 --

developed = "Python 3.13.12, VS Code, macOS - Tahoe 26.4"
requires  = "Python >= 3.9, NumPy >= 1.21"

--                                                          --

"""


from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# -- NUMERICAL :: CONSTANTS --

EPS: float = 1e-12
SQRT2: float = math.sqrt(2.0)
# -- Branch disagreement logging: fraction of range cells; fusion remains strict majority vote. --
TMR_DISAGREEMENT_THRESHOLD: float = 0.5
TIMING_MARGIN: float = 0.8
WATCHDOG_SLEEP: float = 0.05
TWO_PI: float = 2.0 * math.pi
PSD_RATIO_FLOOR_DB: float = -300.0  # when linear PSD ratio <= 1 before log10
DEFAULT_ALPHA_SCALE: float = 1.0
HZ_PER_KHZ: float = 1e3
KF_COVARIANCE_CAP: float = 1e6
# -- Self-test :: fix --
SELFTEST_UNIFORM_PSD: float = 1e-6
SELFTEST_PEAK_BASE_PSD: float = 1e-8
SELFTEST_PEAK_BIN_OFFSET: int = 50
SELFTEST_VOTE_IDX_A: int = 100
SELFTEST_VOTE_IDX_B: int = 200
SELFTEST_FD_ABS_TOL_HZ: float = 1e-6
N_PFA_MONTE_CARLO_FRAMES: int = 1000
# -- Max | empirical - nominal | / nominal for Test 1b; coarse sanity check only (not a formal GO/NO-GO). --
PFA_SELFTEST_REL_TOL: float = 0.5
RNG_SEED_SELFTEST_PFA: int = 12345
RNG_SEED_MAIN: int = 42
RNG_SEED_PSD_CHECK: int = 0
# -- Default reference-branch CFAR geometry (branch 0 in default TMR set) --
REF_BRANCH_TRAIN: int = 128
REF_BRANCH_GUARD: int = 12

# -- LOGGING --

_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"
_LOG_DATEFMT: str = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
logger: logging.Logger = logging.getLogger(__name__)


def _log_structured(level: int, event: str, **fields: Any) -> None:
    """ -- Append sorted key=value pairs for grep-friendly structured logs. -- """
    if not fields:
        logger.log(level, "event=%s", event)
        return
    tail = " ".join(f"{k}={fields[k]!r}" for k in sorted(fields))
    logger.log(level, "event=%s | %s", event, tail)


# -- ESTIMATOR :: RESULT --

class TargetEstimate(NamedTuple):
    """ -- Peak-based estimates from fused PSD; dB field is peak/training PSD ratio, not classical SNR. -- """

    doppler_hz: float
    velocity_m_s: float
    # -- : 10*log10(peak_bin_psd / mean(training_bins_psd)); same FFT bin width on both sides. --
    peak_to_training_mean_psd_db: float


# -- CFG --

@dataclass
class BranchConfig:
    """ -- Single redundant processing branch: window, CFAR geometry, and noise injection. -- """

    window_func: Callable[[int], NDArray[np.float64]]
    train: int
    guard: int
    threshold_scale: float
    method: str
    sigma_noise: float
    window: Optional[NDArray[np.float64]] = None
    window_energy: float = 0.0  # sum(window**2), filled in RadarConfig.__post_init__


@dataclass
class RadarConfig:
    """ -- Global radar and simulation parameters. -- """

    N: int = 16384
    FS: float = 250_000.0
    CARRIER_FREQ: float = 10e9
    C: float = 3e8
    CFAR_TRAIN: int = REF_BRANCH_TRAIN
    CFAR_GUARD: int = REF_BRANCH_GUARD
    PFA: float = 1e-6
    NOISE_STD: float = 1.2
    TARGET_PROB: float = 0.5
    WATCHDOG_TIMEOUT: float = 1.5
    TRACK_CONFIRM: int = 2
    TRACK_LOST: int = 3
    DOPPLER_TOLERANCE: float = 1000.0
    DWELL_SLEEP: float = 0.3
    KF_PROCESS_NOISE: float = 2.0
    KF_MEASURE_NOISE: float = 5.0
    KF_INITIAL_COV: float = 1.0
    KF_DEFAULT_STATE_COV: float = 1.0
    SIM_CHT_VEL_MIN: float = 650.0
    SIM_CHT_VEL_MAX: float = 750.0
    SIM_STD_VEL_MIN: float = 350.0
    SIM_STD_VEL_MAX: float = 450.0
    SIM_CHT_AMP_MIN: float = 1.8
    SIM_CHT_AMP_MAX: float = 2.5
    SIM_STD_AMP_MIN: float = 6.0
    SIM_STD_AMP_MAX: float = 12.0
    SIM_SCINTILLATION_DEPTH: float = 0.2
    SIM_SCINTILLATION_FREQ_HZ: float = 2.0
    SIM_CHT_CLASS_PROB: float = 0.5
    SIM_SCINTILLATION_BASE: float = 1.0
    branches: List[BranchConfig] = field(
        default_factory=lambda: [
            BranchConfig(
                window_func=np.hanning,
                train=REF_BRANCH_TRAIN,
                guard=REF_BRANCH_GUARD,
                threshold_scale=1.0,
                method="ca",
                sigma_noise=0.2,
            ),
            BranchConfig(
                window_func=np.hamming,
                train=64,
                guard=8,
                threshold_scale=0.85,
                method="ca",
                sigma_noise=0.5,
            ),
            BranchConfig(
                window_func=np.blackman,
                train=192,
                guard=16,
                threshold_scale=5.0,
                method="simple",
                sigma_noise=1.0,
            ),
        ]
    )
    freqs: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError("N must be positive")
        if self.FS <= 0:
            raise ValueError("FS must be positive")
        for branch in self.branches:
            branch.window = branch.window_func(self.N)
            w = branch.window.astype(np.float64, copy=False)
            branch.window_energy = float(np.sum(w * w))
            branch.window_energy = max(branch.window_energy, EPS)
        self.freqs = np.fft.fftshift(np.fft.fftfreq(self.N, d=1.0 / self.FS))


# -- PIPELINE :: STATE --

@dataclass
class PipelineState:
    psd_data: List[NDArray[np.float64]]
    final_detections: NDArray[np.bool_]
    votes: NDArray[np.int32]
    reset_event: threading.Event = field(default_factory=threading.Event)


# -- WATCHDOG --

class Watchdog:
    """ -- Monitors main-loop liveness; sets reset_event on timeout. -- """

    def __init__(self, config: RadarConfig, reset_event: threading.Event) -> None:
        self.timeout: float = config.WATCHDOG_TIMEOUT
        self.reset_event: threading.Event = reset_event
        self.last_kick: float = time.monotonic()
        self.alive: bool = True
        self.lock: threading.Lock = threading.Lock()
        self._thread: threading.Thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def kick(self) -> None:
        with self.lock:
            self.last_kick = time.monotonic()

    def _monitor(self) -> None:
        while self.alive:
            with self.lock:
                elapsed = time.monotonic() - self.last_kick
            if elapsed > self.timeout:
                _log_structured(logging.WARNING, "watchdog_timeout", elapsed_s=round(elapsed, 4))
                self.reset_event.set()
                with self.lock:
                    self.last_kick = time.monotonic()
            time.sleep(WATCHDOG_SLEEP)

    def stop(self) -> None:
        self.alive = False


# -- DSP --

_cfar_kernel_cache: Dict[Tuple[int, int], NDArray[np.float64]] = {}


def _get_cfar_kernel(train: int, guard: int) -> NDArray[np.float64]:
    key = (train, guard)
    if key not in _cfar_kernel_cache:
        kernel_len = 2 * (train + guard) + 1
        kernel = np.zeros(kernel_len, dtype=np.float64)
        center = train + guard
        for offset in range(-(train + guard), -guard):
            kernel[center + offset] = 1.0
        for offset in range(guard, guard + train):
            kernel[center + offset] = 1.0
        _cfar_kernel_cache[key] = kernel
    return _cfar_kernel_cache[key]


def ca_cfar(
    psd: NDArray[np.float64],
    train: int,
    guard: int,
    pfa: float,
    alpha_scale: float = DEFAULT_ALPHA_SCALE,
) -> NDArray[np.bool_]:
    """
    -- Cell-averaging CFAR with circular edge padding. --
    Returns:
        -- Boolean detection mask, same shape as ``psd``. --
    Raises:
        -- ValueError: If ``pfa`` is not in (0, 1), or inputs are inconsistent. --
    Notes:
        -- Returns all-false if ``train == 0``, length too small, or ``psd`` is empty. --
        -- Non-finite values in ``psd`` are treated as zero for the threshold test. --
    """
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa must lie in (0, 1)")
    if psd.size == 0:
        return np.zeros(0, dtype=bool)
    if train == 0:
        return np.zeros_like(psd, dtype=bool)
    n_cells = 2 * train
    alpha = n_cells * (pfa ** (-1.0 / n_cells) - 1.0) * alpha_scale
    length = int(psd.shape[0])
    if length <= 2 * (train + guard):
        return np.zeros_like(psd, dtype=bool)
    psd_safe = np.nan_to_num(np.asarray(psd, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pad_len = train + guard
    psd_padded = np.concatenate([psd_safe[-pad_len:], psd_safe, psd_safe[:pad_len]])
    kernel = _get_cfar_kernel(train, guard)
    sums = np.correlate(psd_padded, kernel, mode="valid")
    noise_floor = sums / (2.0 * train)
    noise_floor = np.maximum(noise_floor, EPS)
    return psd_safe > noise_floor * alpha


def _psd_fftshift_complex_windowed(
    windowed: NDArray[np.complex128], fs: float, window_energy: float
) -> NDArray[np.float64]:
    """
    -- Two-sided periodogram (linear power) for complex baseband: fftshift(|FFT(x)|^2) / (fs * U). --
    -- Sign of Doppler is unambiguous relative to fftfreq ordering. --
    """
    n = int(windowed.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    u = max(float(window_energy), EPS)
    spec = np.fft.fft(np.asarray(windowed, dtype=np.complex128))
    p_full = (np.abs(spec) ** 2).astype(np.float64, copy=False)
    scale = fs * u
    return np.fft.fftshift(p_full / scale)


def _tmr_outlier_branch(detections: List[NDArray[np.bool_]]) -> Optional[int]:
    """
    -- Heuristic index of the branch that disagrees most with the others, for logging only. --
    -- Fusion is strict majority vote over all branches; this does not mask or drop a branch. --
    """
    n = len(detections)
    if n < 2:
        return None
    n_cells = int(detections[0].size)
    if n_cells == 0:
        return None
    pairwise: List[List[int]] = []
    for i in range(n):
        row = [int(np.sum(np.not_equal(detections[i], detections[j]))) for j in range(n)]
        pairwise.append(row)
    disagreement = [sum(pairwise[i][j] for j in range(n) if j != i) for i in range(n)]
    if max(disagreement) > TMR_DISAGREEMENT_THRESHOLD * n_cells:
        return int(np.argmax(disagreement))
    return None


def _sanitize_complex(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """ -- Replace -- NON-finite -- complex samples with zero (nan_to_num is -- FLOAT-only -- on some :: NumPy builds). -- """
    a = np.asarray(x, dtype=np.complex128)
    out = np.where(np.isfinite(a.real) & np.isfinite(a.imag), a, 0.0 + 0.0j)
    return out.astype(np.complex128, copy=False)


def process_tmr(
    raw_frame: NDArray[np.complex128],
    config: RadarConfig,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.bool_], List[NDArray[np.float64]], NDArray[np.int32]]:
    """
    -- Run each branch :: (window + PSD + detector) and fuse with strict majority vote. --
    Returns:
        -- Tuple of ``(final_detections, per_branch_psd_list, vote_counts)``. --
    Raises:
        -- ValueError: If ``raw_frame`` length does not match ``config.N`` or branch list is empty. --
    Notes:
        -- NON-finite -- samples in ``raw_frame`` are replaced with zero before processing. --
        -- Per-branch noise is circular complex Gaussian with per-axis variance (sigma/sqrt(2))^2. --
    """
    if not config.branches:
        raise ValueError("config.branches must be non-empty")
    if int(raw_frame.shape[0]) != config.N:
        raise ValueError(f"raw_frame length {raw_frame.shape[0]} != config.N {config.N}")
    raw_safe = _sanitize_complex(np.asarray(raw_frame, dtype=np.complex128))
    detections: List[NDArray[np.bool_]] = []
    psd_data: List[NDArray[np.float64]] = []
    for branch in config.branches:
        assert branch.window is not None
        sig = branch.sigma_noise / SQRT2
        n_i = rng.normal(0.0, sig, config.N)
        n_q = rng.normal(0.0, sig, config.N)
        perturbed = raw_safe + (n_i + 1j * n_q)
        windowed = perturbed * branch.window
        psd = _psd_fftshift_complex_windowed(windowed, config.FS, branch.window_energy)
        psd_data.append(psd)
        if branch.method == "ca":
            det = ca_cfar(psd, branch.train, branch.guard, config.PFA, branch.threshold_scale)
        else:
            med = np.median(psd)
            noise_floor = float(med) if np.isfinite(med) else EPS
            noise_floor = max(noise_floor, EPS)
            det = psd > branch.threshold_scale * noise_floor
        detections.append(det)
    n_br = len(detections)
    votes = np.zeros_like(detections[0], dtype=np.int32)
    for det in detections:
        votes += det.astype(np.int32, copy=False)
    need = n_br // 2 + 1
    final_detections = votes >= need
    faulty = _tmr_outlier_branch(detections)
    if faulty is not None:
        _log_structured(
            logging.WARNING,
            "tmr_branch_disagreement_log_only",
            branch_index=faulty,
            note="fusion_unchanged_majority_vote",
        )
    return final_detections, psd_data, votes


def estimate_target(
    freqs: NDArray[np.float64],
    psd: NDArray[np.float64],
    detections: NDArray[np.bool_],
    config: RadarConfig,
    train: Optional[int] = None,
    guard: Optional[int] = None,
) -> TargetEstimate:
    """
    -- Estimate Doppler (Hz), velocity (m/s), and PSD ratio in dB from fused detections. --

    -- The dB value is ``10*log10(P_peak / mean(P_training))`` with identical bin width for
    numerator and denominator; it is a convenient spectral contrast metric, not a calibrated
    system SNR. --
    """
    if not config.branches:
        raise ValueError("config.branches must be non-empty")
    n = int(psd.shape[0])
    if int(freqs.shape[0]) != n:
        raise ValueError("freqs and psd must have the same length")
    if detections.size != n:
        raise ValueError("detections must match psd shape")
    if not np.any(detections):
        return TargetEstimate(0.0, 0.0, 0.0)
    psd_safe = np.nan_to_num(np.asarray(psd, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    freqs_safe = np.nan_to_num(np.asarray(freqs, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    det_psd = psd_safe[detections]
    if det_psd.size == 0 or not np.any(np.isfinite(det_psd)):
        return TargetEstimate(0.0, 0.0, 0.0)
    idx_det = np.flatnonzero(detections)
    peak_rel = int(np.nanargmax(det_psd))
    peak_idx = int(idx_det[peak_rel])
    if not (0 <= peak_idx < n):
        return TargetEstimate(0.0, 0.0, 0.0)
    fd_est = float(freqs_safe[peak_idx])
    if not np.isfinite(fd_est):
        fd_est = 0.0
    denom = 2.0 * config.CARRIER_FREQ
    if denom <= 0 or not math.isfinite(denom):
        raise ValueError("CARRIER_FREQ must be positive finite")
    vel_est = (fd_est * config.C) / denom
    ref = config.branches[0]
    tr = ref.train if train is None else int(train)
    gd = ref.guard if guard is None else int(guard)
    if tr < 0 or gd < 0:
        raise ValueError("train and guard must be non-negative")
    left_start = max(0, peak_idx - (tr + gd))
    left_end = max(0, peak_idx - gd)
    right_start = min(n, peak_idx + gd + 1)
    right_end = min(n, peak_idx + gd + tr + 1)
    training = np.concatenate([psd_safe[left_start:left_end], psd_safe[right_start:right_end]])
    if training.size == 0:
        local_noise = EPS
    else:
        local_noise = float(np.mean(training))
        if not np.isfinite(local_noise):
            local_noise = EPS
    local_noise = max(local_noise, EPS)
    signal_power = float(psd_safe[peak_idx])
    if not np.isfinite(signal_power):
        signal_power = EPS
    signal_power = max(signal_power, EPS)
    ratio = signal_power / local_noise
    if ratio <= 0.0 or not np.isfinite(ratio):
        psd_ratio_db = PSD_RATIO_FLOOR_DB
    else:
        psd_ratio_db = 10.0 * math.log10(ratio)
    return TargetEstimate(fd_est, vel_est, float(psd_ratio_db))


# -- SIGNAL :: GENERATOR --

def generate_rf_frame(
    config: RadarConfig, rng: np.random.Generator
) -> Tuple[NDArray[np.complex128], Optional[float]]:
    """
    -- Synthesize one dwell of complex baseband (I/Q) samples. --

    -- Target: positive-frequency tone ``exp(j*2*pi*fd*t)`` so Doppler sign matches ``fftfreq``. --
    -- Noise: circular complex Gaussian with E[|n|^2] = NOISE_STD^2. --
    """
    if config.N <= 0 or config.FS <= 0:
        raise ValueError("N and FS must be positive")
    t = np.arange(config.N, dtype=np.float64) / config.FS
    n_i = rng.normal(0.0, config.NOISE_STD / SQRT2, config.N)
    n_q = rng.normal(0.0, config.NOISE_STD / SQRT2, config.N)
    noise = (n_i + 1j * n_q).astype(np.complex128, copy=False)
    if rng.random() > config.TARGET_PROB:
        return noise, None
    if rng.random() < config.SIM_CHT_CLASS_PROB:
        v_target = float(rng.uniform(config.SIM_CHT_VEL_MIN, config.SIM_CHT_VEL_MAX))
        amplitude = float(rng.uniform(config.SIM_CHT_AMP_MIN, config.SIM_CHT_AMP_MAX))
    else:
        v_target = float(rng.uniform(config.SIM_STD_VEL_MIN, config.SIM_STD_VEL_MAX))
        amplitude = float(rng.uniform(config.SIM_STD_AMP_MIN, config.SIM_STD_AMP_MAX))
    fd = (2.0 * v_target * config.CARRIER_FREQ) / config.C
    scintillation = config.SIM_SCINTILLATION_BASE + config.SIM_SCINTILLATION_DEPTH * np.sin(
        TWO_PI * config.SIM_SCINTILLATION_FREQ_HZ * t
    )
    phase = TWO_PI * fd * t
    target_return = (amplitude * scintillation * (np.cos(phase) + 1j * np.sin(phase))).astype(
        np.complex128, copy=False
    )
    return noise + target_return, float(fd)


# -- KALMAN --

class KalmanFilter1D:
    """
    -- Scalar random-walk process :: with one measurement per update. --
    -- This class does not raise; -- NON-finite -- measurements are ignored in ``update``. --
    """

    def __init__(
        self,
        initial_vel: float = 0.0,
        process_noise: float = 1.0,
        measure_noise: float = 10.0,
        initial_cov: float = 1.0,
    ) -> None:
        self.x: float = float(initial_vel)
        self.P: float = float(initial_cov)
        self.Q: float = float(process_noise)
        self.R: float = float(measure_noise)

    def predict(self) -> None:
        """ -- Propagate covariance :: by process noise (state mean unchanged). -- """
        self.P += self.Q
        if self.P > KF_COVARIANCE_CAP:
            self.P = KF_COVARIANCE_CAP

    def update(self, z: float) -> None:
        """ -- Incorporate measurement ``z``; no-op if ``z`` or gains are -- NON-finite. -- """
        if not math.isfinite(z):
            return
        denom = self.P + self.R
        if denom <= EPS or not math.isfinite(denom):
            return
        k_gain = self.P / denom
        if not math.isfinite(k_gain):
            return
        self.x += k_gain * (z - self.x)
        self.P *= 1.0 - k_gain


# -- TRACKER --

class RadarTracker:
    """
    -- Search / lock state machine with Kalman smoothing on reported velocity. --
    -- Does not raise during ``update``; expects finite inputs from the estimator when a hit is true. --
    """

    def __init__(self, config: RadarConfig) -> None:
        self.config: RadarConfig = config
        self.state: str = "SEARCH"
        self.lock_counter: int = 0
        self.miss_counter: int = 0
        self.last_fd: float = 0.0
        self.last_vel: float = 0.0
        self.last_psd_ratio_db: float = 0.0
        self.kf: KalmanFilter1D = KalmanFilter1D(
            process_noise=config.KF_PROCESS_NOISE,
            measure_noise=config.KF_MEASURE_NOISE,
            initial_cov=config.KF_DEFAULT_STATE_COV,
        )
        self.kf_initialized: bool = False

    def reset(self) -> None:
        """ -- Clear track and :: KALMAN state (e.g. after watchdog pipeline reset). -- """
        self.state = "SEARCH"
        self.lock_counter = 0
        self.miss_counter = 0
        self.last_fd = self.last_vel = self.last_psd_ratio_db = 0.0
        self.kf_initialized = False
        self.kf = KalmanFilter1D(
            process_noise=self.config.KF_PROCESS_NOISE,
            measure_noise=self.config.KF_MEASURE_NOISE,
            initial_cov=self.config.KF_DEFAULT_STATE_COV,
        )

    def update(
        self,
        target_hit: bool,
        cur_fd: float,
        cur_vel: float,
        cur_psd_ratio_db: float,
    ) -> Tuple[str, float, float, float]:
        """
        -- Returns:
            ``(state, out_fd_hz, out_vel_m_s, out_psd_ratio_db)`` with ``state`` in ``{"SEARCH","LOCKED"}``. --
        """
        valid = False
        if self.state == "SEARCH":
            if target_hit:
                valid = True
        else:
            if target_hit and abs(cur_fd - self.last_fd) <= self.config.DOPPLER_TOLERANCE:
                valid = True
        if valid:
            self.lock_counter += 1
            self.miss_counter = 0
            self.last_fd, self.last_vel, self.last_psd_ratio_db = (
                cur_fd,
                cur_vel,
                cur_psd_ratio_db,
            )
        else:
            self.miss_counter += 1
            self.lock_counter = 0
        if self.state == "SEARCH" and self.lock_counter >= self.config.TRACK_CONFIRM:
            self.state = "LOCKED"
        elif self.state == "LOCKED" and self.miss_counter >= self.config.TRACK_LOST:
            self.state = "SEARCH"
            self.lock_counter = 0
            self.miss_counter = 0
            self.kf_initialized = False
            self.kf = KalmanFilter1D(
                process_noise=self.config.KF_PROCESS_NOISE,
                measure_noise=self.config.KF_MEASURE_NOISE,
                initial_cov=self.config.KF_DEFAULT_STATE_COV,
            )
        if self.state == "LOCKED":
            out_fd = cur_fd if valid else self.last_fd
            out_vel = cur_vel if valid else self.last_vel
            out_psd_ratio_db = cur_psd_ratio_db if valid else self.last_psd_ratio_db
        else:
            out_fd = cur_fd if target_hit else 0.0
            out_vel = cur_vel if target_hit else 0.0
            out_psd_ratio_db = cur_psd_ratio_db if target_hit else 0.0
        if target_hit and valid:
            if not self.kf_initialized:
                self.kf.x = cur_vel
                self.kf.P = self.config.KF_INITIAL_COV
                self.kf_initialized = True
            else:
                self.kf.predict()
                self.kf.update(cur_vel)
            out_vel = self.kf.x
        elif self.kf_initialized and self.state == "LOCKED":
            self.kf.predict()
            out_vel = self.kf.x
        return self.state, out_fd, out_vel, out_psd_ratio_db


# -- SELF :: TESTS --

def _assert_psd_complex_matches_fft(config: RadarConfig) -> bool:
    """ -- Internal consistency: complex PSD path matches reference FFT layout. -- """
    rng = np.random.default_rng(RNG_SEED_PSD_CHECK)
    x = (
        (rng.standard_normal(config.N) + 1j * rng.standard_normal(config.N))
        / SQRT2
    ).astype(np.complex128, copy=False)
    w = config.branches[0].window
    assert w is not None
    windowed = x * w
    u = config.branches[0].window_energy
    ref = np.fft.fftshift(np.abs(np.fft.fft(windowed)) ** 2) / (config.FS * max(u, EPS))
    tst = _psd_fftshift_complex_windowed(windowed, config.FS, u)
    return bool(np.allclose(ref, tst, rtol=1e-10, atol=1e-12))


def run_self_tests(config: RadarConfig) -> bool:
    """
    -- Run built-in sanity checks. --
    -- Test 1b: i.i.d. exponential PSD per cell models |X|^2 for X complex Gaussian per bin
    (squared-magnitude of complex noise in frequency). -- It is not a claim that real-passband
    rfft bins are exponentially distributed. --  Pass/fail uses ``PFA_SELFTEST_REL_TOL`` on
    relative error vs nominal PFA (coarse Monte Carlo sanity check, not a formal test). --
    """
    _log_structured(logging.INFO, "self_tests_start")
    if not _assert_psd_complex_matches_fft(config):
        logger.error("Test 0 failed: complex PSD path mismatch vs reference FFT")
        return False
    logger.info("Test 0 passed: complex PSD matches reference FFT layout")
    noise_psd = np.ones(config.N, dtype=np.float64) * SELFTEST_UNIFORM_PSD
    det = ca_cfar(noise_psd, config.CFAR_TRAIN, config.CFAR_GUARD, config.PFA, DEFAULT_ALPHA_SCALE)
    if np.any(det):
        logger.error("Test 1 failed: CFAR detected on uniform noise")
        return False
    logger.info("Test 1 passed: CFAR on noise OK")
    rng_pfa = np.random.default_rng(RNG_SEED_SELFTEST_PFA)
    total_cfar_hits = 0
    frames_with_any_detection = 0
    for _ in range(N_PFA_MONTE_CARLO_FRAMES):
        # -- Models complex-Gaussian :: periodogram magnitudes squared (pedagogical PFA check). --
        noise_psd_exp = rng_pfa.exponential(scale=1.0, size=config.N)
        det_pfa = ca_cfar(
            noise_psd_exp,
            config.CFAR_TRAIN,
            config.CFAR_GUARD,
            config.PFA,
            DEFAULT_ALPHA_SCALE,
        )
        nh = int(np.sum(det_pfa))
        total_cfar_hits += nh
        if nh > 0:
            frames_with_any_detection += 1
    total_cells = N_PFA_MONTE_CARLO_FRAMES * config.N
    empirical_PFA = total_cfar_hits / float(total_cells)
    frame_trigger_rate = frames_with_any_detection / float(N_PFA_MONTE_CARLO_FRAMES)
    nominal_pfa = max(float(config.PFA), EPS)
    pfa_rel_err = abs(empirical_PFA - config.PFA) / nominal_pfa
    if pfa_rel_err > PFA_SELFTEST_REL_TOL:
        _log_structured(
            logging.ERROR,
            "self_test_cfar_pfa_fail",
            empirical_PFA=empirical_PFA,
            nominal_PFA=config.PFA,
            pfa_relative_error=pfa_rel_err,
            pfa_relative_tol=PFA_SELFTEST_REL_TOL,
            n_frames=N_PFA_MONTE_CARLO_FRAMES,
            frame_trigger_rate=frame_trigger_rate,
        )
        logger.error(
            "Test 1b failed: empirical per-cell PFA %.6g vs nominal %.6g "
            "(relative error %.4g exceeds sanity tol %.4g)",
            empirical_PFA,
            config.PFA,
            pfa_rel_err,
            PFA_SELFTEST_REL_TOL,
        )
        return False
    _log_structured(
        logging.INFO,
        "self_test_cfar_pfa_pass",
        empirical_PFA=empirical_PFA,
        nominal_PFA=config.PFA,
        pfa_relative_error=pfa_rel_err,
        pfa_relative_tol=PFA_SELFTEST_REL_TOL,
        frame_trigger_rate=frame_trigger_rate,
    )
    logger.info(
        "Test 1b passed: CFAR exponential-PSD (complex-Gaussian |bin|^2 model) "
        "empirical per-cell PFA=%.6g nominal=%.6g; frames_with_any_detection=%s/%s",
        empirical_PFA,
        config.PFA,
        frames_with_any_detection,
        N_PFA_MONTE_CARLO_FRAMES,
    )
    n_br = len(config.branches)
    dummy_det = [np.zeros(config.N, dtype=bool) for _ in range(n_br)]
    dummy_det[0][SELFTEST_VOTE_IDX_A] = True
    dummy_det[1][SELFTEST_VOTE_IDX_A] = True
    if n_br > 2:
        dummy_det[2][SELFTEST_VOTE_IDX_B] = True
    votes = np.zeros(config.N, dtype=np.int32)
    for d in dummy_det:
        votes += d.astype(np.int32, copy=False)
    need = n_br // 2 + 1
    final = votes >= need
    if not final[SELFTEST_VOTE_IDX_A] or (n_br > 2 and final[SELFTEST_VOTE_IDX_B]):
        logger.error("Test 2 failed: majority voting incorrect")
        return False
    logger.info("Test 2 passed: majority voting OK")
    assert config.freqs is not None
    psd_peak = np.ones(config.N, dtype=np.float64) * SELFTEST_PEAK_BASE_PSD
    peak_i = config.N // 2 + SELFTEST_PEAK_BIN_OFFSET
    if not (0 <= peak_i < config.N):
        logger.error("Test 3 failed: peak index out of bounds")
        return False
    psd_peak[peak_i] = 1.0
    mask = np.zeros(config.N, dtype=bool)
    mask[peak_i] = True
    est = estimate_target(config.freqs, psd_peak, mask, config)
    fd, vel, psd_db = est
    if not math.isfinite(fd) or not math.isfinite(vel) or not math.isfinite(psd_db):
        logger.error("Test 3 failed: estimate_target with synthetic peak")
        return False
    if abs(fd - float(config.freqs[peak_i])) > SELFTEST_FD_ABS_TOL_HZ:
        logger.error("Test 3 failed: Doppler estimate mismatch")
        return False
    logger.info("Test 3 passed: estimate_target peak OK")
    _log_structured(logging.INFO, "self_tests_pass")
    return True


# -- MAIN --

def _empty_psd_stack(config: RadarConfig) -> List[NDArray[np.float64]]:
    return [np.zeros(config.N, dtype=np.float64) for _ in range(len(config.branches))]


def main() -> None:
    """
    -- Run the simulation loop :: until KeyboardInterrupt. --
    Raises:
        -- RuntimeError: If ``config.freqs`` is missing after ``RadarConfig()`` (should not occur). --
    """
    config = RadarConfig()
    if not run_self_tests(config):
        _log_structured(logging.WARNING, "self_tests_failed")
    n_br = len(config.branches)
    state = PipelineState(
        psd_data=_empty_psd_stack(config),
        final_detections=np.zeros(config.N, dtype=bool),
        votes=np.zeros(config.N, dtype=np.int32),
    )
    _log_structured(
        logging.INFO,
        "simulation_start",
        fs_khz=round(config.FS / HZ_PER_KHZ, 3),
        n=config.N,
        branches=n_br,
    )
    rng = np.random.default_rng(RNG_SEED_MAIN)
    tracker = RadarTracker(config)

    def reset_pipeline() -> None:
        _log_structured(logging.WARNING, "pipeline_reset")
        state.psd_data = _empty_psd_stack(config)
        state.final_detections = np.zeros(config.N, dtype=bool)
        state.votes = np.zeros(config.N, dtype=np.int32)
        tracker.reset()

    wd = Watchdog(config, state.reset_event)
    prev_tracker_state: str = tracker.state
    try:
        while True:
            loop_start = time.monotonic()
            if state.reset_event.is_set():
                reset_pipeline()
                state.reset_event.clear()
            raw_frame, true_fd = generate_rf_frame(config, rng)
            wd.kick()
            state.final_detections, state.psd_data, state.votes = process_tmr(
                raw_frame, config, rng
            )
            target_hit = bool(np.any(state.final_detections))
            max_votes = int(np.max(state.votes)) if target_hit else 0
            cur_fd: float = 0.0
            cur_vel: float = 0.0
            cur_psd_ratio_db: float = 0.0
            freqs = config.freqs
            if freqs is None:
                raise RuntimeError("config.freqs not initialized")
            if target_hit:
                # -- Heuristic: choose the branch whose fused-detection PSD peak is largest. --
                # -- Fused mask is shared; branches can disagree on bin height. A production --
                # -- system might average PSDs or use a single reference branch; this picks one PSD --
                # -- surface for peak and PSD-ratio-metric extraction in simulation. --
                best_idx = int(
                    np.argmax(
                        [np.max(psd[state.final_detections]) for psd in state.psd_data]
                    )
                )
                est = estimate_target(freqs, state.psd_data[best_idx], state.final_detections, config)
                cur_fd, cur_vel, cur_psd_ratio_db = est
            tracker_state, out_fd, out_vel, out_psd_ratio_db = tracker.update(
                target_hit, cur_fd, cur_vel, cur_psd_ratio_db
            )
            if tracker_state != prev_tracker_state:
                _log_structured(
                    logging.INFO,
                    "tracker_transition",
                    from_state=prev_tracker_state,
                    to_state=tracker_state,
                )
                prev_tracker_state = tracker_state
            proc_time = time.monotonic() - loop_start
            margin_threshold = TIMING_MARGIN * config.WATCHDOG_TIMEOUT
            if proc_time > margin_threshold:
                _log_structured(
                    logging.WARNING,
                    "processing_slow",
                    proc_time_s=round(proc_time, 4),
                    margin_threshold_s=round(margin_threshold, 4),
                    margin_pct=round(TIMING_MARGIN * 100.0, 1),
                )
            ground_truth_target = "Y" if true_fd is not None else "N"
            _log_structured(
                logging.INFO,
                "dwell",
                state=tracker_state,
                ground_truth_target=ground_truth_target,
                votes=f"{max_votes}/{n_br}",
                psd_peak_to_training_mean_db=round(out_psd_ratio_db, 2),
                fd_hz=round(out_fd, 2),
                vel_m_s=round(out_vel, 3),
            )
            time.sleep(config.DWELL_SLEEP)
    except KeyboardInterrupt:
        wd.stop()
        _log_structured(logging.INFO, "simulation_stopped", reason="KeyboardInterrupt")


if __name__ == "__main__":
    main()


"""

-- END OF PIPELINE
-- SPARK :: M2KR4R.V.11.5.2B04VI272
-- STATE :: SIMULATION_READY
-- SIGNATURE :: 2B04VI272-34-VERIFIED

"""