
--                                                          --

M2KR4R — Radar :: DSP :: SiMuLaTioN.

-- CODENAME :: 2B04VI272 --

-- 2B04VI272.V.11.5.M2KR4R --

-- ITERATION :: 34 --

developed = "Python 3.13.12, VS Code, macOS - Tahoe 26.4"
requires  = "Python >= 3.9, NumPy >= 1.21"

--                                                          --


-- This is a Simulation :: Not a real‑time radar processor. --

This is a Python simulation, I built while learning radar signal processing just for fun. It started as a hobby script, but as I dug deeper into how Doppler radars work (windowing, FFT, CFAR, tracking), I turned it into a single‑block demonstration of a complete radar DSP pipeline. No military claims, no hardware – just code that shows the logic.


-- How the radar simulation works --

1. -- Synthetic I/Q baseband --
– The script generates complex samples (I/Q) at 250 kHz. A target appears randomly (50% probability) as a complex exponential `exp(j·2π·fd·t)`, which makes Doppler sign unambiguous. Noise is circular complex Gaussian with adjustable standard deviation.

2. -- Windowing --
– Three parallel branches
(Triple Modular Redundancy style) each apply a different window:
   - Branch 0: Hanning (128 training cells, 12 guard cells)
   - Branch 1: Hamming (64 / 8)
   - Branch 2: Blackman (192 / 16)
   Windows reduce spectral leakage before the FFT.

3. -- FFT → PSD --
– Each branch computes a two‑sided power spectral density:
   `PSD = |FFT(windowed_samples)|² / (fs · sum(window²))`
   This gives correct scaling (W/Hz).

4. -- CA‑CFAR detection --
– Cell‑Averaging Constant False Alarm Rate:
   - For each Doppler bin, the noise floor is the average of training cells (left and right, excluding guard cells).
   - Threshold = `α · noise_floor`, where `α` is derived from the desired PFA (here 1e‑6).
   - Detections are binary masks.

5. -- TMR fusion & majority vote --
– All three branches produce detection masks. The final detection is `True` where at least two branches agree (strict majority). A separate log entry warns when one branch strongly disagrees – but the fusion never drops a branch; it only logs the outlier.

6. -- Peak estimation --
– From the fused detection mask, the script picks the Doppler bin with the highest PSD (using the PSD of the branch that shows the strongest peak inside the mask). That bin gives Doppler frequency, radial velocity, and a PSD contrast metric:
`10·log10(peak_PSD / mean(training_PSD))` – not true SNR, but a useful spectral contrast.

7. -- Tracker & Kalman Filter --
– A simple state machine:
   - `SEARCH` → needs `TRACK_CONFIRM` (2) consecutive detections to go `LOCKED`.
   - `LOCKED` → loses track after `TRACK_LOST` (3) consecutive misses.
   A 1‑D Kalman filter smooths velocity estimates while locked. During misses, the filter coasts (predicts) and the last valid Doppler/velocity is held.

8. -- Watchdog --
– A background thread monitors the main loop. If processing takes longer than 1.5 seconds, it triggers a pipeline reset (clears all states and restarts the tracker).


-- What aircraft speeds are being simulated? --

-- The simulation randomly chooses between two target classes :: --

Class | Speed range (m/s) | Speed range (km/h) | Mach (approx.)

CHT -- (high‑speed) | 650 – 750 | 2340 – 2700 | Mach 1.9 – 2.2 |
STD -- (standard) | 350 – 450 | 1260 – 1620 | Mach 1.05 – 1.35 |

N.O.T.E :: -- CHT : Cheetah | STD : Standard

The carrier frequency is 10 GHz (X‑band), so Doppler shift = `2·v·fc / c`.
For v = 700 m/s → fd ≈ 46.7 kHz – which matches the logged values (~46‑49 kHz).


-- How the majority voting (TMR) works --

- Each branch outputs a boolean array (detection / no detection) for every Doppler bin.
- `votes = sum(detection_masks)` → integer per bin.
- Final detection = `votes >= ceil(branches/2)`.
- If one branch’s detection mask differs strongly from the others (disagreement > 50% of cells), a warning is logged, but the majority vote still stands.
- This is not a fault‑masking system; it’s a simple fusion to show how redundancy improves robustness in simulation.



-- Running the :: code and interpreting the logs --

   -- Excerpt from :: simulation results .. --


   -- Excerpt from :: simulation results .. --



<pre style="overflow-x: auto; white-space: pre; background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-family: monospace;">


22:00:42 [INFO] event=self_tests_start

22:00:42 [INFO] Test 0 passed: complex PSD matches reference FFT layout

22:00:42 [INFO] Test 1 passed: CFAR on noise OK

22:00:43 [INFO] event=self_test_cfar_pfa_pass | empirical_PFA=6.7138671875e-07 frame_trigger_rate=0.011 nominal_PFA=1e-06 pfa_relative_error=0.32861328125 pfa_relative_tol=0.5

22:00:43 [INFO] Test 1b passed: CFAR exponential-PSD (complex-Gaussian |bin|^2 model) empirical per-cell PFA=6.71387e-07 nominal=1e-06; frames_with_any_detection=11/1000

22:00:43 [INFO] Test 2 passed: majority voting OK

22:00:43 [INFO] Test 3 passed: estimate_target peak OK

22:00:43 [INFO] event=self_tests_pass

22:00:43 [INFO] event=simulation_start | branches=3 fs_khz=250.0 n=16384

22:00:43 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:43 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:43 [INFO] event=dwell | fd_hz=48553.47 ground_truth_target='Y' psd_peak_to_training_mean_db=44.38 state='SEARCH' vel_m_s=728.302 votes='3/3'

22:00:44 [INFO] event=tracker_transition | from_state='SEARCH' to_state='LOCKED'

22:00:44 [INFO] event=dwell | fd_hz=27664.18 ground_truth_target='Y' psd_peak_to_training_mean_db=53.91 state='LOCKED' vel_m_s=610.8 votes='3/3'

22:00:44 [INFO] event=dwell | fd_hz=27664.18 ground_truth_target='Y' psd_peak_to_training_mean_db=53.91 state='LOCKED' vel_m_s=610.8 votes='3/3'

22:00:44 [INFO] event=dwell | fd_hz=27664.18 ground_truth_target='Y' psd_peak_to_training_mean_db=53.91 state='LOCKED' vel_m_s=610.8 votes='3/3'

22:00:45 [INFO] event=tracker_transition | from_state='LOCKED' to_state='SEARCH'

22:00:45 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:45 [INFO] event=dwell | fd_hz=-111068.73 ground_truth_target='N' psd_peak_to_training_mean_db=10.56 state='SEARCH' vel_m_s=-1666.031 votes='2/3'

22:00:45 [INFO] event=tracker_transition | from_state='SEARCH' to_state='LOCKED'

22:00:45 [INFO] event=dwell | fd_hz=46691.89 ground_truth_target='Y' psd_peak_to_training_mean_db=45.78 state='LOCKED' vel_m_s=-778.627 votes='3/3'

22:00:46 [INFO] event=dwell | fd_hz=46691.89 ground_truth_target='N' psd_peak_to_training_mean_db=45.78 state='LOCKED' vel_m_s=-778.627 votes='2/3'

22:00:46 [INFO] event=dwell | fd_hz=46691.89 ground_truth_target='N' psd_peak_to_training_mean_db=45.78 state='LOCKED' vel_m_s=-778.627 votes='0/3'

22:00:46 [INFO] event=tracker_transition | from_state='LOCKED' to_state='SEARCH'

22:00:46 [INFO] event=dwell | fd_hz=25527.95 ground_truth_target='Y' psd_peak_to_training_mean_db=56.43 state='SEARCH' vel_m_s=382.919 votes='3/3'

22:00:47 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:47 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:47 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:47 [INFO] event=dwell | fd_hz=25299.07 ground_truth_target='Y' psd_peak_to_training_mean_db=56.77 state='SEARCH' vel_m_s=379.486 votes='3/3'

22:00:48 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:48 [INFO] event=dwell | fd_hz=44799.8 ground_truth_target='Y' psd_peak_to_training_mean_db=44.36 state='SEARCH' vel_m_s=489.178 votes='3/3'

22:00:48 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:49 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:49 [INFO] event=dwell | fd_hz=19882.2 ground_truth_target='N' psd_peak_to_training_mean_db=10.29 state='SEARCH' vel_m_s=405.807 votes='2/3'

22:00:49 [INFO] event=tracker_transition | from_state='SEARCH' to_state='LOCKED'

22:00:49 [INFO] event=dwell | fd_hz=24658.2 ground_truth_target='Y' psd_peak_to_training_mean_db=54.62 state='LOCKED' vel_m_s=389.439 votes='3/3'

22:00:50 [INFO] event=dwell | fd_hz=24658.2 ground_truth_target='N' psd_peak_to_training_mean_db=54.62 state='LOCKED' vel_m_s=389.439 votes='2/3'

22:00:50 [INFO] event=dwell | fd_hz=24658.2 ground_truth_target='N' psd_peak_to_training_mean_db=54.62 state='LOCKED' vel_m_s=389.439 votes='2/3'

22:00:50 [INFO] event=tracker_transition | from_state='LOCKED' to_state='SEARCH'

22:00:50 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:51 [INFO] event=dwell | fd_hz=24978.64 ground_truth_target='Y' psd_peak_to_training_mean_db=54.17 state='SEARCH' vel_m_s=374.68 votes='3/3'

22:00:51 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:51 [INFO] event=dwell | fd_hz=26351.93 ground_truth_target='Y' psd_peak_to_training_mean_db=55.63 state='SEARCH' vel_m_s=382.404 votes='3/3'

22:00:51 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:52 [INFO] event=dwell | fd_hz=48660.28 ground_truth_target='Y' psd_peak_to_training_mean_db=45.08 state='SEARCH' vel_m_s=534.13 votes='3/3'

22:00:52 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:52 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:53 [INFO] event=dwell | fd_hz=0.0 ground_truth_target='N' psd_peak_to_training_mean_db=0.0 state='SEARCH' vel_m_s=0.0 votes='0/3'

22:00:53 [INFO] event=dwell | fd_hz=44952.39 ground_truth_target='Y' psd_peak_to_training_mean_db=44.76 state='SEARCH' vel_m_s=597.974 votes='3/3'

22:00:53 [INFO] event=tracker_transition | from_state='SEARCH' to_state='LOCKED'

22:00:53 [INFO] event=dwell | fd_hz=24658.2 ground_truth_target='Y' psd_peak_to_training_mean_db=56.15 state='LOCKED' vel_m_s=492.804 votes='3/3'

22:00:54 [INFO] event=dwell | fd_hz=24658.2 ground_truth_target='Y' psd_peak_to_training_mean_db=56.15 state='LOCKED' vel_m_s=492.804 votes='3/3'

22:00:54 [INFO] event=dwell | fd_hz=24658.2 ground_truth_target='N' psd_peak_to_training_mean_db=56.15 state='LOCKED' vel_m_s=492.804 votes='0/3'

22:00:54 [INFO] event=tracker_transition | from_state='LOCKED' to_state='SEARCH'


</pre>



| Field                              | Meaning                                                                 |
|------------------------------------|-------------------------------------------------------------------------|
| `ground_truth_target`              | `Y` = a target was actually generated, `N` = noise only                 |
| `state`                            | `SEARCH` (no track) or `LOCKED` (tracking)                              |
| `votes`                            | e.g. `3/3` = all three branches detected a target in that dwell         |
| `fd_hz`                            | Estimated Doppler frequency (positive = approaching in this simulation) |
| `vel_m_s`                          | Radial velocity from Doppler (positive = approaching)                   |
| `psd_peak_to_training_mean_db`     | PSD contrast metric (not real SNR, but useful for visualisation)        |




-- FAQ :: Is the simulation flawless? --

-- No. -- And that’s by design. --

-- I am not a radar engineer – this is a hobby project.

-- The goal was never to model an AN/APG‑77 or any real hardware.


-- There are known limitations: --

-- The heuristic that picks the “best” branch for PSD extraction can sometimes cause sign mismatches (positive Doppler with negative velocity).
-- The PFA self‑test uses a relative tolerance of 0.5 – it’s a coarse sanity check, not a formal validation.
-- No range dimension – Doppler‑only simulation.
-- If you find a bug, feel free to open an issue. But remember: this is a simulation, NOT a certified radar processor.


-- Disclaimer --

-- USE AT YOUR OWN RISK. --

-- This software is provided “as is”, without warranty of any kind. It is intended for educational purposes only. --
-- I does not claim that the simulation reflects the performance of any real‑world radar system. --
-- Do not use this code for any operational or safety‑critical application. --


-- -- -- -- -- -- -- -- -- --

pragma Comment ("Built with Python, NumPy, in VS CODE and too many late nights. – For those who want to see radar DSP in -- ONE Single Python Block of Code. -- ");

M2KR4R
