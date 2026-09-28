# BlueROV2 Physical System Identification and Model Comparison

This repository contains the ROS 2 based physical-data collection, offline preprocessing, system-identification, model-comparison, per-DOF model-selection, disturbance/tether analysis, and frequency-response analysis workflow developed for the BlueROV2 Heavy underwater vehicle.

The purpose of the repository is to turn controlled physical force/moment excitation experiments into validated mathematical vehicle-dynamics models that can later be used for velocity control, station keeping, disturbance rejection, MPC/NMPC, and related underwater-vehicle research.

> **Important:** This README documents the methodology and software architecture. It intentionally does **not** report the final experimental model-selection results. Results are stored under `results/` and should be evaluated separately from this documentation.

---

## 1. Scope

The workflow supports:

- Physical excitation of the BlueROV2 through ROS 2.
- Force/moment command generation through `/rov/control_force`.
- Thruster allocation through the existing allocator.
- MAVLink/vehicle interfacing through the bridge.
- Recording of vehicle state and commanded generalized forces/moments.
- 50 Hz analysis-data preparation.
- Six-degree-of-freedom (6-DOF) system identification.
- Comparison of multiple linear, nonlinear, sparse, state-space, autoregressive, kernel, and Koopman-style models.
- Tether/disturbance calibration for the surge input.
- Global model comparison.
- Per-DOF model comparison and selection.
- Frequency-response and cross-coupling analysis.
- Automatic generation of tables, JSON reports, and figures.

The six vehicle velocity states are:

$$
\nu = [u, v, w, p, q, r]^T
$$

where `u, v, w` are body-frame translational velocities and `p, q, r` are body-frame angular rates.

The generalized control input is:

$$
\tau = [F_x, F_y, F_z, M_x, M_y, M_z]^T.
$$

The central identification objective is to approximate the measured vehicle dynamics in the form

$$
\dot{\nu} = f(\nu,\tau)
$$

or, for discrete-time models,

$$
\nu_{k+1} = F(\nu_k,\tau_k,\ldots).
$$

---

## 2. Repository concept and data flow

The project has two related parts:

1. **ROS 2 package source** under `src/rov_system_identification/`.
2. **Analysis/data root** under `ROV_system_idetification/`.

The analysis root contains the collected data, metadata, processed data, fitted-model outputs, reports, tables, and plots.

Typical physical-data flow:

```text
                    PHYSICAL BLUEROV2
                           |
                           v
                  vehicle state feedback
                           |
                           v
                     /rov/state
                           |
             +-------------+-------------+
             |                           |
             |      Data collector       |
             |                           |
             +-------------+-------------+
                           |
                     /rov/control_force
                           |
                           v
                    ROV allocator
                           |
                    /rov/thruster_pwm
                           |
                           v
                    ROV bridge
                           |
                           v
                   MAVLink / BlueOS
                           |
                           v
                       Thrusters
```

For physical experiments, the deployed **allocator and bridge must be running**. The collector is designed to command generalized forces/moments, while the allocator converts them to individual thruster commands and the bridge transmits those commands to the vehicle. With `require_thruster_pwm: true`, the collector also checks that the PWM path is alive and fresh.

Do not operate a physical ROV from this repository without appropriate vehicle, tank, tether, personnel, software, and emergency-stop safety procedures.

---

## 3. Physical ROS 2 interface

The identification workflow uses the project messages and topics already deployed in the vehicle-control stack.

### State

`/rov/state`

Message: `rov_msgs/msg/VehicleState`

Contains:

```text
x, y, z
vx, vy, vz
roll, pitch, yaw
roll_rate, pitch_rate, yaw_rate
```

The identification pipeline uses the body-frame velocity/angular-rate quantities represented by:

```text
u, v, w, p, q, r
```

### Control input

`/rov/control_force`

Message: `rov_msgs/msg/ControlForce`

Fields:

```text
fx, fy, fz
mx, my, mz
```

### Thruster PWM

`/rov/thruster_pwm`

Message: `rov_msgs/msg/ThrusterPWM`

Fields:

```text
ch1 ... ch8
```

The allocator converts the generalized force/moment request into the eight-channel thruster commands.

### Physical experiment requirement

Before running `collect_physical_data.py` on the real BlueROV2, the normal position/velocity/adaptive controller that would otherwise fight the identification excitation should be disabled. The intended identification chain is:

```text
collector -> allocator -> bridge -> BlueROV2
                                |
                                v
                             /rov/state
```

The identification excitation itself provides the commanded input.

---

## 4. System-identification experiment design

The collector is configured to run:

### 4.1 Individual excitation

Each generalized input axis can be excited independently:

```text
Fx, Fy, Fz, Mx, My, Mz
```

with positive and negative amplitudes.

### 4.2 Cross excitation

Cross excitation is currently configured for selected pairs, with the main physical campaign using planar `Fx/Fy` combinations. Cross excitation is useful for revealing dynamic coupling that cannot be identified reliably from independent one-axis tests alone.

### 4.3 Broadband excitation

Three broadband families are supported:

- PRBS
- Multisine
- Chirp

Current configuration supports deliberate excitation in approximately the `0.05-1 Hz` range through the chirp and selected multisine frequencies. Broadband excitation is useful for estimating dynamics over a frequency range rather than only at a few step/pulse conditions.

### 4.4 Amplitude split

The standard identification split in the current configuration is:

```text
Training     : amplitudes <= 80 N
Validation   : 90 N
Test         : 100 N
```

The values are configuration parameters and should be changed if a new experimental campaign uses a different safe range.

The test split is intended to represent an unseen excitation level for model evaluation.

---

## 5. Data rate and 50 Hz analysis dataset

The physical collector is configured for a 50 Hz analysis rate. The original logging path can produce much faster repeated state rows because it may record the latest ROS message repeatedly between actual state updates.

For identification, the physical raw log is therefore converted to a regular 50 Hz analysis timeline. The project has used a 20 ms grid and avoids interpolation across experiment/segment boundaries.

The analysis dataset should contain approximately:

```text
20 ms sample interval
50 Hz nominal frequency
```

The original high-rate raw log should be kept separately when full experimental provenance is required, but it should not be accidentally placed in the main `data/raw/` directory used by the offline pipeline if the pipeline is intended to analyse only the 50 Hz dataset.

---

# 6. Mathematical models

The pipeline evaluates multiple candidate model classes. They are deliberately diverse because no single mathematical structure should be assumed to be correct before examining the experimental data.

## 6.1 Direct gain model

The direct-gain model represents a simple input-to-response relationship.

A conceptual continuous-time form is:

$$
\dot{\nu} = B\tau.
$$

Here, `B` is a learned input-effectiveness matrix.

### Why use it?

It is a baseline model. It is simple, interpretable, inexpensive, and useful for determining how far a simple static mapping can go.

### Advantages

- Very low complexity.
- Easy to interpret.
- Useful as a baseline.
- Computationally cheap.

### Limitations

- No explicit temporal memory.
- Limited ability to represent inertia and transient dynamics.
- Limited representation of dynamic actuator/hydrodynamic effects.

---

## 6.2 Coupled linear Ridge model

The coupled linear model extends the direct mapping by using the vehicle state and input together.

A generic form is:

$$
\dot{\nu} = A\nu + B\tau + c.
$$

The coefficients are estimated using Ridge regression:

$$
\hat{\Theta}
=
\arg\min_{\Theta}
\left\|Y-X\Theta\right\|_2^2
+
\alpha\left\|\Theta\right\|_2^2.
$$

Ridge regularization is useful because the experimental state/input regressors can be correlated.

### Why use it?

It provides a compact linear approximation while explicitly modelling state dependence and cross-coupling.

---

## 6.3 Nonlinear Ridge model

The nonlinear Ridge model augments the regression feature space with nonlinear terms. Depending on the implementation/configuration, these may include terms such as:

$$
\nu_i,
\tau_j,
\nu_i^2,
\tau_j^2,
\nu_i\tau_j,
\ldots
$$

The nonlinear feature vector is then fitted with regularized least squares.

### Why use it?

Underwater vehicle dynamics contain nonlinear phenomena, particularly velocity-dependent hydrodynamic damping and drag. A nonlinear regression model gives a middle ground between a simple linear model and a fully nonlinear state-space representation.

### Limitations

Increasing the polynomial/nonlinear feature space can increase parameter count and can reduce extrapolation reliability outside the training region.

---

## 6.4 Grey-box quadratic-drag model

The grey-box approach embeds physically motivated quadratic damping/drag terms into an otherwise parameterized model.

A representative form is:

$$
\dot{\nu}
=
M^{-1}
\left(
\tau
-
D_1\nu
-
D_2|\nu|\odot\nu
+\text{other model terms}
\right).
$$

Here, the linear damping term and quadratic velocity-dependent term provide a simple physical approximation of hydrodynamic resistance.

### Why use it?

A grey-box model combines data-driven parameter estimation with known physical structure. It is especially relevant for underwater vehicles because quadratic drag is a common first-order approximation of hydrodynamic resistance.

### Advantages

- Physically interpretable.
- Lower dimensional than a large generic nonlinear model.
- Can provide a useful control-oriented approximation.

---

## 6.5 SINDy

SINDy (Sparse Identification of Nonlinear Dynamics) assumes that the real dynamics can be represented by a relatively small number of active terms from a candidate library.

A generic formulation is:

$$
\dot{\nu}=\Theta(\nu,\tau)\Xi
$$

where:

- `Theta` is a library of candidate functions.
- `Xi` contains the coefficients.
- Sparse regression tries to keep only important terms.

### Why use it?

SINDy can produce an interpretable nonlinear equation instead of only a black-box predictor.

This is attractive when the goal is not only prediction but also discovering which physical/state/input interactions appear important in the measured data.

### Limitations

SINDy is sensitive to derivative quality, library choice, noise, scaling, and threshold selection.

---

## 6.6 Weak SINDy

Weak SINDy reformulates sparse identification using weak/integral forms so that direct numerical differentiation is less dominant.

The general motivation is:

$$
\text{avoid relying entirely on noisy pointwise }\dot{\nu}
$$

by projecting the dynamics over local test functions or windows.

### Why use it?

It is particularly relevant to physical data where differentiation amplifies sensor noise.

### Limitations

Results depend on the weak-form/test-function configuration and numerical implementation.

---

## 6.7 ARX model

ARX = AutoRegressive with eXogenous inputs.

The current vehicle ARX architecture uses two history steps (`na=2`, `nb=2`) and is MIMO.

A conceptual equation for the six-DOF velocity vector is:

$$
\nu_k
=
A_1\nu_{k-1}
+
A_2\nu_{k-2}
+
B_0\tau_k
+
B_1\tau_{k-1}
+
\epsilon_k.
$$

where:

$$
A_1,A_2,B_0,B_1\in\mathbb{R}^{6\times6}.
$$

Therefore, the model can represent:

- velocity memory,
- input memory,
- cross-coupling between the six DOFs,
- dynamic rather than purely static input/output relationships.

With 2 velocity-history blocks and 2 input-history blocks there are 24 regression features per output and 6 outputs:

$$
24\times6=144
$$

model coefficients before considering any additional implementation metadata.

The implementation uses Ridge regularization for coefficient estimation.

### Why use it?

The ARX model is attractive for control because it is:

- discrete-time,
- MIMO,
- compact,
- data-driven,
- able to represent temporal dynamics,
- straightforward to convert into an augmented state-space representation.

This makes ARX particularly relevant for future MPC/NMPC and model-based disturbance rejection.

---

## 6.8 Tether-corrected ARX

The tether-aware models do not change the underlying ARX structure. Instead, the measured surge command is modified using an experimentally estimated equivalent input disturbance:

$$
F_{x,\mathrm{eff}}
=
F_{x,\mathrm{cmd}}
+
d_{\mathrm{equivalent}}.
$$

The calibration is intentionally estimated from the low-force training range rather than from the full force range, because the physical tether bias can vary with amplitude and vehicle position.

The correction should be interpreted as an **equivalent generalized input disturbance** rather than a direct measurement of the physical tether tension.

The same wrapper concept is applied to the other candidate model families for fair comparison.

---

## 6.9 Subspace / MOESP model

Subspace identification methods estimate state-space models directly from measured sequences.

A generic discrete-time realization is:

$$
x_{k+1}=Ax_k+B\tau_k
$$

$$
\nu_k=Cx_k+D\tau_k.
$$

MOESP-style methods exploit block Hankel matrices of measured input/output data to recover a low-dimensional state realization without explicitly selecting nonlinear basis functions.

### Why use it?

Subspace identification is a standard control-oriented approach for data-driven linear dynamical models and is useful as a strong state-space benchmark against ARX.

---

## 6.10 Koopman / DMDc model

Dynamic Mode Decomposition with control (DMDc) and Koopman-inspired modelling approximate nonlinear dynamics in a lifted coordinate space.

A simplified lifted form is:

$$
\phi(\nu_{k+1})
\approx
A_K\phi(\nu_k)+B_K\tau_k
$$

where `phi` is a feature/lifting map.

### Why use it?

It provides a way to represent nonlinear dynamics through a linear evolution model in an augmented space and can be useful for prediction and model-based control.

### Limitations

The quality of the model depends heavily on the lifting/features and the stability/generalization of the learned operator.

---

## 6.11 Gaussian Process model

The Gaussian Process model represents a distribution over functions:

$$
 f(x)\sim\mathcal{GP}(m(x),k(x,x')).
$$

The model can provide both a predicted mean and a measure of predictive uncertainty.

### Why use it?

Gaussian Processes are useful when uncertainty quantification is important, particularly for conservative model-based control or active-learning workflows.

### Limitations

Full GP complexity grows with training-set size, so the implementation limits the number of training samples used.

---

# 7. Disturbance/tether analysis

The X-axis tether analysis estimates whether an apparent surge bias is consistent with a fixed disturbance or whether the bias changes with excitation amplitude and/or starting position.

The early transient is used to estimate an input-response relationship over the configured early-time window.

For a simple local relation:

$$
 a_x = gF_x+b
$$

it can be rewritten as:

$$
 a_x=g(F_x+d)
$$

with

$$
 d=\frac{b}{g}.
$$

Therefore:

$$
F_{x,\mathrm{eff}}=F_{x,\mathrm{cmd}}+d.
$$

This `d` is an **equivalent input disturbance**. It must not automatically be reported as physical tether tension.

The low-force calibration range is configurable in `identification.yaml`.

---

# 8. Global model comparison

The global comparison treats each candidate model as a 6-DOF model and evaluates all six outputs together.

The comparison includes:

- one-step RMSE,
- one-step NRMSE,
- one-step $R^2$,
- 1 s rollout RMSE,
- 3 s rollout RMSE,
- 5 s rollout RMSE,
- effective rollout duration,
- model fit time,
- number of parameters,
- rollout coverage.

## 8.1 RMSE

For a prediction error $e_i$:

$$
RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}.
$$

## 8.2 NRMSE

NRMSE normalizes the RMSE so that errors can be compared across outputs with different scales.

The implementation uses its common normalized form in the project utilities.

## 8.3 R-squared

$$
R^2
=
1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}.
$$

An $R^2$ close to 1 indicates that the predictions explain most of the measured variance. A negative $R^2$ means the model is worse than using the test-set mean as a predictor under that metric.

## 8.4 Rollout error

One-step prediction asks:

$$
\hat{\nu}_{k+1}=F(\nu_k,\tau_k).
$$

Rollout prediction repeatedly feeds model-predicted states back into the model:

$$
\hat{\nu}_{k+2}=F(\hat{\nu}_{k+1},\tau_{k+1}),
$$

and so on.

Rollout tests are therefore substantially more demanding for dynamic models.

## 8.5 Current composite screening score

The comparison code uses a weighted screening score that emphasizes longer-horizon prediction:

```text
5 s rollout RMSE       0.45
3 s rollout RMSE       0.25
1 s rollout RMSE       0.15
1-step NRMSE           0.10
runtime                0.02
complexity             0.03
```

Missing rollout metrics are penalized rather than rewarded, and rollout coverage is included.

This score is an **engineering screening metric**, not a universal scientific metric. For publications, report the underlying metrics as well as the composite score and explain the weighting.

---

# 9. Per-DOF versus global comparison

These are complementary analyses.

## Global comparison

A model is evaluated as a complete 6-output MIMO model:

$$
\nu=[u,v,w,p,q,r]^T.
$$

The global metrics aggregate performance across the six outputs.

The question is:

> How well does this candidate model represent the vehicle as a whole?

## Per-DOF comparison

The same candidate model outputs are evaluated separately for each of:

```text
u  surge velocity
v  sway velocity
w  heave velocity
p  roll rate
q  pitch rate
r  yaw rate
```

The per-DOF selector chooses the candidate giving the best validation performance for each output and then evaluates the selected candidate on the unseen test data.

The question is:

> Which candidate model predicts this particular DOF best?

Per-DOF analysis is therefore primarily a **model-comparison/model-selection procedure**, not a second independent physical system-identification campaign.

A per-DOF result can theoretically produce a mixed selection, for example:

```text
u -> model A
v -> model A
w -> model B
p -> model A
q -> model C
r -> model A
```

whereas the global selection chooses one model structure for the complete vehicle.

---

# 10. Repository structure

A typical project structure is:

```text
ROV_system_idetification/
|
+-- config/
|   +-- identification.yaml
|
+-- data/
|   +-- raw/
|   +-- raw_original/
|
+-- metadata/
|
+-- results/
|   +-- models/
|   +-- tables/
|   +-- plots/
|   +-- reports/
|
+-- README.md
```

The ROS 2 package source is under the ROS 2 workspace:

```text
~/rov_system_id_ws/src/rov_system_identification/
|
+-- config/
|   +-- identification.yaml
|
+-- config_identification.yaml
|
+-- rov_system_identification/
    +-- __init__.py
    +-- common.py
    +-- models.py
    +-- disturbance_analysis.py
    +-- collect_physical_data.py
    +-- run_identification.py
    +-- compare_models.py
    +-- per_dof_selection.py
    +-- frequency_analysis.py
    +-- plot_results.py
    +-- run_full_pipeline.py
```

If additional helper files are added later, they should be documented in this section and their role added to the pipeline description.

---

# 11. Explanation of the system-identification Python files

## `common.py`

Shared infrastructure used throughout the package.

Typical responsibilities include:

- configuration loading,
- directory creation,
- dataset loading,
- protocol/split assignment,
- DOF definitions,
- filtering and differentiation utilities,
- normalized-error calculations,
- R² calculations,
- JSON/report serialization.

Most pipeline scripts should reuse these functions instead of implementing their own incompatible preprocessing.

## `models.py`

Contains the candidate model implementations and common model interfaces.

This is the central mathematical-model file. It contains the model classes/factory used by the identification pipeline and the tether-correction wrapper.

It should be changed carefully because a modification here affects identification, evaluation, and model serialization.

## `collect_physical_data.py`

ROS 2 physical experiment collector.

It generates the configured individual, cross, and broadband excitation sequences and records the vehicle response.

It is the main entry point for collecting new physical datasets.

For real-vehicle collection:

- verify safe amplitudes first,
- ensure the identification controller architecture is correct,
- ensure allocator and bridge are active,
- verify `/rov/state` is healthy,
- verify `/rov/thruster_pwm` is fresh when required,
- keep an emergency neutral/stop procedure available.

## `disturbance_analysis.py`

Analyses the surge/tether disturbance behaviour and estimates the equivalent input disturbance used by tether-corrected model variants.

It also produces tether-related tables, plots, and reports.

## `run_identification.py`

Core offline model-identification/evaluation script.

Responsibilities include:

1. loading/preparing the experimental data,
2. splitting training/validation/test data,
3. fitting enabled models,
4. evaluating one-step prediction,
5. evaluating multi-step rollout,
6. applying tether-aware corrections where requested,
7. saving fitted models and model-level metrics.

## `compare_models.py`

Reads the model-evaluation results and produces the global model comparison.

It handles:

- finite-metric checks,
- missing-metric penalties,
- rollout coverage,
- complexity/runtime terms,
- composite scoring,
- ranked CSV output,
- model-selection report,
- ranking plot.

## `per_dof_selection.py`

Compares candidate model predictions independently for the six DOFs.

It uses validation data for selection and held-out test data for the final per-DOF evaluation.

Outputs include per-DOF validation scores, selected model names, test errors, R² values, and rollout metrics.

## `frequency_analysis.py`

Computes frequency-response information from the available excitation data and writes the FRF table and associated plots.

It is useful for identifying:

- dominant dynamic bandwidth,
- phase relationships,
- input/output frequency dependence,
- possible resonant/slow/fast behaviour,
- cross-axis transfer characteristics.

## `plot_results.py`

Generates the standard analysis plots from the stored experiment/evaluation outputs.

Plots are written to the configured `results/plots/` directory.

## `run_full_pipeline.py`

Orchestration entry point that runs the main offline workflow in sequence.

It is intended to make a reproducible complete identification run from a prepared dataset.

Because some post-processing tools may also be called independently, individual scripts can be rerun without repeating the complete identification stage.

## `__init__.py`

Marks the Python source directory as the ROS 2 Python package and provides the package namespace.

---

# 12. Configuration files

The primary configuration file is:

```text
ROV_system_idetification/config/identification.yaml
```

The ROS 2 package currently also carries configuration copies under:

```text
src/rov_system_identification/config/identification.yaml
src/rov_system_identification/config_identification.yaml
```

Keep these copies synchronized when configuration changes are intentionally part of the repository workflow. When executing the offline pipeline with:

```bash
--root ~/rov_system_id_ws/ROV_system_idetification
```

the analysis-root configuration is the one that should be checked first.

## Collector settings

Important entries include:

```yaml
collector:
  sample_rate_hz: 50.0
  pulse_seconds: 5.0
  neutral_seconds: 5.0
  center_settle_seconds: 5.0
  amplitudes: [10,20,30,40,50,60,70,80,90,100]
  individual: true
  cross: true
  broadband: true
```

Other collector settings define the broadband type, frequencies, output directory, metadata directory, PWM requirement, and PWM freshness timeout.

### Change the raw-data output location

Change:

```yaml
collector:
  output_dir: "~/rov_system_id_ws/ROV_system_idetification/data/raw"
```

### Change the metadata location

Change:

```yaml
collector:
  metadata_dir: "~/rov_system_id_ws/ROV_system_idetification/metadata"
```

## Analysis settings

Important analysis entries include:

```yaml
analysis:
  train_max_amplitude: 80.0
  validation_amplitude: 90.0
  test_amplitude: 100.0
  rollout_seconds: [1.0, 3.0, 5.0]
  arx_na: 2
  arx_nb: 2
```

### Enable/disable models

The `enable_models` list controls which model classes are fitted.

The current configuration supports the original and tether-corrected model variants.

To add/remove a candidate model from a future study, edit this list rather than hard-coding a second model list in an analysis script.

### Tether calibration settings

Relevant entries are:

```yaml
analysis:
  tether_early_start_s: 0.05
  tether_early_end_s: 0.60
  tether_min_gain_mps2_per_N: 1.0e-5
  tether_calibration_min_amplitude_N: 10.0
  tether_calibration_max_amplitude_N: 40.0
```

These control the analysis-time disturbance calibration. They do not hard-code a tether force into the model.

---

# 13. Output locations

All offline results are stored beneath:

```text
ROV_system_idetification/results/
```

## `results/models/`

Serialized fitted models.

Use this directory when a future controller needs to load a previously identified model rather than re-fit from raw data.

## `results/tables/`

CSV tables containing machine-readable numerical results.

Important tables include:

```text
model_comparison_raw.csv
model_comparison_ranked.csv
per_dof_validation_scores.csv
prepared_dataset_with_splits.csv
frequency_response.csv
directional_coupling.csv
direct_pulse_gains.csv
fx_early_transient_slopes.csv
fx_tether_bias_by_amplitude.csv
```

## `results/plots/`

Figures generated by the analysis pipeline, including model ranking, rollout comparisons, per-DOF error plots, frequency-response plots, coupling heatmaps, and tether plots.

## `results/reports/`

JSON reports containing structured experiment/model information.

Important reports include:

```text
identification_run.json
identification_disturbance_calibration.json
fx_tether_bias_report.json
linear_model_matrices_and_decoupling.json
model_selection_report.json
per_dof_model_selection.json
```

---

# 14. Typical offline workflow

After the physical data have been collected and converted to the intended 50 Hz dataset:

```bash
cd ~/rov_system_id_ws
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
```

## Full pipeline

```bash
ros2 run rov_system_identification run_full_pipeline \
  --root ~/rov_system_id_ws/ROV_system_idetification
```

If a pipeline-stage argument changes, confirm the command-line interface with:

```bash
ros2 run rov_system_identification run_full_pipeline -h
```

## Global comparison only

```bash
ros2 run rov_system_identification compare_models \
  --root ~/rov_system_id_ws/ROV_system_idetification
```

## Per-DOF comparison

```bash
ros2 run rov_system_identification per_dof_selection \
  --root ~/rov_system_id_ws/ROV_system_idetification \
  --config ~/rov_system_id_ws/ROV_system_idetification/config/identification.yaml
```

## Frequency analysis

```bash
ros2 run rov_system_identification frequency_analysis \
  --root ~/rov_system_id_ws/ROV_system_idetification
```

## Plot generation

```bash
ros2 run rov_system_identification plot_results \
  --root ~/rov_system_id_ws/ROV_system_idetification
```

Post-processing scripts can be run independently when their input results already exist. This is useful when a comparison script needs correction without repeating an expensive identification run.

---

# 15. Future use with the BlueROV2

The intended future use is to convert the experimentally identified dynamics into a control-oriented model.

A likely model structure for model-based control is:

$$
x_{k+1}=Ax_k+B\tau_k$$

where the ARX model can be converted into an augmented state-space realization.

Potential future applications include:

- velocity control,
- station keeping,
- disturbance rejection,
- current/wave compensation,
- MPC,
- NMPC,
- model-based thruster allocation,
- vehicle-manipulator coordination,
- surface-cleaning motion planning.

For station keeping and cleaning, model validation should eventually include disturbances that are not represented by the identification campaign alone, such as currents, waves, tether effects, or manipulator-induced loads.

A model that predicts the recorded identification dataset well is not automatically guaranteed to provide good closed-loop control performance. Closed-loop experimental validation is therefore required before using the identified model for aggressive control.

---

# 16. How to add a new identification campaign

1. Confirm vehicle safety and verify the maximum safe force/moment amplitudes.
2. Verify the allocator and bridge chain.
3. Confirm `/rov/state` is valid.
4. Confirm `/rov/thruster_pwm` is being refreshed when `require_thruster_pwm: true`.
5. Update `identification.yaml` with the new amplitudes/protocols if required.
6. Collect the physical data.
7. Keep the original raw log in a separate archival location.
8. Create the intended 50 Hz analysis dataset.
9. Ensure only the intended analysis dataset is in `data/raw/`.
10. Run the identification pipeline.
11. Run global comparison.
12. Run per-DOF comparison.
13. Run frequency-response/coupling analysis.
14. Inspect all tables/reports/plots before using the model for control.

---

# 17. Reproducibility recommendations

For every identification campaign, preserve:

- the original raw data,
- the 50 Hz analysis dataset,
- the exact configuration YAML,
- metadata describing the physical experiment,
- the fitted models,
- result tables,
- JSON reports,
- generated plots,
- the Git commit associated with the run.

Do not overwrite an experimental campaign without preserving its data and configuration first.

A useful naming convention is:

```text
rov_identification_YYYYMMDD_HHMMSS.csv
```

for raw/analysis files and matching metadata/report timestamps for provenance.

---

# 18. GitHub backup: important large-file note

This repository intentionally does **not** use a `.gitignore`, because the project owner wants the experimental data and generated outputs backed up.

However, GitHub's normal Git object limit is 100 MiB per file. Files larger than that must use Git Large File Storage (Git LFS). GitHub currently documents a 2 GB maximum LFS file size for GitHub Free and Pro, with larger limits on some paid organization plans. GitHub also recommends keeping repositories small; repositories larger than several GB can become difficult to clone/manage.

This matters for the original high-rate raw CSV. If the raw CSV is around 8.9 GB, it **cannot be uploaded as one Git/LFS file**. It must be split into multiple chunks below your LFS per-file limit. For a GitHub Free/Pro account, using approximately 1.8 GB chunks gives additional safety margin.

The 50 Hz dataset and other result files should also be checked for the 100 MiB regular-Git threshold. Files larger than 100 MiB should be tracked with LFS.

GitHub documentation:

- Repository limits: https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits
- Large files: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Git LFS: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

---

# 19. License and citation

Add the project-specific license and citation information here when the repository is made public.

If the repository is used in a publication, cite the associated system-identification methodology, physical dataset, and software repository according to the project's publication requirements.

---

# 20. Summary

The project provides a reproducible pipeline from physical BlueROV2 excitation to validated 6-DOF data-driven vehicle models:

```text
Physical ROV
    |
    v
Controlled generalized-force excitation
    |
    v
50 Hz analysis dataset
    |
    v
Preprocessing + differentiation
    |
    v
Multiple candidate models
    |
    +-----------------------+
    |                       |
    v                       v
Global comparison       Per-DOF comparison
    |                       |
    +-----------+-----------+
                |
                v
       Validation/Test analysis
                |
                v
        Control-oriented model
                |
                v
      Controller / MPC / NMPC
                |
                v
       Physical closed-loop test
```

The central design principle is to **identify multiple plausible models, evaluate them on held-out data, report the underlying metrics transparently, and only then use the selected dynamics model for control design**.
