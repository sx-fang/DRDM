# DRDM

Code repository for the paper "Deep random difference method for high-dimensional quasilinear parabolic partial differential equations" (arXiv:2506.20308). 


## Repository summary

The repository is organized into five subfolders corresponding to the numerical experiments in Subsections 4.1–4.5 (as indicated by their names). Each subfolder is self-contained and can be run independently. To reproduce the results for a given subsection, run `runtask.py` within the corresponding subfolder.

### Note
The five subfolders share a similar structure but differ slightly because the codebase evolved iteratively. The suffix `---m[number]` in subfolder names indicates the implementation version, where larger numbers correspond to more recent versions.

If you plan to adapt the code to your own problem, we recommend starting with `Subsection_4.5`, which contains the most up-to-date implementation.


## Quick start

### Run with the default config

If `taskfiles/` contains no `.ini` files, `runtask.py` will run using `default_config.ini`.

```bash
python runtask.py
```

Outputs are written to the directory specified by `[Environment] output_dir` (default: `./outputs`).

### Run a batch of tasks

Place one or more `*.ini` config files under `taskfiles/` (or generate them via `taskmaker.py`).
Then run:

```bash
python runtask.py
```

The runner iterates through all `taskfiles/*.ini` files in sorted order.

## Module summary

Here is a brief overview of the main modules in the `Subsection_4.5` subfolder (the latest version). Earlier versions follow a similar structure, but some modules may be merged into a single file (e.g., `martnetdf.py` and `savresult.py` in earlier versions contain code that is now split into multiple modules in `Subsection_4.5`).

### Core runner

- `runtask.py`
	- Main entrypoint.
	- Loads configs, instantiates an example problem, builds networks, selects a loss/method, samples training paths, runs training, and saves results.
	- Supports CPU and GPU; uses torch.distributed (DDP) when `world_size > 1`.

- `default_config.ini`
	- Default configuration template.
	- Used when `taskfiles/` has no task configs.

- `taskfiles/`
	- Directory scanned by `runtask.py` for `*.ini` task configs.
	- Empty by default.

- `taskmaker.py`
	- Generates many task `*.ini` files (parameter sweeps) under `taskfiles/`.
	- Encodes common heuristics for max iterations, batch size, network width, learning rates, etc.

### Problem definitions (examples)

- `ex_meta.py`
	- Defines base problem classes and shared utilities.
	- Includes helpers for automatic differentiation (time/space derivatives) and Monte-Carlo evaluation of reference solutions.
	- Provides curve/point generators used for training and evaluation.

- `ex_*.py`
	- Example problem classes (e.g., `ExHjb`, `ExQuasi`) that inherit from the base classes in `ex_meta.py`.

### Neural networks

- `networks.py`
	- Network components used by the solver.
	- Includes time-dependent and time-independent MLPs (`DNNtx`, `DNNx`), Fourier feature inputs, and wrappers for multiscale and variable-scaling networks.
	- In earlier versions, this module was bundled in `martnetdf.py`.
  
### Sampling and training

- `sampling.py`
	- `PathSampler` for generating pilot paths and (optionally) offline components used by different losses.
	- Implements path refreshing across epochs and returns mini-batches to the training loop.
	- In earlier versions, this module was bundled in `martnetdf.py`.

- `loss_meta.py`
	- Abstract loss interface (`LossCollection`) and the generic training loop (`train`).
	- Includes logging helpers and GPU memory recording.
	- In earlier versions, this module was bundled in `martnetdf.py`.

### Loss functions / methods

- `loss_martnet.py`
	- MartNet-family objectives implemented as `LossCollection` subclasses.
	- Includes:
		- `DfSocMartNet`: derivative-free MartNet/**deep random difference method** for HJB equations (*the two methods are equivalent!*).
		- `SocMartNet`: Soc-MartNet for HJB equations.
		- `QuasiMartNet`: Soc-MartNet for quasi-linear parabolic PDEs.
		- `DfQuasiMartNet`: derivative-free MartNet/**deep random difference method** for quasi-linear parabolic PDEs (*the two methods are equivalent!*).
	- In earlier versions, this module was bundled in `martnetdf.py`.
	
- `loss_martnet_strf.py`
	- Strong-form random difference method (RDM) variants: `SocRdmStrForm` and `QuasiRdmStrForm`.
	- Uses `num_rdmsamp` (even) and optional antithetic sampling.
	- Only the latest version in `Subsection_4.5` includes this module.


- `loss_pinn.py`
	- PINN baselines (`QuasiPinn`, `SocPinn`) using strong-form residual minimization via automatic differentiation.
	- Currently does not support DDP.
	- Only the latest version in `Subsection_4.5` includes this module.
  	
### Results, plotting, and post-processing

- `sav_res_via_runtask.py`
	- Utilities to save training logs and summarize repeats (mean/std).
	- Plot helpers for error curves and path-based diagnostics.
	- In earlier versions, this module was bundled in `savresult.py`.

- `sav_res_via_exmeta.py`
	- Utilities to evaluate/plot solutions on curves and to visualize 2D landscapes.
	- Used by `ex_meta.py` and example problems when saving results.
	- In earlier versions, this module was bundled in `savresult.py`.

## Notes

- DDP is used automatically when `[Environment] world_size` is greater than 1 and GPUs are available.
- Some reference-solution evaluations use Monte Carlo and can be expensive (see `nsamp_mc` in example classes).
