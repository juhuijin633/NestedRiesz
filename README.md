# NestedRiesz: Replication Package

This repository contains the code needed to reproduce the simulation studies and empirical applications in the paper. The package has four exercises: two Monte Carlo studies (run on a SLURM cluster) and two applications (run locally via `code/RUN.py`).

```
NestedRiesz/
├── environment.yml
├── clean_requirements.txt
├── install_glmnet.sh
├── Simulation_1_TimeTreatment/   # Exercise 1
├── Simulation_2_DiD/             # Exercise 2
├── Application_1_Surrogacy/      # Exercise 3
└── Application_2_DiD/            # Exercise 4
```

Pre-computed results are already under each exercise’s `results/` folder.

---

## Setup

All exercises use a conda environment named `riesz` (Python 3.10). Cluster scripts call `conda activate riesz`.

**Linux / HPC**

```bash
conda env create -n riesz -f environment.yml
conda activate riesz
```

**macOS / non-Linux** (`environment.yml` is Linux/CUDA-oriented):

```bash
conda create -n riesz python=3.10 -y
conda activate riesz
python -m pip install -r clean_requirements.txt
python -m pip install "rpy2==3.6.0"   # Exercise 1 Bradic only
```

**R packages** (Exercise 1 Bradic baseline, via `rpy2`):

```bash
R --no-save -e "install.packages(c('glmnet','itertools','doParallel','RCAL'), repos='https://cloud.r-project.org')"
```

On SLURM you can instead run `sbatch install_glmnet.sh` (edit partition/modules for your cluster). Set `R_LIBS_USER` if needed; `Simulation_1_TimeTreatment/code/submit.sh` already exports it.

### Dependencies

Pinned versions are in `environment.yml` and `clean_requirements.txt`. Key packages:

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.7.0 |
| econml | 0.15.1 |
| scikit-learn | 1.5.2 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scipy | 1.15.3 |
| statsmodels | 0.14.4 |
| rpy2 | 3.6.0 |
| R (external) | ≥ 4.2 |
| glmnet (R) | recent |

Cluster logs go under `Simulation_1_TimeTreatment/logs/` and `Simulation_2_DiD/logs/` (relative to the directory you submit from).

---

## Exercise 1: Time-varying treatment simulations

**Location:** `Simulation_1_TimeTreatment/`

This exercise evaluates estimators of \(E[Y(1,1)]\) under time-varying treatment. It crosses a linear and a nonlinear outcome DGP with propensity score specifications (logistic / truncated-logistic / truncated-adversarial), with 500 Monte Carlo iterations at each of \(N = 500, 1000, 2000\). 

Estimators include an oracle, Bradic et al.'s original estimator, and nested Riesz estimators with LASSO, random forests (RF), and a neural network (Net), including (i) Manual Lasso/RF/NN-Ridge estimators using the sequentially doubly robust representation of Bradic, et al. (ii) Manual Lasso-Lasso/RF-RF/NN-NN estimators, and (iii) Auto-Lasso/RF/NN estimators all implemented using implemented using the NNPIV package of Meza and Singh (2021). 

Jobs were run on a SLURM cluster. The main array requests 4 CPUs, 16 GB RAM, and up to 2 hours of wall time per task, with Python 3.10 and R 4.4 modules loaded (edit `code/submit.sh` for your site).

### Submit (main paper configurations)

2 DGPs × 3 sample sizes × 500 iterations = **3,000** jobs:

- Linear outcome + truncated logistic propensity on \([0.1, 0.9]\)
- Nonlinear outcome + truncated adversarial propensity on \([0.3, 0.7]\)

```bash
cd Simulation_1_TimeTreatment
mkdir -p logs
sbatch code/submit.sh
```

Each array task maps to a DGP, \(N\), and iteration index, then runs either `code/run_sim.py` (linear) or `code/run_sim_nonlinear.py` (nonlinear):

```bash
python code/run_sim.py <iteration> <N> <func_name> [lower] [upper]
# example:
python code/run_sim.py 42 1000 truncated_logistic 0.1 0.9
python code/run_sim_nonlinear.py 7 500 truncated_adv 0.3 0.7
```

### Outputs

Per-iteration results are PyTorch files:

```
results/linear/N{N}/{func_or_func_lower_upper}/result_{t}.pt
results/nonlinear/N{N}/{func_lower_upper}/result_{t}.pt
```

After the array finishes, aggregate bias / RMSE / coverage / interval length:

```bash
python code/collect_results.py
```

This writes `results/summary_tables.csv` and `results/summary_tables.txt`.

---

## Exercise 2: DiD simulations

**Location:** `Simulation_2_DiD/`

This exercise evaluates difference-in-differences estimators in a panel setting where covariates evolve between periods. The DGP has a binary treatment, pre- and post-period covariates, and a nonlinearly determined outcome. Three propensity specifications are used (logistic, truncated-logistic, truncated-adversarial), with **500** Monte Carlo iterations at each of \(N = 500, 1000, 2000\). Estimators compared are OLS, a linear DiD estimator, and nested Riesz LASSO / RF / Net.

### Cluster

The job array has **9** tasks (3 propensity models × 3 sample sizes). Each task requests 1 CPU, 8 GB RAM, and up to 4 days of wall time (see `code/submit.sbatch`; edit partition/modules/mail as needed).

```bash
cd Simulation_2_DiD
mkdir -p logs results
sbatch code/submit.sbatch
```

Each task runs `code/RUN.py` with `SLURM_ARRAY_TASK_ID` selecting the \((N,\) propensity\()\) configuration. Results are saved as `.pt` files in `results/`. When the array finishes:

```bash
python code/collect_results.py
```

This prints a summary table and writes `results/summary.csv`.

### Local

To run all nine configurations sequentially and then collect:

```bash
cd Simulation_2_DiD
python code/RUN.py
```

### Outputs

```
results/final_N:{N}_{model}_pred_theta.pt
results/final_N:{N}_{model}_pred_sig.pt
results/final_N:{N}_{model}_ATT_calculations.pt
results/summary.csv
```

---

## Exercise 3: Surrogate application

**Location:** `Application_1_Surrogacy/`

This exercise applies the surrogate estimator to the effect of a job training program on quarterly earnings and employment. It combines an experimental sample (benchmark) with an observational sample and estimates average treatment effects using intermediate-outcome surrogates. Estimators based on LASSO, random forests, and neural networks are compared to a difference-in-means benchmark from the experimental data and related baselines.

The source file `data/quarterly.mat` is converted to `data/river_data.csv` and `data/others_data.csv` by `data/data_manipulation.R`. This is unpublished data from Hotz et al. (2006), available upon request from Opportunity Insights, and is the only external data needed for this exercise.

```bash
cd Application_1_Surrogacy
python code/RUN.py
```

`code/RUN.py` runs, in order, `1. calc_auto.py`, `2. calc_manual.py`, and `3. make_figs_tables.py` for earnings and employment.

### Outputs

```
results/auto_{earn,employ}.csv
results/manual_{earn,employ}.pkl
results/results_{earn,employ}.csv
results/fig_*.png
```

---

## Exercise 4: DiD application (minimum wage)

**Location:** `Application_2_DiD/`

This exercise applies the dynamic DiD estimator to a county-level panel to estimate the effect of minimum wage increases on teenage employment. The estimand is the ATT for counties that raised their minimum wage, for each post-treatment year from **2004 through 2007**. Estimators include a standard OLS first-difference regression, the Caetano linear estimator, and nested Riesz LASSO / RF / Net, with static benchmarks from Chernozhukov et al. and Sant’Anna–Zhao (see `README.txt` and hard-coded values in `code/RUN.py`).

```bash
cd Application_2_DiD
python code/RUN.py
```

`code/RUN.py` loads data, estimates (or loads cached) dynamic results, writes tables, and saves figures. Each year takes several minutes because of cross-fitting; a GPU is not required. Cached dynamic fits live under `results/dynamic_cache/`. Delete those files (or the year CSVs) to recompute.

The panel is downloaded by `code/utils/application.py` from

`https://raw.githubusercontent.com/CausalAIBook/MetricsMLNotebooks/main/data/minwage_data.csv`

so network access is required unless you point the loader at a local copy.

### Outputs

```
results/results_{2004,2005,2006,2007}.csv
results/fig_{2004,2005,2006,2007}.png
results/dynamic_cache/dynamic_{year}.csv
```

---

## License

See `LICENSE` (MIT).
