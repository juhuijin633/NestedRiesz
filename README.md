# NestedRiesz: Replication Package

Code to reproduce the simulations and applications in the paper.

```
NestedRiesz/
├── environment.yml
├── clean_requirements.txt
├── install_glmnet.sh
├── Simulation_1_TimeTreatment/
├── Simulation_2_DiD/
├── Application_1_Surrogacy/
└── Application_2_DiD/
```

## Setup

**Linux / HPC**

```bash
conda env create -n riesz -f environment.yml
conda activate riesz
```

**macOS / non-Linux**

```bash
conda create -n riesz python=3.10 -y
conda activate riesz
python -m pip install -r clean_requirements.txt
python -m pip install "rpy2==3.6.0"
```

**R packages** (needed for Exercise 1 Bradic / `glmnet`):

```bash
R --no-save -e "install.packages(c('glmnet','itertools','doParallel','RCAL'), repos='https://cloud.r-project.org')"
```

On SLURM, you can instead run `sbatch install_glmnet.sh` (edit modules/partition as needed). Cluster submit scripts use `conda activate riesz`.

---

## Exercise 1: Time-varying treatment simulations

**Folder:** `Simulation_1_TimeTreatment/`

```bash
cd Simulation_1_TimeTreatment
sbatch code/submit.sh
```

This submits 3,000 array jobs (linear + truncated logistic; nonlinear + truncated adversarial; \(N \in \{500,1000,2000\}\); 500 iterations). Each job writes a file under:

```
results/linear/N{N}/…/result_{t}.pt
results/nonlinear/N{N}/…/result_{t}.pt
```

When jobs finish:

```bash
python code/collect_results.py
```

**Output:** `results/summary_tables.csv`, `results/summary_tables.txt`

To run a single iteration locally:

```bash
python code/run_sim.py <t> <N> truncated_logistic 0.1 0.9
python code/run_sim_nonlinear.py <t> <N> truncated_adv 0.3 0.7
```

---

## Exercise 2: DiD simulations

**Folder:** `Simulation_2_DiD/`

**Cluster**

```bash
cd Simulation_2_DiD
sbatch code/submit.sbatch
python code/collect_results.py   # after the array finishes
```

**Local** (all 9 configurations, then summary):

```bash
cd Simulation_2_DiD
python code/RUN.py
```

**Output:** `results/final_N:{N}_{model}_*.pt`, `results/summary.csv`

---

## Exercise 3: Surrogate application

**Folder:** `Application_1_Surrogacy/`
The data we use `data/quarterly.mat` which is converted into `data/river_data.csv`, `data/others_data.csv` using `data/data_manipulation.R` is the unpublished data used from Hotz et al., 2006. 
The data is available upon request from Opportunity Insights, and is the only data needed to run this code. 

```bash
cd Application_1_Surrogacy
python code/RUN.py
```

**Output:** `results/auto_{earn,employ}.csv`, `results/manual_{earn,employ}.pkl`, `results/results_{earn,employ}.csv`, `results/fig_*.png`

---

## Exercise 4: DiD application (minimum wage)

**Folder:** `Application_2_DiD/`

```bash
cd Application_2_DiD
python code/RUN.py
```

**Output:** `results/results_{2004–2007}.csv`, `results/fig_{2004–2007}.png`, `results/dynamic_cache/`

Requires network once to download `minwage_data.csv` (see `code/utils/application.py`).

---

## License

See `LICENSE` (MIT).
