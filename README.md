# Saber
This is the replication package for the ASE 2025 paper "Adaptive Request Scheduling for CodeLLM Serving with SLA Guarantees".

## Installation
We recommend to use Python 3.10.16.
Install the required packages with:
```
pip install -r requirements.txt
```

## Quick Start
### Baseline Experiments
Run baseline experiments using static batch size configurations (different max_num_seqs):
```
python run_baseline.py
```

### Run Saber
First, start the vLLM server:
```
vllm serve [path to model] --port 8000 
```
Then run SABER experiments:
```
./run_saber.sh
```

### Results Analysis
Analyze the experimental results using the provided Jupyter notebooks:

RQ1.ipynb - Analysis for Research Question 1 (workload composition sensitivity)

RQ2.ipynb - Analysis for Research Question 2 (dynamic load adaptation)

### Technical Details
The estimation function implementation for predicting token generation speed can be found in the estimation_function/ folder.
