# Architecture

Core pipeline:
1. Dataset prep (`dataset_prep`)
2. Jacobian build (`jacobian_modeling/build.py`)
3. Evaluation plots (`jacobian_modeling/evaluate.py`)
4. Optimization (`jacobian_modeling/optimize.py`)
5. Runtime tracking (`tracking_runtime/*`)

All file paths are controlled via config (`configs/*.yaml`) and resolved relative to the config file.
