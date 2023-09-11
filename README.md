# Simplex Diffusion Language Model (SDLM).

We introduce Text-to-text Self-conditioned Simplex Diffusion (TESS), a text diffusion model that is fully non-autoregressive, employs a new form of self-conditioning, and applies the diffusion process on the logit simplex space rather than the typical learned embedding space.

# How to setup the environment
```
conda env create -f environment.yaml --prefix  ${LOCAL_DIR}/conda/envs/sdlm
python setup develop
```
to update environment after installation:
```
conda env update --file environment.yaml --prune
```

# Process the data.
```
bash scripts/run_process_data.sh  configs/openwebtext.json
```

# Run the training and evaluation
Please see the `run_train` and `run_eval` scripts under the scripts directory.
Example:
```
bash scripts/run_train.sh  configs/accelerate_1_gpu.yaml  configs/simple_data_test.json
```
