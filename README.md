# Simplex Diffusion Language Model (SDLM).

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
