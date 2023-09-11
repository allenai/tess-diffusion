# Benchmarking Scripts

Scripts used for benchmarking generation speeds. Pretty straightforward, should print the generation speeds needed. Assume you have one GPU large enough to fit the models, with requirements correctly installed. You can alter the core speed testing loop to test different generation lengths. However, training models with lengths over 512 tokens will require modifying the roberta model implementation to avoid applying position embeddings. In future work we will explore using underlying models that allow longer sequence lengths.

Example run for TESS inference speed script:
```bash
python  -m scripts.benchmarking_scripts.test_tess_inference_speed --model_name_or_path roberta-base --simplex_value 5  --num_diffusion_steps 1000  --num_inference_diffusion_steps 1000  --beta_schedule squaredcos_improved_ddpm  --top_p 0.8 --self_condition logits_mean --self_condition_mix_before_weights
```

SSD-LM script:
```bash
python -m scripts.benchmarking_scripts.ssdlm_inference_benchmark
```

Autoregressive (i.e., Bart) script:
```bash
python -m scripts.benchmarking_scripts.test_hf_autoregressive_generation
```

The other two scripts shouldn't need runtime arguments, and can be run in a similar manner as above.
