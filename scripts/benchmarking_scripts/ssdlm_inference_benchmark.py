'''
SSD LM Inference script from https://colab.research.google.com/drive/1vNKqvzzJQp3k89QPuns5ibsq-VNC9wGN?usp=sharing
'''

import os
import sys
import torch
import transformers
import accelerate
import numpy as np
from termcolor import colored
import time
import json
import random
import math
import logging
from tqdm.auto import tqdm
from argparse import Namespace
from huggingface_hub.file_download import hf_hub_download
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

#gen_len = 450
block_size = 25
diffusion_steps = 100

for gen_len in [100, 200, 300]:
    print(f'------------------{gen_len}----------------------------------')
    # if doing unconstrained generation
    args = Namespace()
    args.model_name_or_path = "xhan77/ssdlm"
    args.max_seq_length = gen_len + 50
    args.one_hot_value = 5
    args.decoding_block_size = block_size
    args.decode_total_gen_len = gen_len # should be divisible by decode_depth
    args.decode_depth = math.ceil(gen_len / block_size)
    args.decode_log_interval = 100
    args.total_t = diffusion_steps
    args.projection_top_p = 0.95
    args.seed = 2022
    args.decode_ctr_lr = 0.0 # set to 0 for unconstrained generation, large value for controlled generation
    args.use_slow_tokenizer = False

    # a few helper functions
    def get_time_variables(t, total_t, device): # cosine schedule

        def ft(small_t, big_t, s=1e-4):
            return torch.cos((small_t / big_t + s) / (1 + s) * math.pi / 2) ** 2

        alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape, device=device), total_t)
        alpha_t_minus_bar = ft(t-1, total_t) / ft(torch.zeros(t.shape, device=device), total_t)
        beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
        beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
        alpha_t = 1 - beta_t
        return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t


    def apply_controlling_drift(args, perturbed_inputs_diralpha):
        if args.decode_ctr_lr <= 0:
            args.ctr_loss = -1
            return perturbed_inputs_diralpha

        if args.ctr_model is None:
            args.ctr_model = AutoModelForSequenceClassification.from_pretrained(args.ctr_model_name).to(args.accelerator.device)
        optimizing_label_index = args.ctr_opt_label_idx

        for ctr_i in range(1):
            with torch.enable_grad():
                perturbed_inputs_diralpha_4ctr = perturbed_inputs_diralpha.clone()
                perturbed_inputs_diralpha_4ctr.requires_grad_()
                perturbed_inputs_simplex_4ctr = torch.nn.functional.softmax(perturbed_inputs_diralpha_4ctr, dim=-1)
                perturbed_inputs_embeds_4ctr = torch.nn.functional.linear(perturbed_inputs_simplex_4ctr, args.ctr_model.get_input_embeddings().weight.t())
                ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model(inputs_embeds=perturbed_inputs_embeds_4ctr).logits, dim=-1)[:,optimizing_label_index].mean()
                args.ctr_loss = ctr_loss
                ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_inputs_diralpha_4ctr)[0]
            perturbed_inputs_diralpha = perturbed_inputs_diralpha + args.decode_ctr_lr * ctr_delta # we use a fixed balancing factor in this work, which can be improved in the future
        
        return perturbed_inputs_diralpha


    def logits_sampling_projection(logits, top_p, one_hot_value):
        assert len(logits.size()) == 3
        very_low_value = -10000

        probs = torch.nn.functional.softmax(logits, dim=-1)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < top_p
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        valid_indices = nucleus.scatter(2, indices, nucleus)

        filtered_logits = logits.masked_fill(valid_indices == 0, -float('Inf'))
        m = torch.distributions.categorical.Categorical(logits=filtered_logits)
        selected = m.sample()
        return 2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value


    def decode(args, batch_input_ids, dec_depth, total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer):
        batch_size = 1 # for the demo
        if args.decode_truncate_len > 0:
            diffusion_input_ids = batch_input_ids[:, args.context_size:-args.decode_truncate_len]
        else:
            diffusion_input_ids = batch_input_ids[:, args.context_size:]
        
        assert (args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0, "check whether the total generation length is divisible by the depth of decoding"
        unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
        if args.context_size > 0:
            unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        else:
            unit_context_input_ids = None
        history_decode_ids = None

        start_time = time.time()
        for i in range(dec_depth):
            unit_noise = args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size), device=args.accelerator.device)
            xt = unit_noise

            if unit_context_input_ids is not None:
                context_inputs_embeds = model_embedding_lut(unit_context_input_ids)
            else:
                context_inputs_embeds = None

            t_range = list(range(1, total_t+1))
            t_range.reverse()
            progress_bar = tqdm(range(len(t_range)), disable=not args.accelerator.is_local_main_process)
            
            for t in t_range:
                selected_t = torch.tensor([t], device=args.accelerator.device, dtype=torch.float).repeat(batch_size)
                alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t = get_time_variables(selected_t, total_t, args.accelerator.device)
                zt = args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size), device=args.accelerator.device)
                
                perturbed_inputs_diralpha = xt
                perturbed_inputs_simplex = torch.nn.functional.softmax(perturbed_inputs_diralpha, dim=-1)

                perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
                t_progress = selected_t / total_t
                timestep_embeds = timestep_layer(t_progress.view(batch_size,1,1).repeat(1,unit_seq_len,1))

                diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
                if context_inputs_embeds is not None:
                    diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
                outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
                equivalent_score = outputs.logits
                if unit_context_input_ids is not None:
                    equivalent_score = equivalent_score[:, unit_context_input_ids.size(1):].contiguous()

                # controlled generation if the balancing factor > 0
                equivalent_score = apply_controlling_drift(args, equivalent_score)

                projected_logits = logits_sampling_projection(equivalent_score, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)

                xt = torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1) * projected_logits
                xt = xt + torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1) * zt

                progress_bar.update(1)

                if t % args.decode_log_interval == 0 or t == 1:
                    simplex = torch.nn.functional.softmax(xt, dim=-1)
                    #logger.info(f"noise coef at t: {torch.sqrt(1 - alpha_t_bar).item()}")

                    if unit_context_input_ids is not None:
                        context_sequences = tokenizer.batch_decode(unit_context_input_ids.detach())
                        # logger.info(f"context: {context_sequences}")
                    
                    real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                    sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach())
                    # logger.info(f"t={t} (argmax w_t-1): {colored(str(sampled_sequences), 'red')}")

                    simplex = equivalent_score
                    real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                    sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach())
                    # logger.info(f"t={t} (argmax w_logits): {colored(str(sampled_sequences), 'blue')}")

                    #alt_i = 1 # look at the second best candidate; note that the whole sequence is not meaningful; each token can be considered as a substitution for the corresponding token in the argmax sequence
                    #alt_real_token_ids_list = torch.topk(simplex, alt_i+1, dim=-1).indices[:, :, alt_i].view(batch_size, unit_seq_len)
                    #alt_sampled_sequences = tokenizer.batch_decode(alt_real_token_ids_list.clone().detach().to('cpu'))
                    # logger.info(f"t={t} (argsecondmax w_logits): {alt_sampled_sequences}")

                    # logger.info(f"ctr loss: {args.ctr_loss}")
            
            unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
            if history_decode_ids is None:
                history_decode_ids = real_token_ids_list
            else:
                history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        if args.context_size > 0:
            init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
            context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
        else:
            init_context_input_ids = None
            context_sequences = None
        gold_sequences = tokenizer.batch_decode(diffusion_input_ids.clone().detach().to('cpu'))
        sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'))
        logger.info(f"Time taken: {time.time() - start_time}")
        logger.info(f"context: {context_sequences}")
        logger.info(f"gold: {colored(str(gold_sequences), 'yellow')}")
        logger.info(f"generation: {colored(str(sampled_sequences), 'red')}")

        return history_decode_ids, init_context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences

    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [stdout_handler]
    logging.basicConfig(
        level=logging.DEBUG, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path, max_position_embeddings = 100000)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, from_tf=False, config=config, ignore_mismatched_sizes=True)

    model.resize_token_embeddings(len(tokenizer))
    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)

    embedding_sum_layer = torch.nn.Linear(vocab_size, hidden_size, bias=False)
    _stdict = torch.load(os.path.join(hf_hub_download(args.model_name_or_path, "embed_sum_layer.pt")))
    _stdict = dict((_k[len("module."):], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
    embedding_sum_layer.load_state_dict(_stdict)

    timestep_layer = torch.nn.Linear(1, hidden_size, bias=True)
    _stdict = torch.load(os.path.join(hf_hub_download(args.model_name_or_path, "timestep_layer.pt")))
    _stdict = dict((_k[len("module."):], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
    timestep_layer.load_state_dict(_stdict)

    model, embedding_sum_layer, timestep_layer = accelerator.prepare(model, embedding_sum_layer, timestep_layer)

    # a bit more preparation before decoding
    model.eval()
    model_embedding_lut = accelerator.unwrap_model(model).get_input_embeddings()
    args.vocab_size = vocab_size
    args.accelerator = accelerator
    args.ctr_model = None
    args.orig_decode_truncate_len = args.max_seq_length - args.decode_total_gen_len

    # ENTER YOUR PROMPT HERE!
    prompt = "A man of innumerable personalities and powers vs. the most powerful artificial intelligence in this universe: Legion vs. Nimrod! With Nightcrawler in Orchis clutches, David Haller and his allies will have to confront the mastermind who"


    if prompt[0] != ' ': # prepend a space to the prompt if necessary
        prompt = f" {prompt}" # can use ' ' or '\n\n' as prefix to the prompt
    input_ids = torch.LongTensor(tokenizer.encode(prompt, add_special_tokens=False)).to(args.accelerator.device)
    args.context_size = len(input_ids)
    assert args.max_seq_length - args.decode_total_gen_len - args.context_size > 0, "check the length of the prompt"
    args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
    input_ids = input_ids.unsqueeze(0)

    # start sampling from SSD-LM
    with torch.inference_mode():
        for _ in range(5):
            history_decode_ids, context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences = \
                decode(args, input_ids, args.decode_depth, args.total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer)
