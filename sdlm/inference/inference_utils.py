import pdb

import numpy as np
import torch
import torch.nn.functional as F

from sdlm.metrics.metrics import distinct_n_grams, zipf
from sdlm.metrics.perplexity import conditional_perplexity, perplexity
from sdlm.metrics.repetition import repetition
from sdlm.utils import convert_to_simplex, join_texts


def sample_logits(sampling_type, logits, top_p, temperature):
    # top-p (nucleus) sampling.
    if sampling_type == "top_p":
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold.
            sorted_indices_to_keep = cumsum_probs < top_p

            # Shift the indices to the right to keep also the first token below the threshold.
            sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
            sorted_indices_to_keep[..., 0] = 1

            indices_to_keep = sorted_indices_to_keep.scatter(dim=2, index=sorted_indices, src=sorted_indices_to_keep)
            filtered_logits = logits.masked_fill(indices_to_keep == 0, -float("Inf"))

            # sample from the filtered distribution.
            token_ids = torch.distributions.categorical.Categorical(logits=filtered_logits).sample()
        else:
            token_ids = torch.argmax(probs, dim=-1)
    else:
        assert NotImplementedError
    return token_ids


def remove_first_occurrence(string, char):
    # We do not strip as we need the spaces as well.
    if char in string:
        idx = string.index(char)
        string = string[idx + len(char) :]
    return string


def keep_till_first_occurrence(string, chars):
    """Given a list of characters, trim the text after the first occurance between them."""
    idxs = [string.index(char) for char in chars if char in string]
    if len(idxs):
        min_idx = np.min(idxs)
        string = string[:min_idx]
    return string


def process_text(texts):
    # TODO(rabeeh): for now we only cover roberta case.
    texts = [keep_till_first_occurrence(text, ["</s>"]) for text in texts]
    texts = [remove_first_occurrence(text, "<s>") for text in texts]
    return texts


def split_into_masked_and_unmasked(token_ids, span_mask, return_masked=None):
    """Given an span_mask, splits the given token_ids into masked and unmasked parts.

    If return_masked is set, only returns the masked parts, if this is set to False,
    only returns the unmasked parts, and If set to None, returns both parts.
    """

    def update_spans(span, masked, unmasked, mask):
        # TODO: this needs to be here for previous version of the codes.
        # span = torch.stack(span)
        masked.append(span) if mask else unmasked.append(span)

    masked = []
    unmasked = []
    prev_mask = span_mask[0]
    span = []
    for _, (token_id, mask) in enumerate(zip(token_ids, span_mask)):
        if mask == prev_mask:
            span.append(token_id)
        else:
            # Adds the previous span.
            update_spans(span, masked, unmasked, prev_mask)
            prev_mask = mask
            span = [token_id]
    # Adds the last span.
    update_spans(span, masked, unmasked, prev_mask)

    if return_masked is None:
        return masked, unmasked

    return masked if return_masked else unmasked


def concatenate_alternatively(longer, shorter, mark=""):
    """Given two lists of strings, concatenates them alternatively.

    We assume that the concatenated string should starts from elements in the longer
    list (which has one extra element). The shorter text can optionally be embraced with
    a `mark` text on both sides.
    """
    concatenated_str = ""
    for l, s in zip(longer, shorter):
        concatenated_str += l + " " + mark + s + mark + " "
    if len(longer) == len(shorter) + 1:
        return concatenated_str + longer[-1]
    elif len(longer) == len(shorter):
        return concatenated_str[:-1]
    else:
        raise ValueError


def aggregate_list(x):
    str = ""
    if len(x) == 0:
        return str
    for l in x:
        str += l + " "
    return str[:-1]


def logits_projection(logits, sampling_type, top_p, simplex_value, temperature):
    # TODO(rabeeh): huggingface has different sampling, like constrastive one.
    # also there are more variant in diffusion-lm.
    token_ids = sample_logits(sampling_type, logits, top_p, temperature)
    return convert_to_simplex(token_ids, simplex_value, vocab_size=logits.shape[2])


def filter_empty(texts):
    """Filters empty texts and return the remained texts and the their indices."""
    list_of_tuples = [(text, i) for i, text in enumerate(texts) if text != ""]
    if len(list_of_tuples) == 0:
        return [], []
    non_empty_texts, remained_inds = list(zip(*list_of_tuples))
    return list(non_empty_texts), list(remained_inds)


def predict_conditional_generated(span_masks, input_ids, tokenizer, predicted_token_ids, prefix_name, skip_special_tokens):
    masked = list(
        map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=True), predicted_token_ids, span_masks)
    )
    unmasked = list(map(lambda x, y: split_into_masked_and_unmasked(x, y, return_masked=False), input_ids, span_masks))
    pred_masked_texts = [tokenizer.batch_decode(x, skip_special_tokens=skip_special_tokens) for x in masked]
    pred_unmasked_texts = [tokenizer.batch_decode(x, skip_special_tokens=skip_special_tokens) for x in unmasked]
    pred_texts = list(map(lambda x, y: concatenate_alternatively(x, y), pred_unmasked_texts, pred_masked_texts))
    pred_texts_marked = list(
        map(lambda x, y: concatenate_alternatively(x, y, mark="***"), pred_unmasked_texts, pred_masked_texts)
    )
    aggregated_masked_texts = list(map(lambda x: aggregate_list(x), pred_masked_texts))
    predicted_tokens = [np.array(item).tolist() for submasked in masked for item in submasked]
    return {
        # prefix_name: pred_texts,
        prefix_name + "_marked": pred_texts_marked,
        prefix_name + "_masked": aggregated_masked_texts,
        prefix_name + "_masked_tokens": predicted_tokens,
    }


def evaluate_generation(
    results,
    data_args,
    causal_model,
    causal_tokenizer,
    is_conditional_generation,
    prefix_lm_eval=False,
    skip_special_tokens=True,
    eval_for_all_metrics=False,
):
    metrics = {}
    # In case of prefix_lm since the generated text is unified, we can evaluate only the masked parts.
    if prefix_lm_eval:
        gold_text_key = "gold_texts_masked"
        # In case of gpt2, we only have the key of `generated_texts_masked`.
        keys = (
            ["generated_texts_masked"]
            if "generated_texts_masked" in results
            else ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        )
    else:
        keys = ["pred_texts_from_simplex", "pred_texts_from_logits"]
        gold_text_key = "gold_texts"

    if is_conditional_generation:
        gold_texts = results[gold_text_key]
        if not skip_special_tokens:
            gold_texts = process_text(gold_texts)
    if "prefixes" in results:
        prefixes = results["prefixes"]
    else:
        prefixes = None

    for key in keys:
        key_metrics = {}
        texts = results[key]
        if not skip_special_tokens:
            texts = process_text(texts)

        non_empty_texts, remained_indices = filter_empty(texts)
        if len(non_empty_texts) == 0:
            continue

        # Perplexity measured by a causal model.
        if prefixes is None:
            key_metrics.update(
                {"perplexity": perplexity(non_empty_texts, causal_model, causal_tokenizer)["mean_perplexity"]}
            )
        else:
            non_empty_prefixes = [prefix for i, prefix in enumerate(prefixes) if i in remained_indices]
            perplexity_results = conditional_perplexity(non_empty_texts, non_empty_prefixes, causal_model, causal_tokenizer)
            key_metrics.update(
                {
                    "perplexity": perplexity_results["mean_perplexity"],
                    "total_perplexity": perplexity_results["mean_perplexity_total"],
                }
            )

        # Dist-1,2,3 measurements.
        key_metrics.update(distinct_n_grams(texts))

        # Metrics requiring the gold text.
        if is_conditional_generation and eval_for_all_metrics:
            # Note that we need to pass both context and predicted texts to this metric.
            # remained_gold_texts = [text for i, text in enumerate(gold_texts) if i in remained_indices]
            # remained_prefixes = [text for i, text in enumerate(prefixes) if i in remained_indices]
            texts_with_context = join_texts(prefixes, texts)
            gold_with_context = join_texts(prefixes, gold_texts)
            length = data_args.max_seq_length - data_args.truncation_length

        if key + "_tokens" in results and eval_for_all_metrics:
            key_metrics.update(repetition(results[key + "_tokens"], causal_tokenizer))
            key_metrics.update(zipf(results[key + "_tokens"]))

        # Adds the metrics.
        key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
        metrics.update(key_metrics)

    return metrics
