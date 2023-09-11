import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large', max_position_embeddings=100000, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

prompt = "A man of innumerable personalities and powers vs. the most powerful artificial intelligence in this universe: Legion vs. Nimrod! With Nightcrawler in Orchis clutches, David Haller and his allies will have to confront the mastermind who"

inputs = tokenizer(prompt, return_tensors="pt")

model = model.to('cuda')
inputs = inputs.to('cuda')

for lengths in [100, 200, 300]:
    print(f'-----------------{lengths}---------------------------')
    for _ in range(5):
        with torch.inference_mode():
            time_start = time.time()
            outputs = model.generate(**inputs, min_length=lengths, max_length=lengths)
            print(f"Time taken: {time.time() - time_start}")
            print(f"Output: {tokenizer.decode(outputs[0])}")
    print(f'-----------------{lengths} end---------------------------')
    