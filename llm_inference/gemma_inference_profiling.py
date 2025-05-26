from huggingface_hub import login
import json

with open("credentials.json", "r") as f:
    credentials = json.load(f)
    key = credentials["hf_read_token"]
    print(key)
if key:
    print("Logged in to Hugging Face!")

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import time

# Enable parallel processing
# torch.set_num_threads(8)  # Use 8 CPU threads
# torch.set_num_interop_threads(8)  # Use 8 threads for parallel data loading

# Load and cache model locally to avoid downloading it every time
model_name = "gemma-3-1b-it"
model_id = "google/" + model_name
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=f"./llm_cache/{model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir=f"./llm_cache/{model_name}", 
    torch_dtype=torch.bfloat16
)

prompt = "Hello, today is Mother's day, and I want to write a story about a mother and her daughter who have had a hard life due to war in the area. The mother is a strong woman who has been through a lot, but she is still able to find joy in her life. The daughter is a young girl who is still learning about the world and how to live in it. The story is about the mother and daughter's journey together, and how they are able to find joy in their lives despite the challenges they have faced."
inputs = tokenizer(prompt, return_tensors="pt")

MAX_TOKENS = 10

print("Warming up the model with one round of generation...")
model.generate(
    **inputs,
    max_new_tokens=MAX_TOKENS,
    do_sample=False,
)

# Run measurements MAX_TRIALS times and average
MAX_TRIALS = 5
first_token_times = []
avg_token_times = []

for i in range(MAX_TRIALS):
    print(f"\nRun {i+1}/{MAX_TRIALS}:")
    
    # Measure the time to generate the first output token
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        use_cache=True,
    ) 
    print(outputs) # to ensure the generator is executed
    end_time = time.time()
    first_token_time = end_time - start_time
    first_token_times.append(first_token_time)
    print(f"Time to generate first output token: {first_token_time:.2f} seconds")

    # Measure the average time to generate the rest of the tokens
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
        use_cache=True,
    )
    print(outputs) # to ensure the generator is executed
    end_time = time.time()
    avg_token_time = (end_time - start_time - first_token_time) / (MAX_TOKENS - 1)  # Subtract 1 for the first token
    avg_token_times.append(avg_token_time)
    print(f"Average time to generate the rest of the tokens: {avg_token_time:.2f} seconds")

# Calculate and print final averages
final_first_token_avg = sum(first_token_times) / len(first_token_times)
final_avg_token_time = sum(avg_token_times) / len(avg_token_times) # Note: this is avg of avgs.
print(f"\nFinal Results:")
print(f"Average time to generate first token (across {len(first_token_times)} runs): {final_first_token_avg:.2f} seconds")
print(f"Average time to generate subsequent tokens (across {len(avg_token_times)} runs): {final_avg_token_time:.2f} seconds")
    
# Cleanup
print("\nCleaning up memory...")
del model
del tokenizer
import gc
gc.collect()  # Run garbage collector