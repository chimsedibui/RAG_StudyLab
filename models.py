import os
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["HF_HOME"] = ".hf_models"
os.environ["HF_CACHE"] = ".hf_models"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
messages = [
    {"role": "user", "content": "I want to know who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

#meta-llama/Llama-3.2-1B