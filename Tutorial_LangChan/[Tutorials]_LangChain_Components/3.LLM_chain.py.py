import os
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
os.environ["TRANSFORMERS_OFFLINE"] = "1"

model_name: str = ".hf_models/Phi-3-mini-4k-instruct"

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype= torch.bfloat16,
    #quantization_config=nf4_config,
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

max_new_token = 256

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=max_new_token,
    pad_token_id=tokenizer.eos_token_id
)

gen_kwargs = {
    "temperature": 1
}

llm = HuggingFacePipeline(
    pipeline=model_pipeline,
    model_kwargs=gen_kwargs
)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(""""Instruct:{prompt}\nOutput:"                                 
"""
)

chain = prompt | llm
output = chain.invoke(
    {
        "prompt": "Write a detailed analogy between mathematics and a lighthouse."
    }
)

print(output)


