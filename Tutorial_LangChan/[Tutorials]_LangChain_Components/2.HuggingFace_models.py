
import os
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
os.environ["HF_CACHE"] = "/data_hdd_16t/khanhtran/LLM/.hf_models"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_name: str = r".hf_models/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype= torch.bfloat16,    
    low_cpu_mem_usage=True, 
    #trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
max_new_token = 1024

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=max_new_token,
    pad_token_id=tokenizer.eos_token_id
)
gen_kwargs = {
    "temperature": 0.9
}
llm = HuggingFacePipeline(
    pipeline=model_pipeline,
    model_kwargs=gen_kwargs
)

prompt_template = PromptTemplate.from_template(
"""
"Instruct:{prompt}\nOutput:"
"""
)

user_prompt = "Write a detailed analogy between mathematics and a lighthouse."
messages = prompt_template.format(prompt=user_prompt)
output = llm.invoke(messages)
print(output)


