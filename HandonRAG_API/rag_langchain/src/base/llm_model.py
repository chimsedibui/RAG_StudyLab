import os
import torch 
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = ".hf_models"
os.environ["HF_CACHE"] = ".hf_models"

nf8_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

def get_hf_llm(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_new_token = 1024, **kwargs
    ):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf8_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )

    model_pipeline = pipeline(
        task= "text-generation",
        model=model,
        tokenizer= tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )
    return llm

if __name__ == "__main__":
    llm = get_hf_llm()
    print(llm.invoke("What is LLM?"))