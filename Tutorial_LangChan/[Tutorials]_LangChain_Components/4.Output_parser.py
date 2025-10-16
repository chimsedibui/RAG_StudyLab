import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

model_name: str = ".hf_models/Phi-3-mini-4k-instruct"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype= torch.bfloat16,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
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
    "temperature": 0
}

llm = HuggingFacePipeline(
    pipeline=model_pipeline,
    model_kwargs=gen_kwargs
)

from langchain_core.pydantic_v1 import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=Joke)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm 
joke_query = "Tell me a joke."
output = chain.invoke({"query": joke_query})

print(output)

parser_output = parser.invoke(output)
print(parser_output)
chain = prompt | llm | parser
output = chain.invoke({"query": joke_query})
print(output)


