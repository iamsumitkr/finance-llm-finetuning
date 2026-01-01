import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "./lora_finance"  # not included in repo

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

def ask(prompt):
    text = f"""### Instruction:
{prompt}

### Input:

### Response:
"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    ask("Explain EBITDA in one sentence.")
    ask("What happens to bond prices when interest rates rise?")
