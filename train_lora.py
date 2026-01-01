import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import torch


os.environ["WANDB_DISABLED"] = "true"


# robust import for prepare_model_for_int8_training / kbit
try:
    from peft import prepare_model_for_int8_training as prepare_kbit_training
except Exception:
    try:
        from peft import prepare_model_for_kbit_training as prepare_kbit_training
    except Exception:
        raise ImportError(
            "peft helper function not found. Try running: pip install -U peft "
            "or pip install peft==0.5.3"
        )

MODEL_NAME = os.environ.get("MODEL_NAME","meta-llama/Meta-Llama-3-8B-Instruct")  # replace if needed
TRAIN_FILE = os.environ.get("TRAIN_FILE","data/finance_instructions.jsonl")
OUT_DIR = os.environ.get("OUT_DIR","./lora_finance")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 512))

print("Model:", MODEL_NAME)
print("Train file:", TRAIN_FILE)

# Load dataset
ds = load_dataset("json", data_files={"train":TRAIN_FILE})
# Train/val split small
ds = ds["train"].train_test_split(test_size=0.15, seed=42)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def make_prompt(example):
    inst = example.get("instruction","")
    inp = example.get("input","")
    out = example.get("output","")
    prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return prompt

def preprocess(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        text = (
            f"### Instruction:\n{inst}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{out}"
        )
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=ds["train"].column_names
)


# Load model in 8-bit (memory saving)
print("Loading model in 8-bit (this may still be heavy)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

# Prepare for int8 training and apply LoRA
model = prepare_kbit_training(model)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj","k_proj","o_proj"],  # common targets; may vary by model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("PEFT model prepared. Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Training args: tiny batch, accumulate grads to simulate bigger batch
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch size = 8
    warmup_steps=10,
    max_steps=200,    # small quick run for learning; increase later
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    fp16=True,
    save_total_limit=3,
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator
)

trainer.train()
print("Training finished. Saving adapter.")
model.save_pretrained(OUT_DIR)
