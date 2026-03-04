import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# =========================
# Config
# =========================

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/finance_instructions.jsonl")

# ⚠️ Change this in Colab to Google Drive path if mounted
OUT_DIR = "/content/drive/MyDrive/lora_finance"

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 256))

print("Model:", MODEL_NAME)
print("Train file:", TRAIN_FILE)
print("Output dir:", OUT_DIR)

# =========================
# Load Dataset
# =========================

ds = load_dataset("json", data_files={"train": TRAIN_FILE})
ds = ds["train"].train_test_split(test_size=0.15, seed=42)

# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# =========================
# Load Model in 8-bit (Modern Way)
# =========================

print("Loading model in 8-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# =========================
# Prepare for k-bit training
# =========================

try:
    from peft import prepare_model_for_kbit_training
except ImportError:
    from peft import prepare_model_for_int8_training as prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

# =========================
# LoRA Configuration
# =========================

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", trainable_params)

# =========================
# Training Arguments
# =========================

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=20,
    max_steps=300,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="steps",
    save_steps=300,
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator
)

# =========================
# Train
# =========================

trainer.train()

print("Training finished. Saving LoRA adapter...")
model.save_pretrained(OUT_DIR)

print("Saved to:", OUT_DIR)