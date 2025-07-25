import json, re
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # small model to check for learned rule - maybe better to use mistral 7B or llama 3 8b? But harder on a standard computer
TRAIN_PATH = "data/train.jsonl"
EVAL_PATH = "data/eval.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def read_seeds(path):
    seeds=set()
    for line in open(path):
        d=json.loads(line)
        seeds.add(d["seed"])
    return sorted(seeds)

seed_tokens = list(set(read_seeds(TRAIN_PATH)) | set(read_seeds(EVAL_PATH)))
tokenizer.add_tokens(seed_tokens)
print("Added seed tokens:", len(seed_tokens))

raw = load_dataset("json", data_files={"train": TRAIN_PATH, "eval": EVAL_PATH})

def format_example(ex):
    prompt = f"{ex['seed']} {ex['src_lex']}, {ex['tgt_feat']}, XFORM,"
    target = ex["out_lex"]
    text = prompt + " " + target
    return tokenizer(text, truncation=True)

tokenized = raw.map(format_example, remove_columns=raw["train"].column_names)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model.resize_token_embeddings(len(tokenizer))

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

args = TrainingArguments(
    output_dir="xform-lora",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    fp16=True,
    report_to="none"
)

dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=dc,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["eval"],
)

trainer.train()
model.save_pretrained("xform-lora")
tokenizer.save_pretrained("xform-lora")
print("Fine-tune complete.")

