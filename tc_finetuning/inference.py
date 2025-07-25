from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json

tok = AutoTokenizer.from_pretrained("xform-lora")
model = AutoModelForCausalLM.from_pretrained("xform-lora").to("cuda")

prompt = "<SEED_10> genetics, heliocentrism, XFORM,"
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=3)
print(tok.decode(out[0]))
