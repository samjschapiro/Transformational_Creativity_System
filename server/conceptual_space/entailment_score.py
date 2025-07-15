# File 3: entailment_score.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

# one issue here is that textual entiailment sometimes fails for semantic relationships
''' Optionally annotate with:

{"entailment": "semantic", "score": 0.13}

{"entailment": "functional", "score": 1.0}'''

def compute_entailment(premise: str, hypothesis: str, tokenizer_=None, model_=None) -> float:
    """
    Compute the entailment probability that the premise entails the hypothesis.
    Optionally accepts a tokenizer and model; defaults to the module's pre-initialized ones.
    """
    if tokenizer_ is None:
        tokenizer_ = tokenizer
    if model_ is None:
        model_ = model
    inputs = tokenizer_(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model_(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    entailment_prob = probs[0][2].item()  # Index 2 = entailment
    return entailment_prob

# Example usage
if __name__ == "__main__":
    premise = "All transformers use self-attention."
    hypothesis = "Self-attention is used in the transformer architecture."
    print(f"Entailment score: {compute_entailment(premise, hypothesis):.3f}")
