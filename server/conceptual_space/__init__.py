from .concept_query_to_space import get_conceptual_space_from_concept
from .paper_to_space import get_conceptual_space_from_paper
from .entailment_score import compute_entailment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize entailment model and tokenizer at import
_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

def get_conceptual_space(input_str, input_type='concept', model=None, paper_name=None):
    if input_type == 'concept':
        concepts = get_conceptual_space_from_concept(input_str, model=model)
    elif input_type == 'paper':
        if paper_name is None:
            concepts = get_conceptual_space_from_paper("None", input_str, model=model)
        else:
            concepts = get_conceptual_space_from_paper(paper_name, input_str, model=model)
    else:
        raise ValueError("input_type must be either 'concept' or 'paper'")
    return {
        'conceptual_space': concepts,
        'compute_entailment': compute_entailment,
        'entailment_tokenizer': _tokenizer,
        'entailment_model': _model
    } 