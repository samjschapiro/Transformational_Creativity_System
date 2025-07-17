import json
import os
from conceptual_space import get_conceptual_space
from conceptual_space.visualize_space import visualize_conceptual_space

def run_conceptual_space_visualization(final_data_path="outputs/final_data.json", show=True):
    # Load English claims and paper name from final_data.json
    with open(final_data_path, "r") as f:
        data = json.load(f)

    # Extract all English claims
    english_claims = [axiom["english"] for axiom in data["axioms"] if "english" in axiom]

    # Join claims into a single string
    claims_str = "\n".join(english_claims)

    # Extract paper name from the first axiom's source field
    first_source = data["axioms"][0]["source"]
    paper_name = os.path.basename(first_source.split("#")[0]) if "source" in data["axioms"][0] else "None"

    result = get_conceptual_space(claims_str, input_type="paper", paper_name=paper_name)
    concepts = result['conceptual_space']
    model = result['entailment_model']
    tokenizer = result['entailment_tokenizer']

    visualize_conceptual_space(concepts, show=False, save_path="conceptual_space.png", entailment_model=model, entailment_tokenizer=tokenizer)

if __name__ == "__main__":
    run_conceptual_space_visualization()