import os
import json
import uuid

from components.text_extraction import extract_text_from_pdf, extract_text_from_epub, segment_text
from components.response_parsing import extract_claims
from components.logic_formalization import formalize_claims, check_contradictions
from components.pdf_generation import generate_output_pdf
from components.reconstruction import generate_reconstructions
from test_conceptual_space import run_conceptual_space_visualization

def save_to_json(data, filename="output_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def formalize_file(file_path, mode):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".epub":
        text = extract_text_from_epub(file_path)
    else:
        raise ValueError("Unsupported file format")

    segments = segment_text(text)
    parsed_data = extract_claims(segments)

    # Gather all claims
    all_claims_data = []
    for item in parsed_data:
        seg_idx = item.get("segment_index")
        if seg_idx is not None:
            for claim in item.get("claims", []):
                all_claims_data.append({"segment_index": seg_idx, "english": claim})

    # Formalize claims
    print("All claims data:", all_claims_data)
    formalized_data = formalize_claims(all_claims_data, mode)

    # compute Formalizability Index
    total_segments = len(parsed_data)
    formalizable_segments = sum(1 for item in parsed_data if item.get("claims"))
    formalizability_index = (
        formalizable_segments / total_segments if total_segments > 0 else 0
    )

    # Check contradictions
    contradiction_found = check_contradictions(formalized_data.get("axioms", []))
    base_url = file_path
    for ax in formalized_data.get("axioms", []):
        seg_idx = ax["segment_index"]
        ax["source"] = f"{base_url}#segment-{seg_idx}"
        ax["flag"] = "contradiction" if contradiction_found else "none"

    if ax["flag"] == "contradiction":
        print("contradiction found!")

    # Save to JSON in root-level outputs directory
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    json_output_path = os.path.join(outputs_dir, "final_data.json")
    save_to_json(formalized_data, json_output_path)

    # Read final data
    with open(json_output_path, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    # Generate reconstructions
    logic_text, english_text = generate_reconstructions(final_data["axioms"])

    # Now generate the PDF with the Formalizability Index at the top
    output_id = str(uuid.uuid4())
    pdf_output_path = os.path.join(outputs_dir, f"{output_id}.pdf")
    generate_output_pdf(
        logic_text,
        english_text,
        pdf_output_path,
        formalizability_index,
        total_segments,
        formalizable_segments,
    )

    # --- New: Visualize conceptual space ---
    run_conceptual_space_visualization(json_output_path, show=True)

    return {
        "axioms": final_data["axioms"],
        "output_pdf": pdf_output_path,
        "logic_reconstruction": logic_text,
        "english_reconstruction": english_text,
    }