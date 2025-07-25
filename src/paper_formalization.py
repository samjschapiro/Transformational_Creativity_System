import json
import re
from sympy.logic.boolalg import sympify, simplify_logic
from api_call import generate_response
import os
import json
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import nltk
nltk.download('punkt_tab')
from nltk import sent_tokenize 
    # ensure NLTK is set up: python -m nltk.downloader punkt
import PyPDF2
from ebooklib import epub
from bs4 import BeautifulSoup
from pdf_generation import generate_output_pdf
import requests
from urllib.parse import urlparse

tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

# has 13 functions
#       1. extract_text_from_pdf(pdf_path)
#       2. extract_text_from_epub(epub_path)
#       3. segment_text(text)
#       4. save_to_json(data, filename="output_data.json")
#       5. formalize_file(file_path, mode)
#       6. formalize_claims(all_claims_data, mode)
#       7. check_contradictions(axioms)
#       8. compute_entailment(premise, hypothesis, tokenizer_=None, model_=None)
#       9. extract_json_from_response(response_content)
#       11. parse_combined_responses(filename="all_responses.txt")
#       12. extract_claims(segments, output_file="all_responses.txt")
#       13. generate_reconstructions(axioms)


def formalize_file(file_url, mode='english'):
    # Determine if input is a URL or local file path
    parsed = urlparse(file_url)
    is_url = parsed.scheme in ("http", "https")

    if is_url:
        # Download the file to a temporary location
        ext = os.path.splitext(parsed.path)[1].lower()
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tmp_downloads')
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"downloaded{ext}")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        file_path = tmp_path
    else:
        file_path = file_url
        ext = os.path.splitext(file_path)[1].lower()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

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
    base_url = file_url
    for ax in formalized_data.get("axioms", []):
        seg_idx = ax["segment_index"]
        ax["source"] = f"{base_url}#segment-{seg_idx}"

        # ax["flag"] = "contradiction" if contradiction_found else "none"
        # if ax["flag"] == "contradiction":
        #     print("contradiction found!")

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
    generate_output_pdf(logic_text, english_text, pdf_output_path, formalizability_index, total_segments, formalizable_segments)

    return {
        "axioms": final_data["axioms"],
        "output_pdf": pdf_output_path,
        "logic_reconstruction": logic_text,
        "english_reconstruction": english_text,
    }

def formalize_claims(all_claims_data, mode):
    mode_instructions = {
        "logic": "Convert the claims into formal logical propositions.",
        "english": "Convert the claims into structured, simplified English formalizations."
    }

    formatted_claims = "\n".join([f"- (Index {c['segment_index']}) {c['english']}" for c in all_claims_data])

    prompt = f"""
    You are an assistant specialized in logic and formal reasoning.
    I will provide you with a list of English philosophical claims.
    For each claim:
    1. Provide the original English claim.
    2. {mode_instructions[mode]}
    3. Include the segment_index.

    Return a JSON with a "axioms" list. Each object: "segment_index", "english", "formal".
    Return only valid JSON. Do not include Markdown code fences or additional text.
    Input claims:
    {formatted_claims}
    """

    response = generate_response(prompt)
    response_content = response.content.strip()
    response_content = re.sub(r'```json|```', '', response_content).strip()

    try:
        if response_content.startswith("{") or response_content.startswith("["):
            formalized_data = json.loads(response_content)
        else:
            print(f"Invalid response format: {response_content}")
            formalized_data = {"axioms": []}
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Raw response content: {response.content}")
        formalized_data = {"axioms": []}

    return formalized_data

def check_contradictions(axioms):
    statements = [ax["formal"] for ax in axioms if ax.get("formal")]
    if not statements:
        return False
    combined = " & ".join([f"({stmt})" for stmt in statements])
    try:
        expr = sympify(combined) # turns string into symbolic expression
        simplified = simplify_logic(expr)
        return simplified == False # if false, there is contradiction -> returns true
    except:
        return False

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

import re
import json

def extract_json_from_response(response_content):
    try:
        # Remove common markdown wrappers
        clean = re.sub(r'```json|```', '', response_content).strip()

        # Try to match a full JSON array first
        array_match = re.search(r'\[\s*{.*?}\s*]', clean, re.DOTALL)
        if array_match:
            return json.loads(array_match.group(0))

        # Fallback: try to match a single object
        obj_match = re.search(r'{.*}', clean, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(0))

        print("[extract_json] No valid JSON block found.")
        return None
    except json.JSONDecodeError as e:
        print(f"[extract_json] JSON decoding error: {e}")
        return None

def parse_combined_responses(filename="all_responses.txt"):
    if not os.path.exists(filename):
        print(f"No combined response file found: {filename}")
        return []

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    segments = content.split("---END-OF-SEGMENT---")
    parsed_data = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        clean_segment = re.sub(r'```json|```', '', segment).strip()
        clean_segment = clean_segment.replace('```', '').strip()
        clean_segment = clean_segment.strip('`')
        json_match = re.search(r'{.*}', clean_segment, re.DOTALL)
        if not json_match:
            print("Skipping: No valid JSON found in a segment.")
            parsed_data.append({
                "segment_index": None,
                "claims": [],
                "arguments": [],
                "examples": [],
                "decorative": []
            })
            continue

        try:
            data = json.loads(json_match.group(0))
            parsed_data.append(data)
        except json.JSONDecodeError as e:
            print(f"Skipping JSON decoding error: {e}")
            parsed_data.append({
                "segment_index": None,
                "claims": [],
                "arguments": [],
                "examples": [],
                "decorative": []
            })

    return parsed_data

def extract_claims(segments, output_file=None):
    """
    Extract claims and related information from text segments.
    Instead of writing to a file, collect results in a list and return directly.
    If output_file is provided, optionally save the results as JSON.
    """
    results = []

    for seg_idx, sentence in segments:
        prompt = f"""
        You are an assistant specialized in philosophy and logic.
        You will receive a segment of a philosophical text along with a segment index.
        Identify:
        1. Core philosophical claims or axioms.
        2. Supporting arguments.
        3. Illustrative examples.
        4. Decorative or rhetorical language.

        Output a JSON with keys: "segment_index", "claims", "arguments", "examples", "decorative".

        Segment index: {seg_idx}
        Segment text: "{sentence}"
        """
        response = generate_response(prompt)
        if not response or not hasattr(response, "content"):
            result = {
                "segment_index": seg_idx,
                "claims": [],
                "arguments": [],
                "examples": [],
                "decorative": []
            }
        else:
            parsed = extract_json_from_response(response.content)
            if parsed is None:
                result = {
                    "segment_index": seg_idx,
                    "claims": [],
                    "arguments": [],
                    "examples": [],
                    "decorative": []
                }
            else:
                result = parsed
        results.append(result)

    if output_file:
        save_to_json(results, filename=output_file)

    return results

def generate_reconstructions(axioms):
    # Sort axioms by segment_index
    axioms = sorted(axioms, key=lambda x: x.get("segment_index", 0))

    # Build logic-based reconstruction text
    logic_lines = []
    # Only add a title if we have actual axioms
    if any(ax.get("formal", "N/A") != "N/A" for ax in axioms):
        logic_lines.append("=== Formal Logic Reconstruction ===")
        logic_lines.append("")  # Blank line after title
        for ax in axioms:
            seg = ax.get("segment_index", "N/A")
            eng = ax.get("english", "N/A")
            form = ax.get("formal", "N/A")
            # Only print if we have a real formal statement
            if form != "N/A":
                logic_lines.append(f"({seg}) {eng}")
                logic_lines.append(f"    Formal: {form}")
                logic_lines.append("")
    logic_text = "\n".join([line for line in logic_lines if line.strip()])

    # Build English-based reconstruction text
    # If there are English claims, present them
    english_lines = []
    if any(ax.get("english", "N/A") != "N/A" for ax in axioms):
        english_lines.append("=== English Reconstruction of the Argument ===")
        english_lines.append("")  # Blank line after title
        current_seg = None
        for ax in axioms:
            eng = ax.get("english", "N/A")
            seg_idx = ax.get("segment_index", None)
            if eng != "N/A":
                # Add a small separation when segment index changes to indicate a new point
                if current_seg is not None and seg_idx is not None and seg_idx != current_seg:
                    english_lines.append("")
                english_lines.append(f"- {eng}")
                current_seg = seg_idx
        english_lines.append("")
    english_text = "\n".join([line for line in english_lines if line.strip()])

    return logic_text.strip(), english_text.strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    text_content = []
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text_content.append(soup.get_text(separator=' ').strip())
    return "\n".join(text_content).strip()

def segment_text(text):
    sentences = sent_tokenize(text)
    return list(enumerate(sentences))

def save_to_json(data, filename="output_data.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
