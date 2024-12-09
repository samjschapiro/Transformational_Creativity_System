import os
import json
import re
from api_call import generate_response

def extract_json_from_response(response_content):
    try:
        clean_content = re.sub(r'```json|```', '', response_content).strip()
        clean_content = clean_content.replace('```', '').strip()
        clean_content = clean_content.strip('`')
        json_match = re.search(r'{.*}', clean_content, re.DOTALL)
        if not json_match:
            print("Skipping: No valid JSON found in response.")
            return None
        return json.loads(json_match.group(0))
    except (json.JSONDecodeError, Exception) as e:
        print(f"Skipping JSON decoding error: {e}")
        return None

def append_response_to_file(response_content, filename):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(response_content.strip() + "\n---END-OF-SEGMENT---\n")

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

def extract_claims(segments, output_file="all_responses.txt"):
    if os.path.exists(output_file):
        os.remove(output_file)

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
            append_response_to_file(json.dumps({
                "segment_index": seg_idx,
                "claims": [],
                "arguments": [],
                "examples": [],
                "decorative": []
            }), filename=output_file)
            continue

        append_response_to_file(response.content, filename=output_file)

    return parse_combined_responses(filename=output_file)
