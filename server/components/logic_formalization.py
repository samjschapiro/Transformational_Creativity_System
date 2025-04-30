import json
import re
from sympy.logic.boolalg import sympify, simplify_logic

from api_call import generate_response

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