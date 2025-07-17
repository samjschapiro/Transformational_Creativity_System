# File 2: paper_to_space.py
from api_call import generate_response
import json
import re

def get_conceptual_space_from_paper(paper_name: str, english_formalized_claims: str, model=None) -> dict:
    prompt = f"""
      You are a world-class conceptual space engineer.

        Given the formal claims extracted from an academic paper, deconstruct and represent its fully fleshed-out conceptual space as a JSON-FORMATTED directed acyclic graph (DAG).

        - Each node corresponds to a component, assumption, or sub-rule used in the construction of the method.
        - Each edge represents a logical or operational dependency: the target node requires the source node to make sense or function.
        - Walks in the graph must terminate at axioms (self-justifying concepts or primitives).
        - Rules depend, in a directed sense, on axioms and other rules, and can potentially depend on multiple other rules and/or axioms

        The output should be a JSON object of the form:
        {{
        "nodes": [
            {{"id": "n1", "label": "Concept Name", "description": "Short summary", "type": "axiom|rule"}},
            ...
        ],
        "edges": [
            {{"source": "n1", "target": "n2", "relation": "depends_on"}},
            ...
        ]
        }}

        Ensure:
        - Axioms are self-justifying; they do not depend on anything
        - Rules are logically downstream from axioms, but the dependency structure of the graph flows towards the axioms.
        - Depends_on structure is strictly maintained
        - Do not treat the high-level concept itself as an axiom unless it's atomic.
        - Be precise and rigorous; no filler.
        - The output contains ONLY JSON

      Input claims:
      {paper_name, english_formalized_claims}
      """
    response = generate_response(prompt, model=model)
    try:
        content = response.content if hasattr(response, 'content') else str(response)
        # Remove triple backticks and optional 'json' marker
        content = re.sub(r'^```json|^```|```$', '', content.strip(), flags=re.MULTILINE).strip()
        return json.loads(content)
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Raw output:", content)
        return {}