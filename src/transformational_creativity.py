import json
import random

def propose_transformational_idea(graph_json, axioms):
    """
    Proposes a transformational idea that modifies the axioms based on the graph structure.

    Args:
        graph_json (str): JSON string describing the graph. Should contain nodes and edges.
        axioms (list of dict): Each dict should have 'axiom' (str) and 'incoming' (int).

    Returns:
        dict: A proposal describing the transformational idea.
    """
    graph = json.loads(graph_json)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Example transformation: Find the axiom with the most incoming nodes and suggest splitting it
    if not axioms:
        return {"proposal": "No axioms to transform."}

    max_incoming = max(axioms, key=lambda x: x['incoming'])
    proposal = {
        "action": "split_axiom",
        "target_axiom": max_incoming['axiom'],
        "reason": f"Axiom '{max_incoming['axiom']}' has the most incoming nodes ({max_incoming['incoming']}).",
        "suggestion": f"Consider splitting '{max_incoming['axiom']}' into more specific sub-axioms to reduce complexity."
    }

    # Optionally, propose merging two axioms with low incoming nodes
    low_incoming_axioms = [a for a in axioms if a['incoming'] == min(ax['incoming'] for ax in axioms)]
    if len(low_incoming_axioms) >= 2:
        a1, a2 = random.sample(low_incoming_axioms, 2)
        proposal = {
            "action": "merge_axioms",
            "target_axioms": [a1['axiom'], a2['axiom']],
            "reason": f"Axioms '{a1['axiom']}' and '{a2['axiom']}' have the fewest incoming nodes.",
            "suggestion": f"Consider merging '{a1['axiom']}' and '{a2['axiom']}' to simplify the graph."
        }

    return proposal