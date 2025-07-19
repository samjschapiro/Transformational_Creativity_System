from src.api_call import generate_response
import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from src.paper_formalization import compute_entailment
import textwrap
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

def get_conceptual_space(input_str, input_type='paper', model=None, paper_name=None):
    if input_type == 'paper':
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

def get_conceptual_space_from_paper_tr(paper_name: str, english_formalized_claims: str, model=None) -> dict:
    """Return a JSON‑serialisable DAG whose nodes are well‑formed propositions.

    Node types
    ----------
    * "transcendental" – primitive, self‑justifying or constitutive propositions (no outgoing edges).
    * "derived"        – propositions logically supported by other propositions.

    Edge relation
    -------------
    Each edge is a dict of the form {"source": s, "target": t, "relation": "supports"},
    meaning proposition *s* supports or enables proposition *t*.

    The graph must be acyclic and every walk must terminate at a transcendental node.
    """

    prompt = f"""
        You are a world‑class conceptual‑space engineer.

        Given the formal claims extracted from an academic paper, deconstruct and represent its fully fleshed‑out conceptual space as a JSON‑FORMATTED directed acyclic graph (DAG).

        Instructions
        ------------
        1. Each node MUST be a single, well‑formed proposition expressed in English.
        2. Label each node with **exactly one** of the following types:
        • "transcendental" – primitive / constitutive proposition that relies on no other proposition.
        • "derived" – proposition that is supported by one or more other propositions.
        3. Use the edge schema {{"source": "n_i", "target": "n_j", "relation": "supports"}} where the SOURCE proposition supports the TARGET proposition.
        4. Ensure the graph is **acyclic**. Walks must terminate at transcendental nodes.
        5. Do **not** include markdown fences, code blocks, or keys beyond those specified.
        6. Be precise and rigorous; no filler or speculative statements.

        Return **only** a JSON object of the form:
        {{
        "nodes": [
            {{"id": "n1", "label": "Concise name", "description": "Proposition text", "type": "transcendental|derived"}},
            ...
        ],
        "edges": [
            {{"source": "n1", "target": "n2", "relation": "supports"}},
            ...
        ]
        }}

        INPUT (paper name and extracted claims):
        {paper_name}
        ---
        {english_formalized_claims}
        """

    response = generate_response(prompt, model=model)
    content = getattr(response, 'content', str(response)).strip()

    # Remove optional markdown fences that some models include
    content = re.sub(r"^```[a-zA-Z0-9]*|```$", "", content, flags=re.MULTILINE).strip()

    try:
        return json.loads(content)
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Raw output:", content)
        return {}


def get_conceptual_space_from_paper(paper_name: str, english_formalized_claims: str, model=None) -> dict:
    prompt = f"""
      You are a world-class conceptual space engineer.

        Given the formal claims extracted from an academic paper, deconstruct and represent its fully fleshed-out conceptual space as a JSON-FORMATTED directed acyclic graph (DAG).

        - Each node corresponds to a component, assumption, or sub-rule used in the construction of the method.
        - Each edge represents a logical or operational dependency: the target node requires the source node to make sense or function.
        - Walks in the graph must terminate at axioms (self-justifying concepts or primitives).
        - Rules are composed from axioms and other rules.

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
    
def visualize_conceptual_space(conceptual_space, save_path=None, entailment_model=None, entailment_tokenizer=None, show=True, min_dist=20):
    """
    Visualize a conceptual space (DAG) as a static matplotlib graph.
    Nodes are spaced at least min_dist apart, all info is shown in the node, and edges show relation and entailment score.
    Args:
        conceptual_space (dict): The conceptual space with 'nodes' and 'edges'.
        save_path (str or None): If provided, save the plot to this path.
        entailment_model, entailment_tokenizer: Optionally provide preloaded model/tokenizer for efficiency.
        show (bool): Whether to display the plot.
        min_dist (int): Minimum distance between nodes in layout.
    """
    G = nx.DiGraph()
    node_info = {}
    node_labels = {}
    node_label_lens = []
    for node in conceptual_space.get('nodes', []):
        label_lines = [f"{k}: {v}" for k, v in node.items()]
        label = '\n'.join([textwrap.fill(line, 30) for line in label_lines])
        node_attrs = dict(node)
        node_attrs.pop('label', None)
        G.add_node(node['id'], label=label, **node_attrs)
        node_info[node['id']] = node
        node_labels[node['id']] = label
        node_label_lens.append(len(label))
    edge_labels = {}
    for edge in conceptual_space.get('edges', []):
        src = edge['source']
        tgt = edge['target']
        relation = edge.get('relation', '')
        premise = node_info[src].get('description', node_info[src].get('label', ''))
        hypothesis = node_info[tgt].get('description', node_info[tgt].get('label', ''))
        score = compute_entailment(
            premise, hypothesis,
            tokenizer_=entailment_tokenizer, model_=entailment_model
        ) if entailment_model and entailment_tokenizer else None
        label = f"{relation}\nEntail: {score:.2f}" if score is not None else relation
        G.add_edge(src, tgt, relation=relation, entailment=score)
        edge_labels[(src, tgt)] = label
    # Use spring layout to get initial positions
    pos = nx.spring_layout(G, seed=42)
    # Enforce minimum distance between nodes (inline logic)
    nodes = list(pos.keys())
    changed = True
    max_iter = 100
    iter_count = 0
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                x1, y1 = pos[n1]
                x2, y2 = pos[n2]
                dx, dy = x2 - x1, y2 - y1
                dist = math.hypot(dx, dy)
                if dist < min_dist and dist > 0:
                    move = (min_dist - dist) / 2
                    angle = math.atan2(dy, dx)
                    shift_x = move * math.cos(angle)
                    shift_y = move * math.sin(angle)
                    pos[n1][0] -= shift_x
                    pos[n1][1] -= shift_y
                    pos[n2][0] += shift_x
                    pos[n2][1] += shift_y
                    changed = True
    # Dynamically adjust node size based on label length and graph size
    avg_label_len = sum(node_label_lens) / len(node_label_lens) if node_label_lens else 1
    base_node_size = 6000
    if len(G.nodes) > 8:
        base_node_size = max(2000, 6000 - 300 * (len(G.nodes) - 8))
    node_size = [max(base_node_size, 80 * len(node_labels[n])) for n in G.nodes]
    plt.figure(figsize=(max(14, len(G.nodes)*2), max(10, len(G.nodes)*1.5)))
    nx.draw(
        G, pos, with_labels=False, node_color='lightblue', node_size=node_size, font_size=10, edgecolors='black'
    )
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle='-|>', arrowsize=30, width=3, edge_color='black', connectionstyle='arc3,rad=0.08'
    )
    ax = plt.gca()
    for node, (x, y) in pos.items():
        label = node_labels[node]
        ax.text(
            x, y, label, ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
            zorder=10
        )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9, label_pos=0.6)
    plt.title('Conceptual Space Graph')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close() 
