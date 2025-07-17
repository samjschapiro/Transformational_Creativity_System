import networkx as nx
import matplotlib.pyplot as plt
from .entailment_score import compute_entailment
import textwrap
import math

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

# need show=false here becasue matplotlib gui can't display off a background process
def visualize_conceptual_space(conceptual_space, save_path=None, entailment_model=None, entailment_tokenizer=None, show=False, min_dist=20):
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