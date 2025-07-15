"""
Build a dependency-aware DAG from scientific papers using NLI models to measure
directional entailment between claims. This creates a reasoning/proof tree rather
than a topic similarity graph.

Uses claim extraction and NLI models to determine if claims in paper P are
entailed or narrowly subsumed by claims in reference R, creating directed edges
that represent logical dependency rather than topical similarity.
"""
from __future__ import annotations
API_KEY = "c7Xd3p89Ta2T7XFGTzPSi7u3kDRMsHAI8RwlXlFe"

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import requests
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# OpenRouter integration
from openai import OpenAI

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-2.5-flash-preview"

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# API Configuration
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
DEFAULT_FIELDS = "title,year,abstract,fieldsOfStudy"
SLEEP = 0.5  # seconds between API calls (reduced for faster processing)

# Dependency scoring parameters
TOP_K_REFERENCES = 5  # Only recurse on top-k most dependent papers
DEP_THRESHOLD = 0.05  # Minimum dependency score to consider (capture some dependencies)
SAMPLE_K_CLAIMS = 4  # Number of claims to sample per paper
CONTRADICTION_THRESHOLD = 0.7  # Threshold for contradiction edges (adjusted for research dependencies)

# Model selection - Using SciNLI models for scientific literature
USE_SCINLI_MODEL = True  # Use SciNLI-trained models for scientific literature
CLAIM_EXTRACTION_MODEL = "allenai/scibert_scivocab_uncased"  # SciBERT for scientific claim extraction
NLI_MODEL_SCINLI = "tasksource/deberta-small-long-nli"  # SciNLI-trained DeBERTa
NLI_MODEL_SCIFACT = "MilosKosRad/DeBERTa-v3-large-SciFact"  # Fallback SciFact model
NLI_MODEL_GENERAL = "facebook/bart-large-mnli"  # General fallback
FALLBACK_EMBEDDING_MODEL = "gsarti/scibert-nli"  # SciBERT-NLI for embeddings

SEED_PAPERS = [ # DOIs from Xplor - test run
    "2504.18687"  # 1 paper for analysis
]

# Global model cache
models = {}

# Global proposition cache for performance
proposition_cache = {}

def get_model(model_type: str):
    """Lazy load and cache models."""
    global models
    if model_type not in models:
        print(f"[model] Loading {model_type}...")
        if model_type == "claim_extractor":
            # Use SciBERT for scientific claim extraction
            tokenizer = AutoTokenizer.from_pretrained(CLAIM_EXTRACTION_MODEL)
            model = AutoModel.from_pretrained(CLAIM_EXTRACTION_MODEL)
            if torch.cuda.is_available():
                model = model.cuda()
            models[model_type] = (tokenizer, model.eval())
            print(f"[model] Loaded SciBERT claim extraction model: {CLAIM_EXTRACTION_MODEL}")
        elif model_type == "nli":
            if USE_SCINLI_MODEL:
                try:
                    print(f"[model] Attempting to load {NLI_MODEL_SCINLI}")
                    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_SCINLI)
                    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_SCINLI)
                    if torch.cuda.is_available():
                        model = model.cuda()
                    models[model_type] = (tokenizer, model.eval())
                    print(f"[model] Successfully loaded {NLI_MODEL_SCINLI}")
                except Exception as e:
                    print(f"[model] Failed to load {NLI_MODEL_SCINLI}: {e}")
                    print(f"[model] Falling back to {NLI_MODEL_SCIFACT}")
                    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_SCIFACT)
                    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_SCIFACT)
                    if torch.cuda.is_available():
                        model = model.cuda()
                    models[model_type] = (tokenizer, model.eval())
            else:
                tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_SCIFACT)
                model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_SCIFACT)
                if torch.cuda.is_available():
                    model = model.cuda()
                models[model_type] = (tokenizer, model.eval())
        elif model_type == "embedding":
            # Use SciBERT-NLI model for scientific literature embeddings
            models[model_type] = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
            print(f"[model] Loaded scientific embeddings model: {FALLBACK_EMBEDDING_MODEL}")
    return models[model_type]


def extract_propositions_with_llm(abstract: str, max_propositions: int = 6) -> List[str]:
    """
    Extract key propositions/claims from research paper abstract using Gemini 2.5 Flash Preview via OpenRouter.
    This replaces the complex rule-based extraction with a simple, effective LLM approach.
    """
    if not abstract:
        return []
    
    prompt = f"""Extract the key scientific propositions/claims from this research paper abstract. Focus on:
1. Main findings and results
2. Key contributions and discoveries  
3. Important conclusions
4. Novel insights or evidence presented

Return ONLY a JSON list of strings, where each string is a clear, standalone proposition. 
Limit to {max_propositions} most important propositions.
Avoid methodological descriptions - focus on what was found/proven/shown.

Abstract: {abstract}

JSON:"""

    try:
        response = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/s2-crawler",
                "X-Title": "Scientific Paper Dependency Analysis",
            },
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=800
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handle cases where LLM adds explanation)
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "[" in content and "]" in content:
            # Find the JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            content = content[start:end]
        
        # Parse JSON
        propositions = json.loads(content)
        
        # Validate and clean propositions
        valid_propositions = []
        for prop in propositions:
            if isinstance(prop, str) and len(prop.strip()) > 10:
                # Clean up the proposition
                clean_prop = prop.strip()
                if clean_prop.endswith('.'):
                    clean_prop = clean_prop[:-1]
                valid_propositions.append(clean_prop)
        
        return valid_propositions[:max_propositions]
        
    except Exception as e:
        print(f"[llm] Error extracting propositions with LLM: {e}")
        print(f"[llm] Falling back to rule-based extraction")
        return extract_claims_fallback(abstract, max_propositions)


def extract_claims_fallback(text: str, max_claims: int = 6) -> List[str]:
    """Fallback rule-based claim extraction if LLM fails."""
    if not text:
        return []
    
    # Simple sentence-based extraction as fallback
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    finding_indicators = [
        'show', 'demonstrates', 'reveals', 'finds', 'indicates', 'suggests', 
        'evidence', 'results', 'achieves', 'improves', 'reduces', 'increases',
        'outperforms', 'enables', 'provides', 'supports', 'confirms'
    ]
    
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 5 or len(sent.split()) > 40:
            continue
        if any(indicator in sent.lower() for indicator in finding_indicators):
            claims.append(sent)
    
    return claims[:max_claims]


def extract_claims(text: str, max_claims: int = 6) -> List[str]:
    """Extract propositions using LLM-based approach with fallback and caching."""
    # Use text hash for caching
    text_hash = hash(text[:1000])  # Hash first 1000 chars for cache key
    
    if text_hash in proposition_cache:
        return proposition_cache[text_hash]
    
    propositions = extract_propositions_with_llm(text, max_claims)
    proposition_cache[text_hash] = propositions
    return propositions


# Removed complex rule-based functions - now using LLM-based proposition extraction


def compute_nli_entailment(premise: str, hypothesis: str) -> Tuple[float, float, float]:
    """
    Compute NLI scores for a premise-hypothesis pair.
    Returns (entailment_prob, neutral_prob, contradiction_prob)
    """
    tokenizer, model = get_model("nli")
    
    # Prepare input
    inputs = tokenizer(
        premise, 
        hypothesis, 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    )
    
    # Move inputs to same device as model
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()
        probs = F.softmax(logits, dim=-1)
        
        # Move back to CPU for numpy operations
        if probs.is_cuda:
            probs = probs.cpu()
    
    # Map to standard labels based on model type
    if len(probs) == 3:
        if USE_SCINLI_MODEL:
            # SciNLI models use standard NLI mapping: entailment=0, neutral=1, contradiction=2
            entailment_prob = probs[0].item()
            neutral_prob = probs[1].item()
            contradiction_prob = probs[2].item()
        else:
            # SciFact mapping: support=1, contradict/no_evidence=0, unclear=2
            entailment_prob = probs[1].item()
            contradiction_prob = probs[0].item() 
            neutral_prob = probs[2].item()
        
        return entailment_prob, neutral_prob, contradiction_prob
    
    # Fallback for binary classifiers
    return probs[-1].item(), 0.0, 1.0 - probs[-1].item()


def compute_research_dependency(premise: str, hypothesis: str) -> Tuple[float, float, float]:
    """
    Compute research dependency scores more suitable for scientific papers.
    Uses both NLI and semantic similarity for better research paper analysis.
    Returns (support_prob, neutral_prob, contradiction_prob)
    """
    # Get NLI scores
    nli_support, nli_neutral, nli_contradict = compute_nli_entailment(premise, hypothesis)
    
    # For research papers, we need to be less strict about "contradiction"
    # SciFact tends to label different approaches as contradictory, but in research
    # they're often just different/complementary approaches
    
    # Use semantic similarity as a moderating factor
    # Use cached embedding model to avoid repeated loading
    try:
        embedding_model = get_model("embedding")
        embeddings = embedding_model.encode([premise, hypothesis])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    except:
        similarity = 0.0
    
    # Adjust scores based on semantic similarity
    # If claims are semantically similar, reduce contradiction confidence
    similarity_factor = max(0, similarity - 0.3)  # Only positive similarity above 0.3
    
    # Rebalance the scores
    adjusted_contradict = nli_contradict * (1 - similarity_factor * 0.7)
    adjusted_support = nli_support + (similarity_factor * 0.3)
    adjusted_neutral = nli_neutral + (similarity_factor * 0.4)
    
    # Normalize to sum to 1
    total = adjusted_support + adjusted_neutral + adjusted_contradict
    if total > 0:
        adjusted_support /= total
        adjusted_neutral /= total
        adjusted_contradict /= total
    
    return adjusted_support, adjusted_neutral, adjusted_contradict


def dep_score_detailed(child_abs: str, parent_abs: str, sample_k: int = SAMPLE_K_CLAIMS) -> Tuple[float, float, Dict]:
    """
    Compute dependency score with detailed comparison information.
    Returns (dependency_score, contradiction_score, comparison_details)
    """
    child_claims = extract_claims(child_abs)[:sample_k]
    parent_claims = extract_claims(parent_abs)[:sample_k]
    
    if not child_claims or not parent_claims:
        print(f"[dep] No claims extracted, using fallback")
        fallback_score = weeds_precision_fallback(child_abs, parent_abs)
        return fallback_score, 0.0, {"method": "fallback", "child_propositions": [], "parent_propositions": []}
    
    print(f"[dep] Comparing {len(child_claims)} child claims with {len(parent_claims)} parent claims")
    
    entailment_scores = []
    contradiction_scores = []
    comparison_details = []
    
    # Compute pairwise research dependency scores
    for i, c_claim in enumerate(child_claims):
        for j, p_claim in enumerate(parent_claims):
            # Use research-specific dependency scoring
            ent_prob, _, cont_prob = compute_research_dependency(p_claim, c_claim)
            entailment_scores.append(ent_prob)
            contradiction_scores.append(cont_prob)
            
            # Store detailed comparison
            comparison_details.append({
                "child_claim_idx": i,
                "parent_claim_idx": j,
                "child_claim": c_claim,
                "parent_claim": p_claim,
                "entailment_score": float(ent_prob),
                "contradiction_score": float(cont_prob)
            })
    
    # Aggregate scores: mean of top-k entailments
    top_entailments = sorted(entailment_scores, reverse=True)[:sample_k]
    top_contradictions = sorted(contradiction_scores, reverse=True)[:sample_k]
    
    dep_score = float(np.mean(top_entailments)) if top_entailments else 0.0
    cont_score = float(np.mean(top_contradictions)) if top_contradictions else 0.0
    
    print(f"[dep] Final scores: Dependency={dep_score:.3f}, Contradiction={cont_score:.3f}")
    
    # Find best matching pairs - always store top evidence pairs
    # Use the same threshold as DEP_THRESHOLD for consistency
    significant_entailments = [x for x in comparison_details if x["entailment_score"] > DEP_THRESHOLD]
    significant_contradictions = [x for x in comparison_details if x["contradiction_score"] > DEP_THRESHOLD]
    
    # Always store top-3 pairs even if below threshold to ensure evidence exists
    all_entailments = sorted(comparison_details, key=lambda x: x["entailment_score"], reverse=True)[:3]
    all_contradictions = sorted(comparison_details, key=lambda x: x["contradiction_score"], reverse=True)[:3]
    
    # Combine significant pairs with top pairs (remove duplicates)
    best_entailments = sorted(significant_entailments, key=lambda x: x["entailment_score"], reverse=True)[:3]
    best_contradictions = sorted(significant_contradictions, key=lambda x: x["contradiction_score"], reverse=True)[:3]
    
    # If no significant pairs, use top pairs as evidence
    if not best_entailments:
        best_entailments = all_entailments
    if not best_contradictions:
        best_contradictions = all_contradictions
    
    details = {
        "method": "llm_nli",
        "child_propositions": child_claims,
        "parent_propositions": parent_claims,
        "total_comparisons": len(comparison_details),
        "significant_supports": len(significant_entailments),
        "significant_contradicts": len(significant_contradictions),
        "best_supporting_pairs": best_entailments,
        "best_contradicting_pairs": best_contradictions
    }
    
    return dep_score, cont_score, details


def dep_score(child_abs: str, parent_abs: str, sample_k: int = SAMPLE_K_CLAIMS) -> Tuple[float, float]:
    """
    Compute dependency score: how much child's claims are entailed by parent's claims.
    Returns (dependency_score, contradiction_score) - backwards compatible version
    """
    dep, cont, _ = dep_score_detailed(child_abs, parent_abs, sample_k)
    return dep, cont


def weeds_precision_fallback(child_text: str, parent_text: str) -> float:
    """
    Fallback to Weeds Precision using SciBERT embeddings when NLI is unavailable.
    Measures how much of child's vector lies in parent's direction.
    """
    # Use SciBERT for scientific text embeddings
    model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
    
    # Get SciBERT embeddings
    embeddings = model.encode([child_text, parent_text])
    child_emb = embeddings[0]
    parent_emb = embeddings[1]
    
    # Compute projection ratio (simplified Weeds Precision)
    dot_product = np.dot(child_emb, parent_emb)
    parent_norm_sq = np.dot(parent_emb, parent_emb)
    
    if parent_norm_sq > 0:
        projection = dot_product / parent_norm_sq
        return float(max(0, projection))  # Clamp to [0, 1]
    return 0.0


def _normalize_pid(pid: str) -> str:
    """Return a valid Semantic Scholar ID."""
    if ":" in pid:
        return pid
    if re.match(r"^\d+\.\d+(v\d+)?$", pid):
        return f"ARXIV:{pid}"
    if re.match(r"^[0-9a-f]{40}$", pid):
        return pid
    return pid


def safe_get(url: str, **kwargs) -> Optional[requests.Response]:
    """GET with 429 backoff and 404 skip."""
    backoff = SLEEP
    for attempt in range(1, 4):
        print(f"[http] GET {url} (attempt {attempt})")
        r = requests.get(url, **kwargs)
        if r.status_code == 429:
            print(f"[rate] 429 from {url}, backing off {backoff}s")
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 404:
            print(f"[warn] 404 Not Found: {url}")
            return None
        r.raise_for_status()
        return r
    print(f"[error] Failed GET after retries: {url}")
    return None


def fetch_metadata(pid: str) -> Dict:
    """Fetch paper metadata."""
    print(f"[meta] Fetching metadata for {pid}")
    norm = _normalize_pid(pid)
    url = f"{BASE_URL}/{norm}"
    r = safe_get(url, headers={"x-api-key": API_KEY}, params={"fields": DEFAULT_FIELDS})
    time.sleep(SLEEP)
    data = r.json() if r else {}
    return data


def fetch_references_with_abstracts(pid: str, max_refs: int = 10) -> List[Dict]:
    """Fetch references and their abstracts for dependency analysis."""
    print(f"[refs] Fetching references for dependency analysis: {pid}")
    
    refs_ids = []
    offset = 0
    
    # Paginated references endpoint
    while len(refs_ids) < max_refs:
        norm = _normalize_pid(pid)
        url = f"{BASE_URL}/{norm}/references"
        params = {"fields": "citedPaper.paperId", "offset": offset, "limit": min(100, max_refs - len(refs_ids))}
        r = safe_get(url, headers={"x-api-key": API_KEY}, params=params)
        time.sleep(SLEEP)
        if not r:
            break
        chunk = r.json().get("data", [])
        if not chunk:
            break
        for c in chunk:
            cited = c.get("citedPaper", {})
            if cited.get("paperId"):
                refs_ids.append(cited["paperId"])
                if len(refs_ids) >= max_refs:
                    break
        if len(chunk) < 100:
            break
        offset += 100
    
    # Fetch abstracts for references
    refs_with_abstracts = []
    for i, ref_id in enumerate(refs_ids):
        print(f"[refs] Fetching abstract {i+1}/{len(refs_ids)}: {ref_id}")
        ref_meta = fetch_metadata(ref_id)
        if ref_meta.get("abstract"):
            refs_with_abstracts.append({
                "paperId": ref_id,
                "title": ref_meta.get("title", ""),
                "abstract": ref_meta.get("abstract", ""),
                "year": ref_meta.get("year", 0)
            })
    
    return refs_with_abstracts


def filter_references_by_dependency(child_meta: Dict, refs: List[Dict]) -> List[Tuple[Dict, float, float, Dict]]:
    """
    Filter references using dependency scoring.
    Returns list of (reference, dependency_score, contradiction_score, comparison_details) tuples.
    """
    child_abstract = child_meta.get("abstract", "")
    if not child_abstract:
        print(f"[dep] No child abstract, skipping dependency filtering")
        return []
    
    child_year = child_meta.get("year", 9999)
    
    # Filter refs that have abstracts and are older (ensuring DAG property)
    valid_refs = [
        ref for ref in refs 
        if ref.get("abstract") and ref.get("year", 0) < child_year
    ]
    
    if not valid_refs:
        print(f"[dep] No valid references for dependency analysis")
        return []
    
    print(f"[dep] Computing dependency scores for {len(valid_refs)} references")
    
    scored_refs = []
    all_scores = []  # Track all scores for analysis
    for ref in tqdm(valid_refs, desc="Computing dependencies"):
        dep, cont, details = dep_score_detailed(child_abstract, ref["abstract"])
        all_scores.append((dep, cont))
        title = ref.get("title", "Unknown")[:50]
        print(f"[dep] Score {dep:.3f} (cont: {cont:.3f}) - {title}")
        if dep >= DEP_THRESHOLD:
            scored_refs.append((ref, dep, cont, details))
    
    # Show score distribution
    if all_scores:
        dep_scores = [s[0] for s in all_scores]
        cont_scores = [s[1] for s in all_scores]
        print(f"[dep] Dependency scores - Max: {max(dep_scores):.3f}, Min: {min(dep_scores):.3f}, Avg: {np.mean(dep_scores):.3f}")
        print(f"[dep] Contradiction scores - Max: {max(cont_scores):.3f}, Min: {min(cont_scores):.3f}, Avg: {np.mean(cont_scores):.3f}")
    
    # Sort by dependency score (descending)
    scored_refs.sort(key=lambda x: x[1], reverse=True)
    
    # Take top-k
    top_refs = scored_refs[:TOP_K_REFERENCES]
    print(f"[dep] Selected {len(top_refs)} references above threshold {DEP_THRESHOLD}")
    
    return top_refs


# Removed extract_concept function - no longer needed


def recurse(pid: str, depth: int, G: nx.DiGraph, visited: Set[str]) -> None:
    """Recursively build dependency DAG."""
    print(f"[recurse] PID={pid}, depth={depth}")
    start_time = time.time()
    meta = fetch_metadata(pid)
    node_id = meta.get("paperId")
    if not node_id or node_id in visited:
        print(f"[skip] PID={pid} already visited or missing")
        return
    visited.add(node_id)
    
    # Add node with attributes
    title = meta.get("title", "")
    year = str(meta.get("year", ""))
    full_abstract = meta.get("abstract", "")
    
    # Extract propositions for this paper
    prop_start = time.time()
    propositions = extract_claims(full_abstract) if full_abstract else []
    prop_time = time.time() - prop_start
    print(f"[props] Extracted {len(propositions)} propositions for {title[:50]}... ({prop_time:.1f}s)")
    if propositions:
        print(f"[props] Sample proposition: {propositions[0][:100]}...")
    
    G.add_node(
        node_id,
        title=title,
        year=year,
        abstract=full_abstract[:300],  # Shorter truncated abstract for reference
        propositions_json=json.dumps(propositions),  # Store as JSON string for GraphML compatibility
        propositions=propositions,  # Keep as list for processing
        proposition_count=len(propositions)
    )
    print(f"[node] Added {node_id}: {title}")
    
    if depth == 0:
        print(f"[depth] Reached 0, stop recursion for {node_id}")
        return
    
    # Fetch references
    refs = fetch_references_with_abstracts(node_id)
    if not refs:
        print(f"[recurse] No references found for {node_id}")
        return
    
    # Filter by dependency
    dependent_refs = filter_references_by_dependency(meta, refs)
    
    print(f"[recurse] Processing {len(dependent_refs)} dependent refs for {node_id}")
    for ref, dep_score, cont_score, comparison_details in dependent_refs:
        rid = ref.get("paperId")
        if rid and rid not in visited:
            # Add directed edge from child to parent (dependency direction)
            # This represents "child depends on parent" (present → past)
            # Convert lists to JSON strings for GraphML compatibility
            G.add_edge(
                node_id,
                rid, 
                dependency=float(dep_score),
                contradiction=float(cont_score),
                edge_type="supports" if dep_score > cont_score else "contradicts",
                comparison_method=comparison_details.get("method", "unknown"),
                child_propositions_json=json.dumps(comparison_details.get("child_propositions", [])),
                parent_propositions_json=json.dumps(comparison_details.get("parent_propositions", [])),
                best_supporting_pairs_json=json.dumps(comparison_details.get("best_supporting_pairs", [])),
                best_contradicting_pairs_json=json.dumps(comparison_details.get("best_contradicting_pairs", [])),
                total_comparisons=comparison_details.get("total_comparisons", 0),
                significant_supports=comparison_details.get("significant_supports", 0),
                significant_contradicts=comparison_details.get("significant_contradicts", 0)
            )
            recurse(rid, depth - 1, G, visited)
    
    total_time = time.time() - start_time
    print(f"[timing] Completed {node_id} in {total_time:.1f}s")


def verify_dag(G: nx.DiGraph) -> bool:
    """Verify the graph is a DAG and report any cycles."""
    if nx.is_directed_acyclic_graph(G):
        print("[dag] Graph is acyclic ✓")
        return True
    else:
        print("[dag] WARNING: Graph contains cycles!")
        cycles = list(nx.simple_cycles(G))
        print(f"[dag] Found {len(cycles)} cycles")
        for i, cycle in enumerate(cycles[:5]):  # Show first 5 cycles
            print(f"[dag] Cycle {i+1}: {' -> '.join(cycle)}")
        return False


def compute_graph_stats(G: nx.DiGraph) -> Dict:
    """Compute interesting statistics about the dependency graph."""
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(G),
        "avg_dependency": np.mean([d["dependency"] for _, _, d in G.edges(data=True)]) if G.edges else 0,
        "contradiction_edges": sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "contradicts"),
        "support_edges": sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "supports"),
    }
    
    if stats["is_dag"]:
        # Compute topological generations
        topo_gens = list(nx.topological_generations(G))
        stats["topological_levels"] = len(topo_gens)
        stats["nodes_per_level"] = [len(gen) for gen in topo_gens]
    
    return stats


if __name__ == "__main__":
    print(f"[init] Starting dependency-aware DAG builder")
    print(f"[init] NLI Model: {'SciNLI' if USE_SCINLI_MODEL else 'SciFact'}")
    print(f"[init] Top-K: {TOP_K_REFERENCES}, Dep Threshold: {DEP_THRESHOLD}")
    
    G = nx.DiGraph()
    visited: Set[str] = set()
    depth = 2  # Full 2-level recursion
    
    try:
        for sid in tqdm(SEED_PAPERS, desc="Seed papers"):
            tqdm.write(f"▶ Starting seed {sid}")
            recurse(sid, depth, G, visited)
            tqdm.write(f"✔ Finished seed {sid}")
    except KeyboardInterrupt:
        print("[interrupt] KeyboardInterrupt received. Saving current graph state...")
    
    # Verify DAG property
    verify_dag(G)
    
    # Compute and display stats
    stats = compute_graph_stats(G)
    print("\n[stats] Dependency Graph Statistics:")
    for key, value in stats.items():
        print(f"[stats] {key}: {value}")
    
    # Save outputs
    outdir = Path("./out")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create a clean copy for GraphML export (remove list attributes that GraphML can't handle)
    G_clean = G.copy()
    for node_id in G_clean.nodes():
        # Remove the list attributes, keep the JSON versions for GraphML compatibility
        if 'propositions' in G_clean.nodes[node_id]:
            del G_clean.nodes[node_id]['propositions']
    
    # Save graph files
    nx.write_graphml(G_clean, outdir / "dependency_dag.graphml")
    with open(outdir / "dependency_dag.json", "w") as f:
        json.dump(nx.readwrite.json_graph.node_link_data(G), f, indent=2)
    
    # Save statistics
    with open(outdir / "dependency_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n[done] Built dependency DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"[done] Saved to: dependency_dag.graphml, dependency_dag.json, dependency_stats.json") 