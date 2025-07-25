import json, random, pathlib, re
from collections import defaultdict

random.seed(42)

# ---------------------------
# 1. Load raw discovery triples (with principle_root)
# ---------------------------
with open("discoveries.json", "r") as f:
    discoveries = json.load(f)

def canon_lexeme(title: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", title.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return "_".join(cleaned.split())

# domain -> list of lexeme dicts
domain_lexemes = defaultdict(list)
# domain -> set[root]
domain_roots = defaultdict(set)
# root -> set[domains]
root_domains = defaultdict(set)

for d in discoveries:
    field = d["field"].strip()
    title = d["title"].strip()
    root = d.get("principle_root", d.get("principle", "")).strip().lower()
    if not root:
        continue
    lexeme = canon_lexeme(title)
    domain_lexemes[field].append({"title": title, "lexeme": lexeme, "root": root})
    domain_roots[field].add(root)
    root_domains[root].add(field)

# ---------------------------
# 2. Morphological surface generator
# ---------------------------
def morph_root(root: str) -> str:
    # first alphanumeric token
    tok = re.split(r"\s+", root)[0]
    tok = re.sub(r"[^a-z0-9]", "", tok)
    return tok if tok else "x"

def surface_generator(src_lex: str, tgt_root: str) -> str:
    return f"{morph_root(tgt_root)}_{src_lex}"

# ---------------------------
# 3. Coherence function
# ---------------------------
def coherent(src_lex: str, src_domain: str, tgt_root: str, out_lex: str) -> bool:
    # root must exist in source domain AND in at least one other domain
    if tgt_root not in domain_roots[src_domain]:
        return False
    if len(root_domains[tgt_root]) < 2:
        return False  # not cross-domain
    return out_lex == surface_generator(src_lex, tgt_root)

# Build reverse map lexeme -> domain for negatives
lexeme_domain = {}
for dom, items in domain_lexemes.items():
    for item in items:
        lexeme_domain[item["lexeme"]] = dom

# ---------------------------
# 4. Generate positive examples
# ---------------------------
seed_tokens = [f"<SEED_{i}>" for i in range(128)]
positives = []

for root, domains in root_domains.items():
    if len(domains) < 2:
        continue  # not transferable
    for dom in domains:
        for item in domain_lexemes[dom]:
            out = surface_generator(item["lexeme"], root)
            if coherent(item["lexeme"], dom, root, out):
                positives.append({
                    "seed": random.choice(seed_tokens),
                    "src_lex": item["lexeme"],
                    "tgt_feat": root,
                    "tag": "XFORM",
                    "out_lex": out,
                    "coherent": True
                })

# ---------------------------
# 5. Generate negative examples
# ---------------------------
negatives = []
all_roots = list(root_domains.keys())
all_lexemes = list(lexeme_domain.keys())
domains_list = list(domain_lexemes.keys())

def invalid_root_for_domain(dom, root):
    # either root not in domain OR root appears only in this domain (no cross-domain transfer)
    return (root not in domain_roots[dom]) or (root in domain_roots[dom] and len(root_domains[root]) < 2)

# (a) Unlicensed root samples
while len(negatives) < len(positives):
    dom = random.choice(domains_list)
    lexeme = random.choice([x["lexeme"] for x in domain_lexemes[dom]])
    root = random.choice(all_roots)
    if not invalid_root_for_domain(dom, root):
        continue
    out_wrong = surface_generator(lexeme, root)
    negatives.append({
        "seed": random.choice(seed_tokens),
        "src_lex": lexeme,
        "tgt_feat": root,
        "tag": "XFORM",
        "out_lex": out_wrong,
        "coherent": False
    })

# (b) Corrupt surface forms
for ex in positives[:min(len(positives)//2, 500)]:
    wrong = f"{ex['src_lex']}_{morph_root(ex['tgt_feat'])}"
    if wrong == ex["out_lex"]:
        wrong += "_x"
    negatives.append({
        "seed": random.choice(seed_tokens),
        "src_lex": ex["src_lex"],
        "tgt_feat": ex["tgt_feat"],
        "tag": "XFORM",
        "out_lex": wrong,
        "coherent": False
    })

# ---------------------------
# 6. Combine / split
# ---------------------------
dataset = positives + negatives
random.shuffle(dataset)
split = int(0.85 * len(dataset))
train_set = dataset[:split]
eval_set = dataset[split:]

# ---------------------------
# 7. Save
# ---------------------------
pathlib.Path("data").mkdir(exist_ok=True)

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

write_jsonl("data/train.jsonl", train_set)
write_jsonl("data/eval.jsonl", eval_set)

artifact = {
    "domain_roots": {d: sorted(list(rs)) for d, rs in domain_roots.items()},
    "root_domains": {r: sorted(list(ds)) for r, ds in root_domains.items()}
}
with open("data/artifact.json", "w") as f:
    json.dump(artifact, f, indent=2)

print(f"Done. Positives={len(positives)} Negatives={len(negatives)}")


