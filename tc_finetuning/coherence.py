# coherence.py
import json

with open("data/artifact.json") as f:
    art = json.load(f)

lexeme_domain = art["lexeme_domain"]
feature_domain = art["feature_domain"]
domains_features = art["domains_features"]
corr_pairs = set(tuple(p) for p in art["correspondences"])
# Add reverse edges
C = corr_pairs | set((b,a) for a,b in corr_pairs)

def feature_root(feat: str) -> str:
    return feat.split("_")[0]

def surface_generator(src_lex, tgt_feat):
    return f"{feature_root(tgt_feat)}_{src_lex}"

def coherent(src_lex, tgt_feat, out_lex):
    Ds = lexeme_domain.get(src_lex)
    Dt = feature_domain.get(tgt_feat)
    if Ds is None or Dt is None or Ds == Dt: return False
    linked = any(
        (tgt_feat, psi) in C or (psi, tgt_feat) in C
        for psi in domains_features[Ds]
    )
    if not linked: return False
    return out_lex == surface_generator(src_lex, tgt_feat)

# Example usage
if __name__ == "__main__":
    print(coherent("universal_gravitation", "mass_conservation", "mass_universal_gravitation"))  # True
    print(coherent("universal_gravitation", "natural_selection", "selection_universal_gravitation"))  # False
