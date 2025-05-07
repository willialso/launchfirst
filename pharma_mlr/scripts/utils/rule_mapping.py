# scripts/utils/rule_mapping.py

def map_confidence_to_rule(score: float) -> str:
    if score >= 0.8:
        return "202.1(e)(1-6) — Side effect balance"
    elif score >= 0.6:
        return "202.1(e)(12-13) — Reference authority"
    elif score >= 0.4:
        return "202.1(a-b) — Ingredient misrepresentation"
    elif score >= 0.2:
        return "202.1(c-d) — Dosage disclosure"
    else:
        return "No clear CFR rule triggered"
