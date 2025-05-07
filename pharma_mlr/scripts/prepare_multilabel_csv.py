#!/usr/bin/env python3
import json, csv, ast
import pandas as pd

# 1) Load your original CSV
df = pd.read_csv("violation_training_dataset.csv")

# 2) Load the rule‐config keys
with open("configs/violation_rules.json") as f:
    RULE_KEYS = set(json.load(f).keys())

# 3) Build new columns
records = []
for _, row in df.iterrows():
    text = row["text"]
    binary = 1 if row["violation_label"] == "violation" else 0
    
    # parse original violation_reason (might be comma‑sep or JSON list)
    orig = row.get("violation_reason", "")
    if pd.isna(orig) or not orig.strip():
        rules = []
    else:
        # try JSON decode or split on comma 
        try:
            rules = list(ast.literal_eval(orig))
        except:
            rules = [r.strip() for r in orig.split(",") if r.strip()]
    # keep only known keys
    rules = [r for r in rules if r in RULE_KEYS]
    
    records.append({
        "text": text,
        "label": binary,
        # store as JSON array literal
        "fda_rules": json.dumps(rules)
    })

# 4) Write out the new CSV
with open("violation_training_multilabel.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text","label","fda_rules"])
    writer.writeheader()
    writer.writerows(records)

print("Wrote violation_training_multilabel.csv with", len(records), "rows.")
