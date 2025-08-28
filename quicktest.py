# tiny snippet you can run in a Python REPL
import gzip, json, statistics as st
path = "data/sample_full_line_heur.jsonl.gz"
notes = []
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        n = rec.get("notes", {})
        if "L_adjusted_to" in n:
            notes.append(n)
print("adjusted:", len(notes), "of 2000")
print("avg L bump:", st.mean((n["L_adjusted_to"] - n["L_adjusted_from"]) for n in notes) if notes else 0.0)
