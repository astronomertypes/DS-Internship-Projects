import json

with open("docs/results.json") as f:
    data = json.load(f)

lines = ["# Results", "", "Auto-generated from `docs/results.json`. Do not edit by hand.", ""]
for section, metrics in data.items():
    lines += [f"## {section}", "", "| Metric | Value |", "|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {round(v, 4) if isinstance(v, float) else v} |")
    lines.append("")
with open("docs/results.md", "w") as f:
    f.write("\n".join(lines))
print("Wrote docs/results.md")