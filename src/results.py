import json, os

PATH = "docs/results.json"

def save_metrics(section: str, metrics: dict):
    os.makedirs("docs", exist_ok=True)
    data = {}
    if os.path.exists(PATH):
        with open(PATH) as f:
            data = json.load(f)
    data[section] = metrics
    with open(PATH, "w") as f:
        json.dump(data, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))