"""Fix the TP8 routing cutoff before evaluating the independent full sweep."""
import json
import math
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(Path("active-run.txt").read_text().strip())
pilot = run / "pilot"
config = json.loads((pilot / "config.json").read_text())
samples = {}
for requested in config["sizes"]:
    record = json.loads((pilot / f"m{requested:04d}.json").read_text())
    assert record["m"] == requested and record["rounds"] == config["rounds"]
    methods = record["methods"]
    assert set(methods) == {"aiter", "direct", "sharded"}
    for name, stats in methods.items():
        assert len(stats["rank_us"]) == 8
        assert not any(stats["errors"])
        assert stats["us"] > 0 and math.isfinite(stats["us"])
        assert math.isclose(stats["us"], statistics.median(max(s) for s in zip(*stats["rank_us"])))
    samples[requested] = {n: s["us"] for n,s in methods.items()}
candidates = sorted({0, *samples})
scores = [(statistics.mean(v["direct" if m <= c else "sharded"] / v["aiter"] for m,v in samples.items()), c) for c in candidates]
value, cutoff = min(scores)
if cutoff == max(samples):
    cutoff = 4096
result = {"cutoff": cutoff, "source": "pilot", "n": len(samples),
          "mean_selected_over_aiter": value, "top5": sorted(scores)[:5]}
path = run / "routing.json"
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(result, indent=2))
tmp.replace(path)
print(json.dumps(result))
