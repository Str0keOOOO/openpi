"""Count scene frequencies in downloaded OFFICIAL per-episode metadata; CPU only.

Writes a proposal, never silently modifies benchmark.json. Not a success filter.
"""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    counts = Counter()
    files = 0
    for f in sorted(args.dataset_root.rglob("ep_meta.json")):
        payload = json.loads(f.read_text())
        # Different releases use one object per file or an episode-indexed map.
        records = [payload] if "layout_id" in payload else list(payload.values())
        for r in records:
            if not isinstance(r, dict) or "layout_id" not in r or "style_id" not in r:
                continue
            layout, style = r["layout_id"], r["style_id"]
            if isinstance(layout, int) and isinstance(style, int) and 11 <= layout <= 60 and 11 <= style <= 60:
                counts[(layout, style)] += 1
        files += 1
    if not counts:
        raise SystemExit("No usable official pretrain episode scene metadata found; no frequency claim made.")
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    layout, style = ranked[0][0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dataset_root": str(args.dataset_root.resolve()), "files": files,
        "selection_basis": "most_frequent_pair_in_supplied_official_metadata",
        "layout_id": layout, "style_id": style,
        "scene_counts": [{"layout_id": k[0], "style_id": k[1], "count": v} for k, v in ranked],
    }, indent=2))


if __name__ == "__main__":
    main()
