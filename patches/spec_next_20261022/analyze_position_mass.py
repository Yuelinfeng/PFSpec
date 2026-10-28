"""Summarize cumulative backup mass from saved Position policies without a GPU."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = []
    for row in data["rows"]:
        ids, scores, policy = row["ids"], row["scores"], row["target_policy"]
        if not ids:
            continue
        order = sorted(range(len(ids)), key=lambda index: (-scores[index], index))
        point = ids[order[0]]
        rejection = 1.0 - policy[point]
        measures = {}
        for count in range(1, 5):
            backups = [ids[index] for index in order[1:count + 1]]
            mass = sum(policy[token] for token in backups)
            measures[str(count)] = dict(backup_ids=backups, available=len(backups),
                                        mass=mass, conditional_hit=mass / rejection if rejection > 0 else 0)
        rows.append(dict(source=row["source"], prefix=row["prefix"], point=point,
                         rejection=rejection, widths=measures))
    source_values = {}
    for source in sorted({row["source"] for row in rows}):
        subset = [row for row in rows if row["source"] == source]
        source_values[source] = {str(count): {
            key: statistics.fmean(row["widths"][str(count)][key] for row in subset)
            for key in ("mass", "conditional_hit", "available")
        } for count in range(1, 5)}
    result = dict(input=str(args.input), contexts=len(rows), sources=len(source_values),
                  scope="Saved teacher prefixes; candidate mass only, no online speed prediction",
                  source_equal={str(count): {
                      key: statistics.fmean(values[str(count)][key] for values in source_values.values())
                      for key in ("mass", "conditional_hit", "available")
                  } for count in range(1, 5)}, by_source=source_values, rows=rows)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("contexts", "sources", "source_equal")}))


if __name__ == "__main__":
    main()
