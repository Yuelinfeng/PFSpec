"""Development-only probability fit; export inference without sklearn dependency."""
import argparse
from collections import Counter
import json
from pathlib import Path

from probability_model import FEATURES, feature_vector, predict_raw


def export_model(estimator):
    trees = []
    for row in estimator.estimators_:
        tree = row[0].tree_
        trees.append(dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                          feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
                          value=tree.value[:, 0, 0].tolist()))
    return dict(features=list(FEATURES), initial=float(estimator.init_.constant_[0, 0]),
                learning_rate=float(estimator.learning_rate), trees=trees)


def rows(path, split):
    output = []
    for line in path.open():
        event = json.loads(line)
        source = event["source_id"]
        if source not in split:
            continue
        context = event["round_context"]
        labels = event["labels"].get("target_mass_by_token", {})
        main = context.get("main_position")
        for candidate in context.get("candidates", context.get("candidate_pool", ())):
            token = candidate["token_id"]
            if str(token) not in labels:
                continue
            output.append(dict(source=source, x=feature_vector(context, candidate),
                               y=float(labels[str(token)]), main=token == main))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    from sklearn.ensemble import GradientBoostingRegressor
    import numpy as np
    split = json.loads(args.split_report.read_text())["split"]
    train = rows(args.rounds, split["train_sources"])
    valid = rows(args.rounds, split["validation_sources"])
    counts = Counter(r["source"] for r in train)
    weights = np.asarray([1.0 / counts[r["source"]] for r in train])
    weights *= len(weights) / weights.sum()
    x = np.asarray([r["x"] for r in train]); y = np.asarray([r["y"] for r in train])
    xv = np.asarray([r["x"] for r in valid]); yv = np.asarray([r["y"] for r in valid])
    reports = []
    best = None
    # Only the original two-source development validation decides capacity.
    for depth, leaf, iterations in [(2, 16, 16), (2, 16, 32), (2, 16, 64), (3, 24, 32)]:
        estimator = GradientBoostingRegressor(n_estimators=iterations, max_depth=depth,
            min_samples_leaf=leaf, learning_rate=0.1, loss="squared_error", random_state=20261023)
        estimator.fit(x, y, sample_weight=weights)
        pred = np.clip(estimator.predict(xv), 0, 1)
        per_source = []
        for source in split["validation_sources"]:
            indices = [i for i, r in enumerate(valid) if r["source"] == source and not r["main"]]
            per_source.append(float(np.mean((pred[indices] - yv[indices]) ** 2)))
        score = float(np.mean(per_source))
        exported = export_model(estimator)
        delta = max(abs(predict_raw(exported, r["x"]) - q)
                    for r, q in zip(valid, estimator.predict(xv)))
        if delta > 1e-12:
            raise AssertionError("Portable prediction differs")
        report = dict(depth=depth, min_leaf=leaf, trees=iterations, source_equal_backup_mse=score,
                      all_mae=float(np.mean(abs(pred-yv))), portable_max_abs_difference=delta)
        reports.append(report)
        if best is None or score < best[0]:
            best = (score, exported, report)
    args.out.mkdir(parents=True)
    (args.out/"probability_model.json").write_text(json.dumps(best[1], indent=2)+"\n")
    (args.out/"fit_report.json").write_text(json.dumps(dict(split=split, train_rows=len(train),
        validation_rows=len(valid), candidates=reports, selected=best[2],
        scope="Original development sources only; no online speed claim"), indent=2)+"\n")
    print(json.dumps(dict(selected=best[2], train_rows=len(train), validation_rows=len(valid))))


if __name__ == "__main__":
    main()
