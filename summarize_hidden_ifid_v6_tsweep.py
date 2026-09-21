#!/usr/bin/env python3
"""
Summarize v6 hidden-iFID t-sweep results into complete, ordered CSV files.

The script supports three input levels:

1. Entire sweep root:
   .../hidden_ifid_v6_tsweep

2. One scope's result root:
   .../hidden_ifid_v6_tsweep/class/results
   .../hidden_ifid_v6_tsweep/global/results

3. One specific condition:
   .../hidden_ifid_v6_tsweep/class/results/t_0p10

For every discovered (scope, t) condition, the script reads ONLY the exact,
expected paths:

    <condition>/<expected-tag>/hidden_ifid_metrics.json

It never uses fuzzy model-name matching. The primary score is:

    metrics.main_hidden_fid_raw

Outputs:
    main_hidden_fid_raw_wide.csv
    main_hidden_fid_raw_long.csv
    all_metrics_long.csv
    issues.csv
    manifest.json
    by_condition/<scope>_<t_slug>_main_hidden_fid_raw.csv

CSV files are UTF-8 with BOM so they open cleanly in Excel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PRIMARY_METRIC = "main_hidden_fid_raw"

# Exact user-requested model order.
# display_name, exact result directory tag, expected experiment directory basename
MODEL_ORDER: list[tuple[str, str, str]] = [
    ("sdvae", "ifid-sdvae", "sit-xl-sdvae-400k"),
    ("sd3vae", "ifid-sd3vae", "sit-xl-sd3-shift-400k"),
    ("fluxvae", "ifid-fluxvae", "sit-xl-flux-shift-400k"),
    ("dcae", "ifid-dcae", "sit-xl-dcae-400k"),
    ("softvq", "ifid-softvq", "sit-XL-softvq-400k"),
    ("vavae", "ifid-vavae", "sit-xl-vavae-400k"),
    ("vavae64", "ifid-vavae64", "sit-xl-vavae64-shift-400k"),
    ("eqvae", "ifid-eqvae", "sit-XL-eqvae-400k"),
    ("maetok", "ifid-maetok", "sit-XL-maetok-400k"),
    ("invae", "ifid-invae", "sit-xl-invae-400k"),
    ("repae-sdvae", "ifid-repae-sdvae", "sit-xl-repae-400k"),
    ("detok", "ifid-detok", "sit-xl-detok-400k"),
    ("qwenimgvae", "ifid-qwenimgvae", "sit-xl-qw-shift-400k"),
    ("rae", "ifid-rae", "sit-xl-rae-shift-400k"),
    ("svg", "ifid-svg", "sit-xl-svg-400k"),
    ("flux2", "ifid-flux2", "sit-xl-flux2vae-400k"),
    ("svg-t2i-p256", "ifid-svg-t2i-p256", "sit-xl-svgt2iP-400k"),
    ("dmvae", "ifid-dmvae", "sit-xl-dmvae-400k"),
    ("uae", "ifid-uae", "sit-xl-uae-400k"),
    ("vtps", "ifid-vtps", "sit-xl-vtps-400k"),
    ("vtpb", "ifid-vtpb", "sit-xl-vtpb-400k"),
    ("vtpl", "ifid-vtpl", "sit-xl-vtpl-400k"),
    (
        "scalerae-siglip2",
        "ifid-scalerae-siglip2",
        "sit-xl-raet2i_siglip2-400k",
    ),
    (
        "scalerae-webssl",
        "ifid-scalerae-webssl",
        "sit-xl-raet2i_webssl-400k",
    ),
    ("pae-dino", "ifid-pae-dino", "sit-xl-pae-dinov2-400k"),
]

EXPECTED_TAGS = {tag for _, tag, _ in MODEL_ORDER}
T_SLUG_RE = re.compile(r"^t_(?P<value>[mp0-9]+(?:p[0-9]+)?)$")


@dataclass(frozen=True)
class Condition:
    scope: str
    t_slug: str
    t_value: float
    directory: Path

    @property
    def column_name(self) -> str:
        return f"{self.scope}_{self.t_slug}"


@dataclass
class Record:
    condition: Condition
    order: int
    model: str
    tag: str
    expected_exp: str
    json_path: Path
    status: str
    score: float | None
    settings: dict[str, Any]
    metrics: dict[str, Any]


@dataclass
class Issue:
    severity: str
    issue_type: str
    scope: str
    t_slug: str
    model: str
    tag: str
    path: str
    message: str


def parse_t_slug(name: str) -> float:
    """Parse t_0p10 -> 0.10 and t_m0p10 -> -0.10."""
    match = T_SLUG_RE.fullmatch(name)
    if not match:
        raise ValueError(f"Invalid t directory name: {name}")
    raw = match.group("value").replace("m", "-").replace("p", ".")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite t value in directory name: {name}")
    return value


def t_to_slug(value: float) -> str:
    """Canonical fallback slug, preserving a readable decimal representation."""
    text = format(value, ".12g")
    return "t_" + text.replace("-", "m").replace(".", "p")


def parse_t_filter(raw: str | None) -> set[float] | None:
    if raw is None or not raw.strip():
        return None
    parts = [part for part in re.split(r"[\s,;]+", raw.strip()) if part]
    values: set[float] = set()
    for part in parts:
        value = float(part)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite t filter value: {part}")
        values.add(value)
    return values


def close_float(a: float, b: float, atol: float = 1e-8) -> bool:
    return abs(a - b) <= atol


def infer_sweep_root(root: Path) -> Path:
    root = root.resolve()

    if root.name.startswith("t_") and root.parent.name == "results":
        # .../<scope>/results/t_x
        return root.parents[2]

    if root.name == "results" and root.parent.name in {"class", "global"}:
        # .../<scope>/results
        return root.parents[1]

    if root.name in {"class", "global"} and (root / "results").is_dir():
        # .../<scope>
        return root.parent

    return root


def discover_conditions(
    root: Path,
    scope_filter: str,
    t_filter: set[float] | None,
) -> tuple[Path, list[Condition]]:
    """Discover exact result conditions from one of the supported input levels."""
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    candidates: list[tuple[str, Path]] = []

    # One exact t directory.
    if root.is_dir() and root.name.startswith("t_") and root.parent.name == "results":
        scope = root.parent.parent.name
        if scope not in {"class", "global"}:
            raise ValueError(
                f"Cannot infer scope from specific t directory: {root}"
            )
        candidates.append((scope, root))

    # One scope's "results" directory.
    elif root.is_dir() and root.name == "results" and root.parent.name in {
        "class",
        "global",
    }:
        scope = root.parent.name
        candidates.extend(
            (scope, path)
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("t_")
        )

    # One scope directory.
    elif root.is_dir() and root.name in {"class", "global"}:
        result_root = root / "results"
        if not result_root.is_dir():
            raise FileNotFoundError(f"Missing results directory: {result_root}")
        candidates.extend(
            (root.name, path)
            for path in result_root.iterdir()
            if path.is_dir() and path.name.startswith("t_")
        )

    # Entire sweep root.
    else:
        for scope in ("class", "global"):
            result_root = root / scope / "results"
            if not result_root.is_dir():
                continue
            candidates.extend(
                (scope, path)
                for path in result_root.iterdir()
                if path.is_dir() and path.name.startswith("t_")
            )

    conditions: list[Condition] = []
    discovery_errors: list[str] = []

    for scope, directory in candidates:
        if scope_filter != "all" and scope != scope_filter:
            continue
        try:
            t_value = parse_t_slug(directory.name)
        except ValueError as exc:
            discovery_errors.append(str(exc))
            continue
        if t_filter is not None and not any(
            close_float(t_value, requested) for requested in t_filter
        ):
            continue
        conditions.append(
            Condition(
                scope=scope,
                t_slug=directory.name,
                t_value=t_value,
                directory=directory.resolve(),
            )
        )

    if discovery_errors:
        print("[WARN] Ignored malformed t directories:", file=sys.stderr)
        for message in discovery_errors:
            print(f"  - {message}", file=sys.stderr)

    # Class first, global second; numeric t order within each scope.
    scope_rank = {"class": 0, "global": 1}
    conditions.sort(key=lambda c: (scope_rank[c.scope], c.t_value, c.t_slug))

    if not conditions:
        raise RuntimeError(
            f"No result conditions discovered under {root}. "
            "Expected paths such as class/results/t_0p10."
        )

    return infer_sweep_root(root), conditions


def scalar_for_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def read_record(
    condition: Condition,
    order: int,
    model: str,
    tag: str,
    expected_exp: str,
    issues: list[Issue],
) -> Record:
    json_path = condition.directory / tag / "hidden_ifid_metrics.json"

    def add_issue(
        severity: str,
        issue_type: str,
        message: str,
        path: Path = json_path,
    ) -> None:
        issues.append(
            Issue(
                severity=severity,
                issue_type=issue_type,
                scope=condition.scope,
                t_slug=condition.t_slug,
                model=model,
                tag=tag,
                path=str(path),
                message=message,
            )
        )

    if not json_path.is_file():
        add_issue("error", "missing_json", "Expected metrics JSON does not exist")
        return Record(
            condition=condition,
            order=order,
            model=model,
            tag=tag,
            expected_exp=expected_exp,
            json_path=json_path,
            status="missing",
            score=None,
            settings={},
            metrics={},
        )

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_issue(
            "error",
            "invalid_json",
            f"{type(exc).__name__}: {exc}",
        )
        return Record(
            condition=condition,
            order=order,
            model=model,
            tag=tag,
            expected_exp=expected_exp,
            json_path=json_path,
            status="invalid_json",
            score=None,
            settings={},
            metrics={},
        )

    if not isinstance(payload, dict):
        add_issue("error", "invalid_payload", "Top-level JSON must be an object")
        payload = {}

    settings = payload.get("settings", {})
    metrics = payload.get("metrics", {})

    if not isinstance(settings, dict):
        add_issue("error", "invalid_settings", "settings must be a JSON object")
        settings = {}
    if not isinstance(metrics, dict):
        add_issue("error", "invalid_metrics", "metrics must be a JSON object")
        metrics = {}

    score = finite_number(metrics.get(PRIMARY_METRIC))
    if score is None:
        add_issue(
            "error",
            "missing_or_invalid_primary_metric",
            f"metrics.{PRIMARY_METRIC} is missing, non-numeric, NaN, or infinite",
        )

    # Validate t against the directory name.
    json_t = finite_number(settings.get("t_input_before_shift"))
    if json_t is None:
        add_issue(
            "warning",
            "missing_or_invalid_t_setting",
            "settings.t_input_before_shift is missing or invalid",
        )
    elif not close_float(json_t, condition.t_value):
        add_issue(
            "error",
            "t_mismatch",
            f"Directory says t={condition.t_value}, JSON says t={json_t}",
        )

    # Validate experiment identity against explicit model mapping.
    exp_path = str(settings.get("exp_path", ""))
    exp_basename = Path(exp_path).name if exp_path else ""
    if not exp_path:
        add_issue(
            "warning",
            "missing_exp_path",
            "settings.exp_path is missing",
        )
    elif exp_basename != expected_exp:
        add_issue(
            "error",
            "experiment_mismatch",
            f"Expected experiment '{expected_exp}', got '{exp_basename}'",
        )

    # Scope validation. Global v6 may omit nn_scope entirely.
    nn_scope = settings.get("nn_scope")
    if condition.scope == "class":
        if nn_scope != "within_class":
            add_issue(
                "error",
                "scope_mismatch",
                f"Expected nn_scope='within_class', got {nn_scope!r}",
            )
        same_class = finite_number(metrics.get("nn_same_class_rate"))
        if same_class is None:
            add_issue(
                "warning",
                "missing_same_class_rate",
                "metrics.nn_same_class_rate is missing or invalid",
            )
        elif not close_float(same_class, 1.0, atol=1e-12):
            add_issue(
                "error",
                "class_nn_invariant_failed",
                f"Expected nn_same_class_rate=1.0, got {same_class}",
            )
    else:
        if nn_scope not in (None, "", "global"):
            add_issue(
                "warning",
                "unexpected_global_scope_value",
                f"Global result has nn_scope={nn_scope!r}",
            )

    n_reference = finite_number(settings.get("n_reference"))
    n_query = finite_number(settings.get("n_query"))
    if n_reference is not None and int(n_reference) != 50000:
        add_issue(
            "warning",
            "unexpected_reference_count",
            f"Expected 50000 references, got {n_reference:g}",
        )
    if n_query is not None and int(n_query) != 50000:
        add_issue(
            "warning",
            "unexpected_query_count",
            f"Expected 50000 queries, got {n_query:g}",
        )

    status = "ok" if score is not None else "invalid_metric"
    return Record(
        condition=condition,
        order=order,
        model=model,
        tag=tag,
        expected_exp=expected_exp,
        json_path=json_path,
        status=status,
        score=score,
        settings=settings,
        metrics=metrics,
    )


def find_unexpected_model_dirs(
    condition: Condition,
    issues: list[Issue],
) -> None:
    for child in sorted(condition.directory.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("ifid-") and child.name not in EXPECTED_TAGS:
            issues.append(
                Issue(
                    severity="warning",
                    issue_type="unexpected_model_directory",
                    scope=condition.scope,
                    t_slug=condition.t_slug,
                    model="",
                    tag=child.name,
                    path=str(child),
                    message="Directory is not in the fixed 25-model mapping",
                )
            )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_condition_csvs(output_dir: Path, records: list[Record]) -> list[Path]:
    by_condition: dict[tuple[str, str], list[Record]] = {}
    for record in records:
        by_condition.setdefault(
            (record.condition.scope, record.condition.t_slug), []
        ).append(record)

    paths: list[Path] = []
    target_dir = output_dir / "by_condition"
    for (scope, t_slug), group in sorted(
        by_condition.items(),
        key=lambda item: (
            0 if item[0][0] == "class" else 1,
            item[1][0].condition.t_value,
        ),
    ):
        group.sort(key=lambda record: record.order)
        path = target_dir / f"{scope}_{t_slug}_{PRIMARY_METRIC}.csv"
        rows = [
            {
                "order": record.order,
                "model": record.model,
                "tag": record.tag,
                PRIMARY_METRIC: "" if record.score is None else record.score,
                "status": record.status,
                "json_path": str(record.json_path),
            }
            for record in group
        ]
        write_csv(
            path,
            ["order", "model", "tag", PRIMARY_METRIC, "status", "json_path"],
            rows,
        )
        paths.append(path)
    return paths


def write_long_primary(output_dir: Path, records: list[Record]) -> Path:
    path = output_dir / f"{PRIMARY_METRIC}_long.csv"
    rows: list[dict[str, Any]] = []
    for record in records:
        settings = record.settings
        rows.append(
            {
                "scope": record.condition.scope,
                "t": record.condition.t_value,
                "t_slug": record.condition.t_slug,
                "order": record.order,
                "model": record.model,
                "tag": record.tag,
                PRIMARY_METRIC: "" if record.score is None else record.score,
                "status": record.status,
                "json_path": str(record.json_path),
                "exp_path": scalar_for_csv(settings.get("exp_path")),
                "nn_scope": scalar_for_csv(settings.get("nn_scope")),
                "nn_metric_full_hidden": scalar_for_csv(
                    settings.get("nn_metric_full_hidden")
                ),
                "retrieval_protocol": scalar_for_csv(
                    settings.get("retrieval_protocol")
                ),
                "depth": scalar_for_csv(settings.get("depth")),
                "alpha": scalar_for_csv(settings.get("alpha")),
                "n_reference": scalar_for_csv(settings.get("n_reference")),
                "n_query": scalar_for_csv(settings.get("n_query")),
                "tokens": scalar_for_csv(settings.get("tokens")),
                "hidden_dim": scalar_for_csv(settings.get("hidden_dim")),
                "seed": scalar_for_csv(settings.get("seed")),
            }
        )

    fields = [
        "scope",
        "t",
        "t_slug",
        "order",
        "model",
        "tag",
        PRIMARY_METRIC,
        "status",
        "json_path",
        "exp_path",
        "nn_scope",
        "nn_metric_full_hidden",
        "retrieval_protocol",
        "depth",
        "alpha",
        "n_reference",
        "n_query",
        "tokens",
        "hidden_dim",
        "seed",
    ]
    write_csv(path, fields, rows)
    return path


def write_wide_primary(
    output_dir: Path,
    records: list[Record],
    conditions: list[Condition],
) -> Path:
    path = output_dir / f"{PRIMARY_METRIC}_wide.csv"
    lookup = {
        (record.condition.scope, record.condition.t_slug, record.tag): record
        for record in records
    }
    condition_columns = [condition.column_name for condition in conditions]

    rows: list[dict[str, Any]] = []
    for order, (model, tag, _) in enumerate(MODEL_ORDER, start=1):
        row: dict[str, Any] = {
            "order": order,
            "model": model,
            "tag": tag,
        }
        for condition in conditions:
            record = lookup[(condition.scope, condition.t_slug, tag)]
            row[condition.column_name] = (
                "" if record.score is None else record.score
            )
        rows.append(row)

    write_csv(path, ["order", "model", "tag", *condition_columns], rows)
    return path


def write_all_metrics(output_dir: Path, records: list[Record]) -> Path:
    path = output_dir / "all_metrics_long.csv"

    setting_keys: set[str] = set()
    metric_keys: set[str] = set()
    for record in records:
        setting_keys.update(record.settings.keys())
        metric_keys.update(record.metrics.keys())

    # Put the primary metric first, then alphabetize remaining metrics.
    ordered_metric_keys = (
        [PRIMARY_METRIC] if PRIMARY_METRIC in metric_keys else []
    ) + sorted(metric_keys - {PRIMARY_METRIC})
    ordered_setting_keys = sorted(setting_keys)

    base_fields = [
        "scope",
        "t",
        "t_slug",
        "order",
        "model",
        "tag",
        "status",
        "json_path",
    ]
    fields = (
        base_fields
        + [f"setting_{key}" for key in ordered_setting_keys]
        + [f"metric_{key}" for key in ordered_metric_keys]
    )

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "scope": record.condition.scope,
            "t": record.condition.t_value,
            "t_slug": record.condition.t_slug,
            "order": record.order,
            "model": record.model,
            "tag": record.tag,
            "status": record.status,
            "json_path": str(record.json_path),
        }
        for key in ordered_setting_keys:
            row[f"setting_{key}"] = scalar_for_csv(record.settings.get(key))
        for key in ordered_metric_keys:
            row[f"metric_{key}"] = scalar_for_csv(record.metrics.get(key))
        rows.append(row)

    write_csv(path, fields, rows)
    return path


def write_issues(output_dir: Path, issues: list[Issue]) -> Path:
    path = output_dir / "issues.csv"
    rows = [
        {
            "severity": issue.severity,
            "issue_type": issue.issue_type,
            "scope": issue.scope,
            "t_slug": issue.t_slug,
            "model": issue.model,
            "tag": issue.tag,
            "path": issue.path,
            "message": issue.message,
        }
        for issue in issues
    ]
    write_csv(
        path,
        [
            "severity",
            "issue_type",
            "scope",
            "t_slug",
            "model",
            "tag",
            "path",
            "message",
        ],
        rows,
    )
    return path


def print_preview(records: list[Record], conditions: list[Condition]) -> None:
    lookup = {
        (record.condition.scope, record.condition.t_slug, record.tag): record
        for record in records
    }

    for condition in conditions:
        print()
        print(
            f"[{condition.scope} | {condition.t_slug} | "
            f"t={condition.t_value:g}]"
        )
        print("-" * 58)
        for order, (model, tag, _) in enumerate(MODEL_ORDER, start=1):
            record = lookup[(condition.scope, condition.t_slug, tag)]
            value = (
                "MISSING"
                if record.score is None
                else f"{record.score:.10f}"
            )
            print(f"{order:02d}. {model:<22} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize v6 class/global t-sweep hidden-iFID JSON files "
            "with a fixed 25-model order."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/video_ssd/kongzishang/xutongda/git/IFID/"
            "exps_interpolate/hidden_ifid_v6_tsweep"
        ),
        help=(
            "Entire sweep root, one scope's results directory, "
            "or one specific t directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <inferred sweep root>/summary",
    )
    parser.add_argument(
        "--scope",
        choices=["all", "class", "global"],
        default="all",
        help="Optional scope filter.",
    )
    parser.add_argument(
        "--t-values",
        type=str,
        default=None,
        help='Optional filter, e.g. "0.05 0.10 0.25" or "0.05,0.10".',
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a nonzero exit code when any error-level issue exists.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not print ordered scores to stdout.",
    )
    args = parser.parse_args()

    try:
        t_filter = parse_t_filter(args.t_values)
        sweep_root, conditions = discover_conditions(
            args.root,
            args.scope,
            t_filter,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (sweep_root / "summary").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    issues: list[Issue] = []
    records: list[Record] = []

    for condition in conditions:
        find_unexpected_model_dirs(condition, issues)
        for order, (model, tag, expected_exp) in enumerate(
            MODEL_ORDER,
            start=1,
        ):
            records.append(
                read_record(
                    condition=condition,
                    order=order,
                    model=model,
                    tag=tag,
                    expected_exp=expected_exp,
                    issues=issues,
                )
            )

    # Keep deterministic row order.
    condition_rank = {
        (condition.scope, condition.t_slug): index
        for index, condition in enumerate(conditions)
    }
    records.sort(
        key=lambda record: (
            condition_rank[
                (record.condition.scope, record.condition.t_slug)
            ],
            record.order,
        )
    )

    condition_files = write_condition_csvs(output_dir, records)
    long_path = write_long_primary(output_dir, records)
    wide_path = write_wide_primary(output_dir, records, conditions)
    all_metrics_path = write_all_metrics(output_dir, records)
    issues_path = write_issues(output_dir, issues)

    ok_count = sum(record.score is not None for record in records)
    expected_count = len(conditions) * len(MODEL_ORDER)
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)

    manifest = {
        "input_root": str(args.root.expanduser().resolve()),
        "inferred_sweep_root": str(sweep_root),
        "output_dir": str(output_dir),
        "primary_metric": PRIMARY_METRIC,
        "model_order": [
            {
                "order": order,
                "model": model,
                "tag": tag,
                "expected_experiment": exp,
            }
            for order, (model, tag, exp) in enumerate(MODEL_ORDER, start=1)
        ],
        "conditions": [
            {
                "scope": condition.scope,
                "t_slug": condition.t_slug,
                "t": condition.t_value,
                "directory": str(condition.directory),
            }
            for condition in conditions
        ],
        "expected_scores": expected_count,
        "valid_scores": ok_count,
        "missing_or_invalid_scores": expected_count - ok_count,
        "error_count": error_count,
        "warning_count": warning_count,
        "files": {
            "wide": str(wide_path),
            "long": str(long_path),
            "all_metrics": str(all_metrics_path),
            "issues": str(issues_path),
            "by_condition": [str(path) for path in condition_files],
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88)
    print("V6 hidden-iFID t-sweep summary")
    print(f"Input:              {args.root.expanduser().resolve()}")
    print(f"Conditions:         {len(conditions)}")
    print(f"Expected scores:    {expected_count}")
    print(f"Valid scores:       {ok_count}")
    print(f"Missing/invalid:    {expected_count - ok_count}")
    print(f"Validation errors:  {error_count}")
    print(f"Warnings:           {warning_count}")
    print(f"Primary wide CSV:   {wide_path}")
    print(f"Primary long CSV:   {long_path}")
    print(f"All metrics CSV:    {all_metrics_path}")
    print(f"Issues CSV:         {issues_path}")
    print(f"Manifest:           {manifest_path}")
    print("=" * 88)

    if not args.no_preview:
        print_preview(records, conditions)

    if error_count:
        print(
            f"\nThere are {error_count} error-level validation issues. "
            f"See: {issues_path}",
            file=sys.stderr,
        )

    if args.strict and error_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
