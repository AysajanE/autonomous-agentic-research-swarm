from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
ANALYSIS = ROOT / "src" / "analysis"
SCRIPTS = ROOT / "scripts"
for path in (TESTS_ROOT, ANALYSIS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from golden.harness import GoldenRepo  # noqa: E402
from inference import SmallTError, driscoll_kraay_se  # noqa: E402
from runtime_test_utils import chdir, load_quality_gates_module, write_json, write_text  # noqa: E402


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"golden_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


estimate_regimes = _load_script("estimate_regimes")
spec_curve = _load_script("spec_curve")
quality_gates = load_quality_gates_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statistical_config(*, survival_threshold: int = 1) -> dict[str, object]:
    return {
        "vce": "newey_west",
        "lags": 0,
        "regime_breaks": {
            "method": "bai_perron_dynamic_programming_piecewise_constant",
            "series": "str",
            "max_breaks": 1,
            "min_segment": 2,
            "imposed_date": "2024-03-01",
        },
        "specification_curve": {
            "construction_variants": {
                "keying": ["rollup_day", "date_aggregate"],
                "missingness": ["drop", "zero_fill"],
            },
            "analysis_variants": [
                {
                    "id": "base",
                    "estimator": "regime_difference",
                    "regime_date": "2024-03-01",
                }
            ],
            "survival_threshold": survival_threshold,
        },
    }


def _toy_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = (
        ("2024-01-01", False),
        ("2024-01-02", False),
        ("2024-02-01", False),
        ("2024-02-02", False),
        ("2024-03-01", True),
        ("2024-03-02", True),
        ("2024-04-01", True),
        ("2024-04-02", True),
    )
    for date, post in dates:
        for rollup_id, fees, rent in (
            ("a", 10.0, 2.0 if post else 8.0),
            ("b", 20.0, 2.0 if post else 10.0),
        ):
            rows.append(
                {
                    "date_utc": date,
                    "rollup_id": rollup_id,
                    "l2_fees_eth": fees,
                    "rent_paid_eth": rent,
                }
            )
    # Seeded ETL-layer missingness bias: drop and zero-fill are observably distinct.
    rows[2]["rent_paid_eth"] = None
    return pd.DataFrame(rows)


class GoldenM3cTest(unittest.TestCase):
    def test_battle_panel_regime_break_and_regime_difference_artifacts(self) -> None:
        manifest = estimate_regimes.estimate_regime_manifest(
            panel_path=ROOT / "data/processed/panels/daily_rollup_panel.csv",
            panel_manifest_path=ROOT / "data/processed_manifest/daily_rollup_panel_2026-04-09.json",
            series_name="str",
            max_breaks=1,
            min_segment=90,
            imposed_date="2024-03-13",
        )
        self.assertEqual(manifest["break_dates"], ["2024-03-19"])
        self.assertEqual(manifest["break_indices"], [808])
        self.assertAlmostEqual(float(manifest["ssr"]), 25.424877013106013, places=10)
        self.assertEqual(manifest["imposed_regime_comparison"]["distance_days"], 6)
        self.assertEqual(
            manifest["inputs"][0]["sha256"],
            "7a048a4e542136afd906ad56c95d00ef1543dff7096946010a1caa49042013d1",
        )

        panel = pd.read_csv(ROOT / "data/processed/panels/daily_rollup_panel.csv")
        config = {
            "vce": "newey_west",
            "lags": 7,
            "regime_breaks": {
                "method": "bai_perron_dynamic_programming_piecewise_constant",
                "series": "str",
                "max_breaks": 1,
                "min_segment": 90,
                "imposed_date": "2024-03-13",
            },
            "specification_curve": {
                "construction_variants": {
                    "keying": ["rollup_day", "date_aggregate"],
                    "missingness": ["drop", "zero_fill"],
                },
                "analysis_variants": [
                    {
                        "id": "locked",
                        "estimator": "regime_difference",
                        "regime_date": "2024-03-13",
                        "lags": 7,
                    }
                ],
                "survival_threshold": 2,
            },
        }
        artifact = spec_curve.build_spec_curve(
            panel=panel,
            claim_id="REF-STR-REGIME",
            headline_estimate=-0.5746080215460282,
            config=config,
            prereg_lock_sha256="a" * 64,
        )
        by_id = {spec["spec_id"]: spec for spec in artifact["specs"]}
        aggregate = by_id[
            "keying=date_aggregate|missingness=drop|analysis=locked"
        ]
        self.assertAlmostEqual(float(aggregate["estimate"]), -0.5746080215460282, places=12)
        self.assertAlmostEqual(float(aggregate["se"]), 0.015651924672764692, places=12)
        self.assertEqual(aggregate["n"], 1559)
        self.assertEqual(artifact["survival_count"], 2)
        self.assertTrue(artifact["headline_within_curve"])

    def test_dk_guard_and_seeded_construction_bias(self) -> None:
        with self.assertRaises(SmallTError):
            driscoll_kraay_se(
                np.array([1.0, -1.0, 1.0, -1.0]),
                np.ones((4, 1)),
                np.array([1, 1, 2, 2]),
                0,
                min_time_periods=3,
            )
        artifact = spec_curve.build_spec_curve(
            panel=_toy_panel(),
            claim_id="TOY-BIAS",
            headline_estimate=-0.44,
            config=_statistical_config(survival_threshold=4),
            prereg_lock_sha256="b" * 64,
        )
        estimates = {
            (
                spec["construction_variant"]["keying"],
                spec["construction_variant"]["missingness"],
            ): round(float(spec["estimate"]), 12)
            for spec in artifact["specs"]
        }
        self.assertEqual(len(estimates), 4)
        self.assertNotEqual(estimates[("rollup_day", "drop")], estimates[("rollup_day", "zero_fill")])
        self.assertNotEqual(estimates[("rollup_day", "drop")], estimates[("date_aggregate", "drop")])
        self.assertEqual(artifact["survival_count"], 4)
        self.assertTrue(artifact["headline_within_curve"])

    def test_statistical_reporting_gate_green_then_rule_level_vce_red(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = GoldenRepo.create(tmp)
            config = _statistical_config(survival_threshold=4)
            body = (
                "# Analysis plan\n\n"
                "```json\n"
                + json.dumps({"statistical_reporting": config}, indent=2, sort_keys=True)
                + "\n```\n"
            )
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            write_text(
                repo.root,
                "docs/prereg/analysis_plan.lock.md",
                "\n".join(
                    [
                        "---",
                        "schema_version: research_swarm.prereg_lock.v1",
                        "phase: 2b",
                        "status: locked",
                        "locked_at_utc: 2026-07-10T12:00:00Z",
                        f"locked_sha256: {digest}",
                        "locked_by: Golden Owner",
                        "lock_version: 1",
                        "---",
                        "",
                    ]
                )
                + body,
            )
            panel_path = repo.root / "data/processed/panels/toy.csv"
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            _toy_panel().to_csv(panel_path, index=False)
            panel_sha = _sha(panel_path)
            write_json(
                repo.root,
                "reports/analysis/regime_breaks.json",
                {
                    "schema_version": "research_swarm.regime_breaks.v1",
                    "inputs": [
                        {
                            "path": "data/processed/panels/toy.csv",
                            "sha256": panel_sha,
                            "bytes": panel_path.stat().st_size,
                        }
                    ],
                    "method": "bai_perron_dynamic_programming_piecewise_constant",
                    "series": "str",
                    "max_breaks": 1,
                    "min_segment": 2,
                    "break_dates": ["2024-03-01"],
                    "ssr": 0.0,
                    "parameter_source": "active_analysis_lock",
                    "prereg_lock_sha256": digest,
                    "git_sha": "1" * 40,
                },
            )
            table = write_text(
                repo.root,
                "reports/tables/causal.csv",
                "claim_id,estimate,se,ci_lower,ci_upper,n,vce,headline\n"
                "C-CAUSAL,-0.44,0.05,-0.54,-0.34,16,newey_west,true\n",
            )
            uncertainty = write_json(
                repo.root,
                "reports/analysis/uncertainty.json",
                {
                    "schema_version": "research_swarm.headline_uncertainty.v1",
                    "claim_id": "C-CAUSAL",
                    "estimate": -0.44,
                    "confidence_interval": {"lower": -0.54, "upper": -0.34},
                    "measurement_uncertainty": {
                        "coverage_gap_fraction": 0.1,
                        "source": "coverage-audit",
                        "lower": -0.594,
                        "upper": -0.306,
                    },
                },
            )
            sensitivity = write_json(
                repo.root,
                "reports/analysis/sensitivity/C-CAUSAL.json",
                {
                    "method": "cinelli_hazlett_rv",
                    "value": 0.25,
                    "benchmark": 0.1,
                    "threshold": 0.2,
                    "passes": True,
                },
            )
            curve = spec_curve.build_spec_curve(
                panel=_toy_panel(),
                claim_id="C-CAUSAL",
                headline_estimate=-0.44,
                config=config,
                prereg_lock_sha256=digest,
            )
            write_json(repo.root, "reports/analysis/spec_curve/C-CAUSAL.json", curve)
            write_json(
                repo.root,
                "contracts/claims.yaml",
                {
                    "schema_version": "research_swarm.claims.v1",
                    "claims": [
                        {
                            "claim_id": "C-CAUSAL",
                            "statement": "A causal golden estimate.",
                            "type": "causal",
                            "headline": True,
                            "supporting_artifacts": [
                                {"path": "reports/tables/causal.csv", "sha256": _sha(table)}
                            ],
                            "verification_command": "python scripts/noop_gate.py",
                            "uncertainty_artifact": {
                                "path": "reports/analysis/uncertainty.json",
                                "sha256": _sha(uncertainty),
                            },
                            "sensitivity_artifact": {
                                "path": "reports/analysis/sensitivity/C-CAUSAL.json",
                                "sha256": _sha(sensitivity),
                            },
                            "identification_strategy": "docs/prereg/analysis_plan.lock.md#identification",
                        }
                    ],
                },
            )
            repo.git("add", "-A")
            with chdir(repo.root):
                green = quality_gates.gate_statistical_reporting()
                self.assertTrue(green.ok, green.details)
                table.write_text(
                    table.read_text(encoding="utf-8").replace("newey_west", "clustered"),
                    encoding="utf-8",
                )
                red = quality_gates.gate_statistical_reporting()
            self.assertFalse(red.ok)
            reasons = {failure["reason"] for failure in red.details["failures"]}
            self.assertIn("statistical_table_vce_required", reasons)


if __name__ == "__main__":
    unittest.main()
