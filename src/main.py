"""Project pipeline entrypoint.

Execution order:
1) Ingestion/preprocessing
2) EDA
3) Modeling
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INGESTION_MODULES: List[str] = [
	"src.ingestion_preprocessing.clean_split",
]

EDA_MODULES: List[str] = [
	"src.eda.initial_eda",
	"src.eda.eda",
	"src.eda.eda_log",
	"src.eda.decomp",
	"src.eda.decomp_log",
	"src.eda.test",
	"src.eda.test_log",
]

MODELING_MODULES: List[str] = [
	"src.modeling.sarima",
	"src.modeling.sarima_log",
	"src.modeling.man_sarima",
	"src.modeling.man_sarima_log",
	"src.modeling.auto_arima",
	"src.modeling.auto_arima_log",
	"src.modeling.prophet",
	"src.modeling.champ_model",
]


def _run_module(module_name: str) -> None:
	print(f"\n[PIPELINE] Running {module_name}...")
	completed = subprocess.run(
		[sys.executable, "-m", module_name],
		cwd=PROJECT_ROOT,
		check=False,
	)
	if completed.returncode != 0:
		raise RuntimeError(
			f"Module failed: {module_name} (exit code {completed.returncode})"
		)


def main() -> None:
	print("[PIPELINE] Starting full pipeline")

	for module_name in INGESTION_MODULES:
		_run_module(module_name)

	for module_name in EDA_MODULES:
		_run_module(module_name)

	for module_name in MODELING_MODULES:
		_run_module(module_name)

	print("\n[PIPELINE] Pipeline completed successfully")


if __name__ == "__main__":
	main()
