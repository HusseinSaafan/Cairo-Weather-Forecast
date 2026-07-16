"""Compare existing model metrics in artifacts and persist champion model.

Champion is selected by the lowest RMSE from *_metrics.json files.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)


ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "model_eval")
CHAMP_PATH = os.path.join(ARTIFACTS_DIR, "champ.pkl")
COMPARISON_PATH = os.path.join(ARTIFACTS_DIR, "model_comparison.json")


def _save_logs(payload: Dict[str, Any]) -> str:
	day_dir = os.path.join(LOGS_DIR, datetime.now().strftime("%Y-%m-%d"))
	os.makedirs(day_dir, exist_ok=True)
	log_path = os.path.join(day_dir, f"champ_model_summary_{datetime.now():%Y%m%d_%H%M%S}.txt")

	with open(log_path, "w", encoding="utf-8") as f:
		f.write("Champion model comparison summary\n")
		f.write(f"Generated at: {payload['generated_at']}\n")
		f.write(f"Selection metric: {payload['selection_metric']}\n\n")
		f.write("Leaderboard (sorted by RMSE)\n")
		for rank, item in enumerate(payload["results"], start=1):
			f.write(
				f"{rank}. {item['model']}: MSE={item['mse']:.6f}, MAE={item['mae']:.6f}, RMSE={item['rmse']:.6f}\n"
			)

		champ = payload["champion"]
		f.write("\nChampion\n")
		f.write(f"model={champ['model']}\n")
		f.write(f"mse={champ['mse']:.6f}\n")
		f.write(f"mae={champ['mae']:.6f}\n")
		f.write(f"rmse={champ['rmse']:.6f}\n")
		f.write(f"source_artifact={champ['source_artifact']}\n")
		f.write(f"champ_path={champ['champ_path']}\n")

	return log_path


def _save_plot(results: List[Dict[str, Any]]) -> str:
	os.makedirs(EVAL_FIG_DIR, exist_ok=True)
	plot_path = os.path.join(EVAL_FIG_DIR, "model_comparison_metrics.png")

	labels = [item["model"] for item in results]
	mse_vals = [item["mse"] for item in results]
	mae_vals = [item["mae"] for item in results]
	rmse_vals = [item["rmse"] for item in results]

	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	fig.suptitle("Model Metrics Comparison (lower is better)")

	axes[0].barh(labels, mse_vals, color="#4C78A8")
	axes[0].set_title("MSE")
	axes[0].invert_yaxis()

	axes[1].barh(labels, mae_vals, color="#F58518")
	axes[1].set_title("MAE")
	axes[1].invert_yaxis()

	axes[2].barh(labels, rmse_vals, color="#54A24B")
	axes[2].set_title("RMSE")
	axes[2].invert_yaxis()

	for ax in axes:
		ax.set_xlabel("Error")
		ax.grid(axis="x", linestyle="--", alpha=0.3)

	fig.tight_layout()
	fig.savefig(plot_path, dpi=300)
	plt.close(fig)

	return plot_path


def _validate_metrics(model_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
	for key in ("mse", "mae", "rmse", "artifact_path"):
		if key not in metrics:
			raise ValueError(f"Model '{model_name}' did not return required key: {key}")

	validated = {
		"model": model_name,
		"mse": float(metrics["mse"]),
		"mae": float(metrics["mae"]),
		"rmse": float(metrics["rmse"]),
		"artifact_path": str(metrics["artifact_path"]),
	}
	return validated


def _read_metrics_files() -> List[Dict[str, Any]]:
	results: List[Dict[str, Any]] = []
	for filename in sorted(os.listdir(ARTIFACTS_DIR)):
		if not filename.endswith("_metrics.json"):
			continue
		metrics_path = os.path.join(ARTIFACTS_DIR, filename)
		with open(metrics_path, "r", encoding="utf-8") as f:
			payload = json.load(f)
		model_name = str(payload.get("model") or filename.replace("_metrics.json", ""))
		results.append(_validate_metrics(model_name, payload))
	return results


def main() -> Dict[str, Any]:
	os.makedirs(ARTIFACTS_DIR, exist_ok=True)

	print("Reading metrics from artifacts/*.json...")
	results = _read_metrics_files()

	if not results:
		raise RuntimeError("No *_metrics.json files found in artifacts directory")

	for metrics in sorted(results, key=lambda item: item["rmse"]):
		print(
			f"{metrics['model']}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}"
		)

	champion = min(results, key=lambda item: item["rmse"])
	champion_artifact = champion["artifact_path"]
	if not os.path.exists(champion_artifact):
		raise FileNotFoundError(f"Champion artifact does not exist: {champion_artifact}")

	shutil.copyfile(champion_artifact, CHAMP_PATH)

	payload: Dict[str, Any] = {
		"generated_at": datetime.now().isoformat(timespec="seconds"),
		"selection_metric": "rmse",
		"results": sorted(results, key=lambda item: item["rmse"]),
		"champion": {
			"model": champion["model"],
			"rmse": champion["rmse"],
			"mae": champion["mae"],
			"mse": champion["mse"],
			"source_artifact": champion_artifact,
			"champ_path": CHAMP_PATH,
		},
	}

	with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)

	log_path = _save_logs(payload)
	plot_path = _save_plot(payload["results"])

	print("\nModel comparison complete.")
	print(f"Champion model: {champion['model']} (RMSE={champion['rmse']:.6f})")
	print(f"Saved champion artifact: {CHAMP_PATH}")
	print(f"Saved comparison report: {COMPARISON_PATH}")
	print(f"Saved log summary: {log_path}")
	print(f"Saved comparison plot: {plot_path}")

	return payload


if __name__ == "__main__":
	main()
