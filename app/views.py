"""Flask views for BayesStrike."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from features import (
    NaiveBayesModel,
    evaluate_existing_model,
    preprocess_text,
    train_pipeline,
)

UPLOAD_FIELD = "dataset"
TEXT_FIELD = "description"


def init_routes(app: Flask) -> None:
    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict() -> str:
        text = request.form.get(TEXT_FIELD, "").strip()
        if not text:
            flash("Please provide a CVE description to classify.", "warning")
            return redirect(url_for("index"))
        model_path = Path(app.config["MODEL_PATH"])
        if not model_path.exists():
            flash("No trained model found. Train a model before predicting.", "danger")
            return redirect(url_for("index"))
        model = NaiveBayesModel.load(model_path)
        label, log_probs = model.predict_text(text)
        scores = {cls: f"{score:.4f}" for cls, score in log_probs.items()}
        return render_template("index.html", prediction=label, scores=scores, input_text=text)

    @app.route("/train", methods=["POST"])
    def train() -> str:
        uploaded = request.files.get(UPLOAD_FIELD)
        if not uploaded or uploaded.filename == "":
            flash("Please upload a CSV or JSON dataset.", "warning")
            return redirect(url_for("index"))
        destination = Path(app.config["UPLOAD_FOLDER"]) / uploaded.filename
        uploaded.save(destination)
        model_path = Path(app.config["MODEL_PATH"])
        try:
            with open(destination, "rb"):
                pass
            train_pipeline(
                data_path=destination,
                model_path=model_path,
                seed=42,
                laplace=1.0,
                top_k=10,
            )
            flash(f"Training complete. Model saved to {model_path}.", "success")
        except Exception as exc:
            flash(f"Training failed: {exc}", "danger")
        return redirect(url_for("index"))

    @app.route("/evaluate", methods=["POST"])
    def evaluate() -> str:
        uploaded = request.files.get(UPLOAD_FIELD)
        if not uploaded or uploaded.filename == "":
            flash("Please upload a CSV or JSON dataset for evaluation.", "warning")
            return redirect(url_for("index"))
        destination = Path(app.config["UPLOAD_FOLDER"]) / uploaded.filename
        uploaded.save(destination)
        model_path = Path(app.config["MODEL_PATH"])
        try:
            report_path = destination.with_suffix(".report.json")
            _, summary = evaluate_existing_model(
                model_path=model_path,
                data_path=destination,
                report_path=report_path,
            )
            flash("Evaluation completed successfully.", "success")
            return render_template(
                "results.html",
                summary=_format_summary(summary),
                report_path=report_path.name,
            )
        except Exception as exc:
            flash(f"Evaluation failed: {exc}", "danger")
            return redirect(url_for("index"))

    @app.route("/reports/<path:filename>")
    def download_report(filename: str):  # type: ignore[override]
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)


def _format_summary(summary: Dict[str, object]) -> Dict[str, object]:
    summary = dict(summary)
    summary["accuracy"] = f"{summary['accuracy']:.4f}"
    summary["metrics"] = {
        cls: {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in stats.items()}
        for cls, stats in summary["metrics"].items()
    }
    return summary
