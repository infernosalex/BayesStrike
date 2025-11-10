"""Flask application for interacting with the BayesStrike classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from flask import Flask

from .views import init_routes


def create_app(model_path: Optional[str] = None) -> Flask:
    app = Flask(__name__)
    app.config.setdefault("MODEL_PATH", model_path or "models/cve_nb.json")
    app.config.setdefault("UPLOAD_FOLDER", str(Path("uploads")))
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    init_routes(app)
    return app
