"""Tests for the Multinomial Naive Bayes CVE classifier."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import features


class SeverityMappingTests(unittest.TestCase):
    def test_severity_from_score(self) -> None:
        cases = [
            (0.0, "LOW"),
            (3.9, "LOW"),
            (4.0, "MEDIUM"),
            (6.9, "MEDIUM"),
            (7.5, "HIGH"),
            (9.5, "CRITICAL"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(features.severity_from_score(score), expected)


class DocumentPipelineTests(unittest.TestCase):
    def test_build_documents_skips_empty_token_rows(self) -> None:
        records = [
            ("Remote overflow exploit enables root access", "CRITICAL"),
            ("the and or", "LOW"),  # removed entirely by preprocessing
        ]
        docs, skipped = features.build_documents(records)
        self.assertEqual(len(docs), 1)
        self.assertEqual(skipped, 1)
        self.assertEqual(docs[0].label, "CRITICAL")

    def test_train_save_load_and_predict(self) -> None:
        dataset = [
            ("Critical overflow vulnerability allows remote root exploit", "CRITICAL"),
            ("High privilege escalation flaw grants admin rights", "HIGH"),
            ("Medium information leak reveals debug output", "MEDIUM"),
            ("Low cosmetic issue misaligns tooltip text", "LOW"),
        ]
        documents, skipped = features.build_documents(dataset)
        self.assertEqual(skipped, 0)
        model = features.train_classifier(documents, laplace=1.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.json"
            model.save(model_path)
            reloaded = features.NaiveBayesModel.load(model_path)

        critical_label, _ = reloaded.predict_text("Remote overflow exploit achieves root access")
        self.assertEqual(critical_label, "CRITICAL")
        low_label, _ = reloaded.predict_text("Minor cosmetic tooltip text issue")
        self.assertEqual(low_label, "LOW")

        accuracy, _, _ = features.evaluate_model(reloaded, documents)
        self.assertGreaterEqual(accuracy, 0.75)


class DatasetLoadingTests(unittest.TestCase):
    def test_load_csv_dataset_with_score_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "cves.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                fieldnames = ["description", "cvss_score", "severity"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"description": "Low issue", "cvss_score": 2.0})
                writer.writerow({"description": "High issue", "cvss_score": 7.5})
                writer.writerow({"description": "", "severity": "HIGH"})  # skipped

            records, total = features.load_dataset(csv_path)

        self.assertEqual(total, 3)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][1], "LOW")
        self.assertEqual(records[1][1], "HIGH")

    def test_load_json_dataset_with_mixed_fields(self) -> None:
        payload = {
            "records": [
                {"description": "Critical buffer overflow", "severity": "critical"},
                {"summary": "Medium info leak", "cvss_score": 5.0},
                {"details": "", "severity": "LOW"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "cves.json"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            records, total = features.load_dataset(json_path)

        self.assertEqual(total, 3)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][1], "CRITICAL")
        self.assertEqual(records[1][1], "MEDIUM")


class StratifiedSplitTests(unittest.TestCase):
    def test_stratified_split_preserves_each_label(self) -> None:
        documents = []
        for label in features.SEVERITY_LEVELS:
            for idx in range(5):
                token = f"{label.lower()}_{idx}"
                documents.append(features.Document(tokens=[token], label=label))

        train, test = features.stratified_split(documents, train_ratio=0.8, seed=123)

        self.assertEqual(len(train) + len(test), len(documents))
        self.assertTrue(train)
        self.assertTrue(test)
        for label in features.SEVERITY_LEVELS:
            train_count = sum(1 for doc in train if doc.label == label)
            test_count = sum(1 for doc in test if doc.label == label)
            self.assertGreater(train_count, 0, msg=f"missing {label} in train split")
            self.assertGreater(test_count, 0, msg=f"missing {label} in test split")


class TrainingPipelineTests(unittest.TestCase):
    def test_train_pipeline_from_csv_file(self) -> None:
        rows = [
            ("Critical overflow allows code execution", "CRITICAL"),
            ("Critical kernel flaw enables RCE", "CRITICAL"),
            ("High privilege escalation in kernel module", "HIGH"),
            ("High buffer overflow leaks credentials", "HIGH"),
            ("Medium info leak via verbose debug", "MEDIUM"),
            ("Medium default credentials allow read access", "MEDIUM"),
            ("Low cosmetic issue misplaces tooltip", "LOW"),
            ("Low typo in about page", "LOW"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "cves.csv"
            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["description", "severity"])
                writer.writerows(rows)
            model_path = Path(tmp_dir) / "model.json"

            with redirect_stdout(io.StringIO()):
                model = features.train_pipeline(
                    data_path=data_path,
                    model_path=model_path,
                    seed=42,
                    laplace=1.0,
                    top_k=5,
                )
            self.assertTrue(model_path.exists())
            label, _ = model.predict_text(rows[0][0])
        self.assertEqual(label, "CRITICAL")


class EvaluationPipelineTests(unittest.TestCase):
    def test_evaluate_existing_model_emits_report(self) -> None:
        rows = [
            ("Critical overflow allows code execution", "CRITICAL"),
            ("High privilege escalation", "HIGH"),
            ("Medium information disclosure", "MEDIUM"),
            ("Low cosmetic typo", "LOW"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "eval.csv"
            with data_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["description", "severity"])
                writer.writerows(rows)
            model_path = Path(tmp_dir) / "model.json"
            with redirect_stdout(io.StringIO()):
                features.train_pipeline(
                    data_path=data_path,
                    model_path=model_path,
                    seed=7,
                    laplace=1.0,
                    top_k=3,
                )
            report_path = Path(tmp_dir) / "report.json"
            with redirect_stdout(io.StringIO()):
                _, summary = features.evaluate_existing_model(
                    model_path=model_path,
                    data_path=data_path,
                    report_path=report_path,
                )

            self.assertTrue(report_path.exists())
            self.assertGreaterEqual(summary["accuracy"], 0.75)
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["total_records"], len(rows))
            self.assertIn("confusion_matrix", persisted)


if __name__ == "__main__":  # pragma: no cover - unittest discovery
    unittest.main()
