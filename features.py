"""Multinomial Naive Bayes classifier for CVE severities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SEVERITY_LEVELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
TRAIN_RATIO = 0.8
DEFAULT_SEED = 42
DEFAULT_LAPLACE = 1.0
DEFAULT_TOP_K = 10
DEFAULT_FETCH_DELAY = 0.6
DEFAULT_FETCH_WINDOW_DAYS = 30


@dataclass
class Document:
    tokens: List[str]
    label: str


@dataclass
class NaiveBayesModel:
    terms: List[str]
    class_log_prior: Dict[str, float]
    class_log_likelihoods: Dict[str, List[float]]
    laplace: float
    class_doc_counts: Dict[str, int]
    class_word_totals: Dict[str, int]
    vocabulary: Dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.vocabulary = {term: idx for idx, term in enumerate(self.terms)}

    def predict_tokens(self, tokens: Sequence[str]) -> Tuple[str, Dict[str, float]]:
        token_counts = Counter(token for token in tokens if token in self.vocabulary)
        scores: Dict[str, float] = {}
        for label, log_prior in self.class_log_prior.items():
            score = log_prior
            likelihoods = self.class_log_likelihoods[label]
            for token, freq in token_counts.items():
                score += freq * likelihoods[self.vocabulary[token]]
            scores[label] = score
        predicted = max(scores.items(), key=lambda item: item[1])[0]
        return predicted, scores

    def predict_text(self, text: str) -> Tuple[str, Dict[str, float]]:
        return self.predict_tokens(preprocess_text(text))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "terms": self.terms,
            "class_log_prior": self.class_log_prior,
            "class_log_likelihoods": self.class_log_likelihoods,
            "laplace": self.laplace,
            "class_doc_counts": self.class_doc_counts,
            "class_word_totals": self.class_word_totals,
        }

    def save(self, path: Path) -> None:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "NaiveBayesModel":
        with path.expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            terms=payload["terms"],
            class_log_prior=payload["class_log_prior"],
            class_log_likelihoods={k: list(v) for k, v in payload["class_log_likelihoods"].items()},
            laplace=payload["laplace"],
            class_doc_counts=payload["class_doc_counts"],
            class_word_totals=payload["class_word_totals"],
        )


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def preprocess_text(text: str) -> List[str]:
    return [token for token in tokenize(text) if token and token not in STOPWORDS]


def load_dataset(path: Path) -> Tuple[List[Tuple[str, str]], int]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_dataset(path)
    if suffix == ".json":
        return _load_json_dataset(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def _load_csv_dataset(path: Path) -> Tuple[List[Tuple[str, str]], int]:
    records: List[Tuple[str, str]] = []
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            normalized = _normalize_record(row)
            if normalized:
                records.append(normalized)
    return records, total


def _load_json_dataset(path: Path) -> Tuple[List[Tuple[str, str]], int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            payload = data["records"]
        else:
            payload = list(data.values())
    elif isinstance(data, list):
        payload = data
    else:
        raise ValueError("JSON dataset must be a list or dict with a 'records' list.")
    records: List[Tuple[str, str]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_record(entry)
        if normalized:
            records.append(normalized)
    return records, len(payload)


def _normalize_record(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    lowered = {str(key).lower(): value for key, value in record.items()}
    description = _get_description(lowered)
    if not description:
        return None
    severity = _get_severity(lowered)
    if severity is None:
        return None
    return description.strip(), severity


def _get_description(record: Dict[str, Any]) -> Optional[str]:
    for key in ("description", "details", "summary", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _get_severity(record: Dict[str, Any]) -> Optional[str]:
    severity_keys = ("severity", "base_severity", "cvss_severity")
    score_keys = ("cvss_score", "base_score", "score", "cvssv3")
    for key in severity_keys:
        severity = record.get(key)
        normalized = _normalize_severity(severity)
        if normalized:
            return normalized
    for key in score_keys:
        score = _normalize_score(record.get(key))
        if score is not None:
            return severity_from_score(score)
    return None


def _normalize_severity(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in SEVERITY_LEVELS else None


def _normalize_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 10.0 else None


def severity_from_score(score: float) -> str:
    if score <= 3.9:
        return "LOW"
    if score <= 6.9:
        return "MEDIUM"
    if score <= 8.9:
        return "HIGH"
    return "CRITICAL"


def build_documents(records: Sequence[Tuple[str, str]]) -> Tuple[List[Document], int]:
    documents: List[Document] = []
    skipped = 0
    for description, label in records:
        tokens = preprocess_text(description)
        if not tokens:
            skipped += 1
            continue
        documents.append(Document(tokens=tokens, label=label))
    if not documents:
        raise ValueError("No documents remained after preprocessing – check dataset quality.")
    return documents, skipped


def stratified_split(
    documents: Sequence[Document],
    *,
    train_ratio: float,
    seed: int,
) -> Tuple[List[Document], List[Document]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Document]] = {label: [] for label in SEVERITY_LEVELS}
    for doc in documents:
        buckets.setdefault(doc.label, []).append(doc)
    train: List[Document] = []
    test: List[Document] = []
    for label, bucket in buckets.items():
        if not bucket:
            continue
        rng.shuffle(bucket)
        split_idx = max(1, int(round(len(bucket) * train_ratio)))
        if split_idx >= len(bucket):
            split_idx = len(bucket) - 1 if len(bucket) > 1 else len(bucket)
        train.extend(bucket[:split_idx])
        test.extend(bucket[split_idx:])
    if not test:
        if len(train) < 2:
            raise ValueError("Unable to create a test split – add more samples.")
        test.append(train.pop())
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_vocabulary_terms(documents: Sequence[Document]) -> List[str]:
    vocab: Dict[str, int] = {}
    for doc in documents:
        for token in doc.tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return list(vocab.keys())


def train_classifier(documents: Sequence[Document], laplace: float) -> NaiveBayesModel:
    if not documents:
        raise ValueError("Cannot train without documents.")
    terms = build_vocabulary_terms(documents)
    if not terms:
        raise ValueError("Vocabulary is empty – preprocessing removed all tokens.")
    vocabulary = {term: idx for idx, term in enumerate(terms)}
    class_doc_counts = {label: 0 for label in SEVERITY_LEVELS}
    for doc in documents:
        class_doc_counts[doc.label] += 1
    total_docs = len(documents)
    num_classes = len(SEVERITY_LEVELS)
    prior_alpha = 1.0
    class_log_prior: Dict[str, float] = {}
    for label in SEVERITY_LEVELS:
        prior = (class_doc_counts[label] + prior_alpha) / (total_docs + prior_alpha * num_classes)
        class_log_prior[label] = math.log(prior)
    class_word_counts: Dict[str, List[int]] = {label: [0] * len(terms) for label in SEVERITY_LEVELS}
    class_word_totals: Dict[str, int] = {label: 0 for label in SEVERITY_LEVELS}
    for doc in documents:
        frequencies = Counter(doc.tokens)
        target = class_word_counts[doc.label]
        for token, freq in frequencies.items():
            idx = vocabulary[token]
            target[idx] += freq
            class_word_totals[doc.label] += freq
    class_log_likelihoods: Dict[str, List[float]] = {}
    vocab_size = len(terms)
    for label in SEVERITY_LEVELS:
        denominator = class_word_totals[label] + laplace * vocab_size
        likelihoods: List[float] = []
        for idx in range(vocab_size):
            numerator = class_word_counts[label][idx] + laplace
            probability = numerator / denominator if denominator > 0 else 1.0 / vocab_size
            likelihoods.append(math.log(probability))
        class_log_likelihoods[label] = likelihoods
    return NaiveBayesModel(
        terms=terms,
        class_log_prior=class_log_prior,
        class_log_likelihoods=class_log_likelihoods,
        laplace=laplace,
        class_doc_counts=class_doc_counts,
        class_word_totals=class_word_totals,
    )


def evaluate_model(
    model: NaiveBayesModel,
    documents: Sequence[Document],
) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    labels = SEVERITY_LEVELS
    confusion: Dict[str, Dict[str, int]] = {
        true_label: {pred_label: 0 for pred_label in labels}
        for true_label in labels
    }
    correct = 0
    for doc in documents:
        prediction, _ = model.predict_tokens(doc.tokens)
        if prediction == doc.label:
            correct += 1
        confusion[doc.label][prediction] += 1
    accuracy = correct / len(documents)
    metrics: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(confusion[label].values())
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return accuracy, metrics, confusion


def print_metrics(
    accuracy: float,
    metrics: Dict[str, Dict[str, float]],
    confusion: Dict[str, Dict[str, int]],
) -> None:
    print(f"Accuracy: {accuracy:.4f}")
    print("\nPer-class precision/recall/F1:")
    header = f"{'Class':<10}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}"
    print(header)
    print("-" * len(header))
    for label in SEVERITY_LEVELS:
        stats = metrics[label]
        print(
            f"{label:<10}{stats['precision']:>12.4f}{stats['recall']:>12.4f}{stats['f1']:>12.4f}{int(stats['support']):>10}"
        )
    print("\nConfusion matrix (rows=true, cols=pred):")
    header_row = [" "] + list(SEVERITY_LEVELS)
    print("\t".join(header_row))
    for true_label in SEVERITY_LEVELS:
        row = [true_label]
        for pred_label in SEVERITY_LEVELS:
            row.append(str(confusion[true_label][pred_label]))
        print("\t".join(row))


def print_top_words(model: NaiveBayesModel, top_k: int) -> None:
    print(f"\nTop {top_k} indicative tokens per class:")
    for label in SEVERITY_LEVELS:
        total_words = model.class_word_totals.get(label, 0)
        if total_words == 0:
            print(f"  {label}: insufficient training samples.")
            continue
        likelihoods = model.class_log_likelihoods[label]
        ranked = sorted(zip(model.terms, likelihoods), key=lambda item: item[1], reverse=True)[:top_k]
        terms = ", ".join(term for term, _ in ranked)
        print(f"  {label}: {terms}")


def train_pipeline(
    data_path: Path,
    model_path: Path,
    *,
    seed: int,
    laplace: float,
    top_k: int,
) -> NaiveBayesModel:
    records, total = load_dataset(data_path)
    if not records:
        raise ValueError(f"Dataset did not contain any usable CVE entries: {data_path}")
    documents, skipped = build_documents(records)
    print(f"Loaded {len(records)} labeled CVEs out of {total} rows from {data_path}.")
    if skipped:
        print(f"Skipped {skipped} entries with empty vocab after preprocessing.")
    train_docs, test_docs = stratified_split(documents, train_ratio=TRAIN_RATIO, seed=seed)
    print(f"Training samples: {len(train_docs)}, test samples: {len(test_docs)}")
    model = train_classifier(train_docs, laplace=laplace)
    accuracy, metrics, confusion = evaluate_model(model, test_docs)
    print_metrics(accuracy, metrics, confusion)
    print_top_words(model, top_k=top_k)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    return model


def handle_train(args: argparse.Namespace) -> None:
    try:
        train_pipeline(
            data_path=Path(args.data),
            model_path=Path(args.model_path),
            seed=args.seed,
            laplace=args.laplace,
            top_k=args.top_k,
        )
    except Exception as exc:
        raise SystemExit(str(exc))


def handle_fetch_train(args: argparse.Namespace) -> None:
    try:
        import fetch_nvd  # Local import to avoid hard dependency when unused.
    except ImportError as exc:  # pragma: no cover - import failure path
        raise SystemExit(f"fetch_nvd module unavailable: {exc}")

    output_path = Path(args.output)
    print(
        f"Fetching CVEs from NVD ({args.start_year}-{args.end_year}) into {output_path}..."
    )
    try:
        total = fetch_nvd.fetch_cves_to_csv(
            start_year=args.start_year,
            end_year=args.end_year,
            output_path=output_path,
            api_key=args.api_key,
            delay=args.delay,
            window_days=args.window_days,
            sample_size=args.sample_size,
            seed=args.fetch_seed,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to fetch CVEs: {exc}")

    print(f"Fetched {total} CVEs. Starting training phase...\n")
    train_pipeline(
        data_path=output_path,
        model_path=Path(args.model_path),
        seed=args.seed,
        laplace=args.laplace,
        top_k=args.top_k,
    )


def handle_predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model path {model_path} does not exist. Train a model first.")
    model = NaiveBayesModel.load(model_path)
    label, log_probs = model.predict_text(args.text)
    print(f"Predicted severity: {label}")
    print("Log-probabilities:")
    for severity in SEVERITY_LEVELS:
        score = log_probs.get(severity)
        if score is None:
            continue
        print(f"  {severity:<8}: {score:.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use a Multinomial Naive Bayes classifier for CVE descriptions."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the classifier from a dataset.")
    train_parser.add_argument("--data", required=True, help="Path to CSV or JSON dataset.")
    train_parser.add_argument(
        "--model-path",
        default="models/cve_nb.json",
        help="Where to store the trained model (default: models/cve_nb.json).",
    )
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splitting.")
    train_parser.add_argument(
        "--laplace",
        type=float,
        default=DEFAULT_LAPLACE,
        help="Laplace smoothing applied to word likelihoods.",
    )
    train_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="How many indicative tokens to print per class.",
    )
    train_parser.set_defaults(func=handle_train)

    predict_parser = subparsers.add_parser("predict", help="Predict severity for a new description.")
    predict_parser.add_argument("text", help="CVE description to classify (quote for multi-word).")
    predict_parser.add_argument(
        "--model-path",
        default="models/cve_nb.json",
        help="Path to a trained model JSON file.",
    )
    predict_parser.set_defaults(func=handle_predict)

    fetch_train_parser = subparsers.add_parser(
        "fetch-train",
        help="Fetch CVEs from the NVD API and immediately train the classifier.",
    )
    fetch_train_parser.add_argument("--start-year", type=int, required=True, help="First year to download.")
    fetch_train_parser.add_argument("--end-year", type=int, required=True, help="Last year to download.")
    fetch_train_parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV for fetched CVEs (reused by the training step).",
    )
    fetch_train_parser.add_argument("--api-key", help="Optional NVD API key for higher rate limits.")
    fetch_train_parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_FETCH_DELAY,
        help="Delay between NVD requests (seconds).",
    )
    fetch_train_parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_FETCH_WINDOW_DAYS,
        help="Window size (days) for each NVD request window.",
    )
    fetch_train_parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optionally down-sample fetched CVEs before training.",
    )
    fetch_train_parser.add_argument(
        "--fetch-seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed when sampling the fetched dataset.",
    )
    fetch_train_parser.add_argument(
        "--model-path",
        default="models/cve_nb.json",
        help="Where to store the trained model (JSON).",
    )
    fetch_train_parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the train/test split.",
    )
    fetch_train_parser.add_argument(
        "--laplace",
        type=float,
        default=DEFAULT_LAPLACE,
        help="Laplace smoothing applied to token likelihoods.",
    )
    fetch_train_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of indicative tokens to print per class.",
    )
    fetch_train_parser.set_defaults(func=handle_fetch_train)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
