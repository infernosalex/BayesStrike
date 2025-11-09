# BayesStrike

![BayesStrike logo](logo.png)

Multinomial Naive Bayes experiments for labelling CVE descriptions with the standard CVSS v3.1 severities (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

> Mathematical derivations are being drafted in `docs/math.md` (very early notes).

## Data expectations

- Input files can be CSV or JSON. Each record must expose a textual `description` and either a categorical `severity` or a numeric `cvss_score` (0–10).
- When only a score is present we map it with the v3.1 brackets: 0–3.9 `LOW`, 4.0–6.9 `MEDIUM`, 7.0–8.9 `HIGH`, 9.0–10.0 `CRITICAL`.
- Training ignores rows that lack text after preprocessing (lower-case, punctuation removal, stopword filtering, unigram tokens).

## Model summary

We implement Multinomial Naive Bayes from scratch. Let `y` denote the severity class and `x = (w_1, …, w_n)` the token counts for a CVE description. The classifier maximises

```
log P(y | x) ∝ log P(y) + Σ_i c_i · log P(w_i | y)
```

Class priors use add-one smoothing: `P(y) = (N_y + 1) / (N + |Y|)`. Word likelihoods use Laplace smoothing with parameter α (default 1.0):

```
P(w | y) = (count_y(w) + α) / (Σ_w' count_y(w') + α · |V|)
```

All computations happen in log-space to avoid numerical underflow. Predictions return both the argmax label and per-class log-probabilities.

## CLI usage

Training performs an 80/20 stratified split, prints accuracy, precision/recall/F1 per class, a confusion matrix, and the top indicative tokens per severity.

```bash
python3 features.py train --data data/cves.csv --model-path models/cve_nb.json
```

Predicting uses a saved model:

```bash
python3 features.py predict --model-path models/cve_nb.json "Buffer overflow in XYZ allows remote code execution"
```

Evaluating an existing model against a labeled dataset without retraining:

```bash
python3 features.py evaluate --data data/cves.csv --model-path models/cve_nb.json --report-path reports/latest.json
```

You can also fetch fresh CVEs from the NVD API and immediately train a model:

```bash
python3 features.py fetch-train --start-year 2023 --end-year 2023 --output data/cves_2023.csv \
  --model-path models/cve_nb.json --api-key $NVD_KEY
```

This command shells out to the project’s `fetch_nvd` logic, respects the same rate-limit parameters, and then reuses the standard training pipeline.

## References

- [NVD 2.0 API](https://nvd.nist.gov/developers/vulnerabilities) – data source.
- [CVSS v3.1 Specification](https://www.first.org/cvss/specification-document) – severity ranges.
- Mitchell, T. *Machine Learning*, McGraw Hill, 1997 – Multinomial Naive Bayes formulation.
