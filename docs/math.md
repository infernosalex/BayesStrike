# BayesStrike – Mathematical Notes (WIP)

> Work in progress – this document seeds the derivations that will guide future experiments.

## 1. Model Setup

We treat every CVE description as a bag of unigram tokens extracted after lower‑casing, removing punctuation, and filtering stopwords. For each severity class \(y \in \{\text{LOW}, \text{MEDIUM}, \text{HIGH}, \text{CRITICAL}\}\) we track token counts \(c_{y,w}\) over the project vocabulary \(V\).

The joint model assumes conditional independence among tokens given the class label:

\[
P(\mathbf{x}, y) = P(y) \prod_{w\in V} P(w \mid y)^{c_w}
\]

where \(c_w\) is the count of token \(w\) in the description. The classifier follows the maximum a posteriori (MAP) decision rule.

## 2. Parameters (to be expanded)

- **Class priors.** Currently derived with add-one smoothing. Next revision will compare Dirichlet priors that reflect historical CVSS distributions.
- **Token likelihoods.** We use Laplace smoothing today; future notes will analyse the evidence for asymmetric priors (e.g., different \(\alpha_y\)).

More detailed derivations, including log-domain simplifications and ties to multinomial conjugacy, will be added in the next iteration.

## 3. Evaluation Goals

This section will capture mathematical targets for calibration and expected log-likelihood improvements. For now it simply bookmarks:

1. Confusion-matrix normalisation proofs.
2. Connections between per-class F1 and Bayesian risk.
3. How indicative tokens relate to likelihood ratios \(\log \frac{P(w\mid y)}{P(w\mid y')}\).

