# Science: a topological-spectral descriptor of activations across grokking

Audience: ML-interpretability researchers. The verdicts are in
[`RESULTS.md`](RESULTS.md); this document covers the question, the setup, the
descriptor, the baselines and the statistical protocol.

## 1. Question

The multiview descriptor of Reinoso & Escudero (2026) characterises a trained
network by two views of its hidden activations:

- the **functional graph** of its units (X1);
- the **persistent homology** of the activation point cloud (X2);
- their label-free fusion (X3).

It was built to discriminate architectures in a zoo of 40 trained image
classifiers. It was then ported to geophysical flow fields
([GEM benchmark](https://github.com/Dcrr1717/gem-geophysical-descriptor)),
where it tied, but did not beat, strong physical baselines.

Here we bring it back to neural activations. We ask two interpretability
questions, each with a pre-registered Go/No-Go rule against strong baselines:

1. **Detection.** Grokking is a sharp, delayed transition from memorization
   to generalization (Power et al. 2022). Does the geometry and topology of
   the activations reveal it, and does it do so earlier, more accurately or
   more cleanly than standard representation diagnostics?
2. **Discrimination.** From one activation matrix, can the descriptor tell
   *which layer* it came from, *which task* the network was trained on, and
   whether the network is in the *memorized* or the *grokked* state? Can it
   do so better than the same baselines?

A tie is a valid, reportable outcome. The GEM benchmark already showed that
"separates the classes" and "beats strong baselines" are different claims.

## 2. Why grokking is the right testbed

- **A known ground truth in time.** Train accuracy saturates early; held-out
  accuracy jumps much later. The transition can be read off held-out
  accuracy, so every detector can be scored against it.
- **A known mechanism with geometric content.** For modular addition, Nanda
  et al. (2023) reverse-engineered the grokked one-layer transformer. It
  embeds tokens on circles at a few key Fourier frequencies, composes them
  with trigonometric identities in attention and the MLP, and reads out
  cos(w(a + b - c)). Liu et al. (2022) observed ring-shaped embeddings in
  generalizing models. If any representation change is topological, this one
  should be: circles (H1 classes) and periodic, strongly correlated neuron
  groups (dense functional-graph communities) should appear at grokking.
- **Known progress measures to beat.** Weight norm falls steadily under weight
  decay (Liu et al. 2023, "Omnigrok"; Nanda et al. 2023). The circuit forms
  before held-out accuracy moves ("circuit formation", then "cleanup"; Nanda
  et al. 2023). CKA (Kornblith et al. 2019) and linear probes (Alain & Bengio
  2016) are the standard representation-level diagnostics.

## 3. Setup

**Data.** We use the tasks `c = op(a, b) mod p` with p = 59. Each input is
the sequence `[a, b, =]`, and the answer is read at `=`. A fixed random
fraction of all pairs is the training set and the rest is held out. The
operations come from the task list of Power et al.:

| op | definition | pairs |
|---|---|---|
| `add` | a + b mod p | p^2 |
| `sub` | a - b mod p | p^2 |
| `div` | a * b^-1 mod p (Fermat inverse; b != 0) | p(p - 1) |

Train fraction: 0.5 (`add`), 0.6 (`sub`), 0.55 (`div`). It was calibrated
per operation in pilot runs so that each one shows delayed generalization
within 6000 CPU steps (RESULTS.md, "Pilot").

`mul` (a * b with all pairs, including the zero row and column) is
implemented. It was dropped after pilot runs: its held-out accuracy rose
gradually, alongside train accuracy, with no memorization plateau, so it is
not a grokking task in this budget (see RESULTS.md, "Pilot").

**Model.** A one-layer decoder-only transformer as in Nanda et al. (2023):
learned token and positional embeddings, d_model = 128, 4 heads, a ReLU MLP
with 512 units, no LayerNorm, untied unembedding. It is trained full batch
with AdamW (lr 1e-3, betas (0.9, 0.98), weight decay 1.0, 10 warm-up steps),
from scratch, on a CPU. The forward pass is exact but cheap. The layer-0
Q/K/V are linear in the embeddings, so they are computed once per vocabulary
token, and only the `=` position is propagated past the attention; a unit
test checks it against the plain implementation.

**Logged activations.** Every `log_every` steps the run records activations
on a fixed probe set of 1024 input pairs (train and held-out mixed) at four
sites:

| site | what it is | units |
|---|---|---|
| `embed` | token + position embedding of operand a | 128 |
| `resid_mid_0` | residual stream at `=` after attention | 128 |
| `mlp_0` | post-ReLU MLP neurons at `=` | 512 |
| `resid_post_0` | final residual stream at `=` (read by the unembedding) | 128 |

**Controls.** Runs with weight decay 0 memorize but do not grok in the budget.
They test specificity: a detector that "finds a transition" there is not
specific to grokking. The controls differ from the grokking runs in weight
decay as well as in outcome, so magnitude differences between the two groups
cannot separate "grokked" from "trained with weight decay".

**Phase thresholds** (config key `phases`, defaults):
- memorized: train accuracy >= 0.99;
- generalization onset: held-out accuracy >= 0.5;
- grokked: held-out accuracy >= 0.9;
- the "grokked" class of Experiment 2: held-out accuracy >= 0.95.

A grokking run must show *delayed* generalization: held-out accuracy still
below the onset threshold when memorization completes.

## 4. The descriptor (authentic, ported verbatim)

The descriptor is applied to the activation matrix `act` (probe inputs x
units) of one site at one checkpoint. Code: `src/tsi/descriptor/authentic.py`.
`tests/test_authentic.py` reproduces the original `desc_x1` / `desc_x2` of the
Reinoso & Escudero repository to 1e-12.

- **X1, functional graph (R^10).** Take the first 80 units (the original cap)
  and compute Pearson correlations of their activations across inputs. An
  edge joins two units with |r| > 0.40. Dead units (std <= 1e-6, frequent
  with ReLU) get no edge. The ten invariants are density, average
  clustering, transitivity, degree mean / std / max fraction, fraction of
  connected components, edges per node, and betweenness mean / std. X1
  answers: do units fire in coherent, correlated groups?
- **X2, persistent homology (R^12).** Take the first 50 inputs, project them
  on their top 15 principal components, min-max scale to [0, 1], and compute
  Vietoris-Rips persistence up to H1 (ripser). Summarise H0 and H1 with
  `sig6`: mean, std, max, sum, count and 75th percentile of the finite
  persistences. X2 answers: does the cloud of input representations cluster
  (H0) or form loops (H1)?
- **X3, fusion (R^22).** X3 = [X1/|X1|, X2/|X2|], the label-free variant of
  the original paper.

GEM's `authentic.py` computes X2 as the cubical sublevel-set persistence of
a 2D field. That filtration needs a spatial grid, which an activation matrix
does not have. The neural original of X2 is the Rips persistence of the
activation cloud. GEM documents that variant (`x2_rips_pointcloud`) but does
not port it, so we port it from the same upstream module.

## 5. Baselines

All are computed on the same activation matrices and the same checkpoints.

| baseline | definition |
|---|---|
| `cka_final` | linear CKA (Kornblith et al. 2019) between the site's activations at t and at the final checkpoint |
| `probe_acc` | 2-fold held-out accuracy of a closed-form ridge classifier decoding the answer c from the site's activations |
| `act_stats` | mean, total variance, mean unit variance, mean abs, fraction of zeros, fraction of dead units, mean row norm |
| `pca_stats` | participation ratio, entropy effective rank, explained variance of the top 1 / 5 / 10 PCs |
| `act_pca` | `act_stats` + `pca_stats` (the "raw activation statistics" of the brief) |
| `weight_norm` | parameter L2 norms: embeddings, attention, MLP, unembedding, total |
| `baselines_union` | all of the above concatenated |

`cka_final` is retrospective: it needs the final model, which a real-time
detector would not have. The descriptor needs only the current checkpoint.
`probe_acc` decodes the answer, so on `resid_post_0` (which the unembedding
reads linearly) it is close to a re-measurement of held-out accuracy itself.
These are deliberately strong, label-adjacent baselines.

## 6. Protocol

### Experiment 1: detection

- **Window.** [t_mem, t_end], where t_mem is the first checkpoint with train
  accuracy >= 0.99. Defining it needs only train labels.
- **Detector.** The same label-free detector runs on every signal
  (`src/tsi/changepoint.py`). The trajectory (checkpoints x features, all
  sites concatenated) is z-scored per feature, and a single least-squares
  change point in the multivariate mean is fitted. It reports the change
  time, the step R^2 (variance explained by one step: 1 = a perfect step) and
  the 10 % / 50 % onsets of the step projection.
- **Reference.** t_ref is the change point that the same detector finds in
  held-out accuracy.
- **Scores.**
  - *Timing error*: |t_cp - t_ref| / window.
  - *Cleanness*: step R^2.
  - *Early warning*: on memorization-plateau checkpoints (t_mem <= t, held-out
    accuracy < 0.5), regress the number of steps until held-out accuracy
    reaches 0.9. Ridge and random forest, leave-one-run-out, out-of-fold
    R^2. Training step (`epoch`) is an extra trivial baseline here.
  - *Lead*: 10 % / 50 % onsets of each signal relative to those of
    held-out accuracy (descriptive).
  - *Specificity*: step R^2 on the non-grokking controls.

### Experiment 2: discrimination

Samples are (run, checkpoint, site) triples at or after memorization, from
runs that grok.

| task | label |
|---|---|
| `layer` | which of the 4 sites |
| `task` | add / sub / div |
| `phase` | memorized (held-out < 0.5) vs grokked (held-out >= 0.95) |

The protocol is the GEM one, unchanged:
- repeated StratifiedGroupKFold with group = training run, so no run is ever
  split between train and test folds;
- logistic regression and a 200-tree random forest, keeping the best per
  feature set;
- macro-F1 and confusion matrices.

### Verdict rule (both experiments)

The comparisons are paired: every feature set sees the same folds and runs.
The 95 % CI of each margin comes from a bootstrap over training runs.

- **GO vs baselines:** the best topological view (X1, X2, X1+X2, X3) beats
  the best single baseline by >= +0.05, with the CI above 0.
- **GO incremental:** topology added to the baselines beats them by
  >= +0.05, with the CI above 0. Does topology add signal the baselines miss?
  - In Experiment 1 this is (best baseline + topology) vs the best baseline.
    A concatenation of heterogeneous baselines is a *worse* change-point
    detector than its best member, so beating `baselines_union` there would
    prove nothing. The union comparison is reported as a secondary row.
  - In Experiment 2 it is `baselines_union` + topology vs `baselines_union`,
    as in GEM; a classifier can ignore useless columns.
- **Headroom** = 1 - accuracy(`baselines_union`). If it is below 0.05, an
  incremental GO is arithmetically impossible, and we say so.

In Experiment 2, the training step alone (`epoch`) is shown as a trivial
reference. It shows how much of a task is solvable from "how far into
training" alone.

Choosing the best of several topological views is optimistic for topology.
Choosing the best of several baselines is optimistic for the baselines. Both
choices are reported.

## 7. What a positive or negative result would mean

- **GO on detection** would mean the activation topology moves at a different
  time, or more sharply, than CKA, probes and weight norm. That would make it
  a candidate progress measure in the sense of Nanda et al. (2023).
- **A tie** would mean that the topological signal is real but redundant: the
  representation change at grokking is already fully visible to linear
  similarity and decodability.
- **GO on discrimination** would mean the descriptor encodes properties of
  the representation (depth, computation) that summary statistics and CKA
  do not.

## References

- Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using
  linear classifier probes. arXiv:1610.01644.
- Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence
  barcodes. *Journal of Applied and Computational Topology*, 5, 391-423.
- Chughtai, B., Chan, L., & Nanda, N. (2023). A toy model of universality:
  reverse engineering how networks learn group operations. *ICML 2023*.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An
  Introduction*. AMS.
- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of
  neural network representations revisited. *ICML 2019*, PMLR 97:3519-3529.
- Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M.
  (2022). Towards understanding grokking: an effective theory of
  representation learning. *NeurIPS 2022*.
- Liu, Z., Michaud, E. J., & Tegmark, M. (2023). Omnigrok: grokking beyond
  algorithmic data. *ICLR 2023*.
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023).
  Progress measures for grokking via mechanistic interpretability.
  *ICLR 2023*. arXiv:2301.05217.
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022).
  Grokking: generalization beyond overfitting on small algorithmic datasets.
  arXiv:2201.02177.
- Reinoso Reinoso, C. D., & Escudero Villa, A. I. (2026). Descriptor
  multivista de activaciones neuronales, mediante la fusión de grafos
  funcionales y homología persistente (PanAm 2026; *Revista Bases de la
  Ciencia*). Code: github.com/Dcrr1717/multiview-nn-descriptor.
- Rieck, B., Togninalli, M., Bock, C., Moor, M., Horn, M., Gumbsch, T., &
  Borgwardt, K. (2019). Neural persistence: a complexity measure for deep
  neural networks using algebraic topology. *ICLR 2019*.
- Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the
  pizza: two stories in mechanistic explanation of neural networks.
  *NeurIPS 2023*.
