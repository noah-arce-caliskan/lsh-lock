# NIST SP 800-90B integration

## Setup and execution

This wrapper invokes NIST's implementation; it does not reimplement the ten
estimators or modify `src/estimators.rs`. The reference experiments used
`ea_non_iid` 1.1.8 from official commit
`87c104d0ed4cbc96103e7b8b38d6f2c7e0a6b289`.

Build the tool separately using its
[official installation instructions](https://github.com/usnistgov/SP800-90B_EntropyAssessment).
For a new checkout:

```bash
git clone https://github.com/usnistgov/SP800-90B_EntropyAssessment
cd SP800-90B_EntropyAssessment
git checkout 87c104d0ed4cbc96103e7b8b38d6f2c7e0a6b289
cd cpp
make non_iid
```

The official README lists the required compiler and system libraries. Its
`selftest` directory provides reference-output checks. On the machine running
this wrapper, install NumPy in the chosen Python environment:

```bash
python3 -m pip install numpy
```

From this repository:

```bash
cargo build --release
./target/release/lsh-lock nist-entropy \
  -t /share/common/embeddings/synlock_embeddings/iris_embeddings/test \
  --nist-bin /path/to/ea_non_iid \
  -o nist-results/iris --trials 1000 --seed 67 --jobs 2
```

Alternatively call `python3 scripts/nist_entropy.py` with the same arguments.
`--python /path/to/python3` selects a Python environment for the Rust command.
`--keep-inputs` retains the individual byte-per-bit streams; by default they
are temporary and can be regenerated from the original readings and saved
selection indices. `--no-controls` disables the default random controls.
`--timeout` is a limit in seconds per native run (default 180).

For Ben's evaluation, run this **once per dataset/feature export**, alongside
the per-subset estimator suite. The command deliberately takes no locker
index file: a 60/70-bit subset would yield a different, much shorter stream
than the full-template protocol evaluated here. No NIST score is folded into
the minimum across Ben's other estimators. If per-subset NIST measurements
are needed, that requires an explicit separate protocol and controls.

## Input and selection

Input layout is `<identity>/<reading>`, with comma-separated binary values.
Identity and reading filenames are sorted lexicographically. Hidden entries
are ignored; malformed readings, empty identities, or inconsistent widths
are errors. Identity labels and original reading values are preserved.

For each seed, NumPy `default_rng(seed)` chooses one reading uniformly from
each identity's saved readings. These full vectors are concatenated in sorted
identity order, with their original coordinate order. There is exactly one
reading per label in each real input; additional trials change the selected
readings, not the participant population. Seeds default to 67 through 1066.
Labels with different reading counts are supported.

NIST receives each bit as a separate byte valued 0 or 1, with
`bits_per_symbol=1`:

```text
ea_non_iid -i -a -vv -o native.json input.bin 1
```

Each trial also gets independent fair-bit control data with the same shape,
using `default_rng(SeedSequence([20261015, selection_seed]))`. This reproduces
the completed face/iris/fingerprint control policy. These controls differ
from Ben's capture-noise-matched latent-identity simulations.

The wrapper verifies all ten native estimates are present, finite and in
[0,1], verifies the reported overall value equals their minimum, and checks
the native input hash. Missing tests or native errors fail the command and
leave logs/checkpoints for inspection; they are never filled with zeros or
silently omitted. An existing output directory is refused.

## Outputs

- `input_manifest.json`: original identity/reading paths and file hashes,
  reading counts, dimensions and the hash of the ordered binary matrix.
- `selected_rows.npy`: the exact source-row choices in each real trial.
- `raw/`: original official JSON and logs for every real/control run.
- `completed_runs.jsonl`: checkpoint records, including input hashes.
- `report.json`: all ten outputs per trial, tool/NumPy versions, binary hash,
  selection policy and per-estimator minimum, 5th percentile, median and
  maximum. The suite minimum is a separately labeled descriptive statistic.
- `all_scores.csv`: one row per test per run, in bits per input bit.

## Tests

```bash
python3 scripts/test_nist_entropy.py
cargo test --locked
```

Protocol tests cover variable capture counts, preservation of original
labels, binary input validation, reproducibility, missing/non-finite native
estimates, inconsistent overall outputs, and summary semantics. Real-tool
regression checks should additionally compare all ten estimates and input
hashes to the saved reference experiments; test fixtures alone do not
validate the native numerical implementation.

## Short paper-methods description

We supplement the biometric estimators with the ten non-IID estimators in
the official NIST SP 800-90B implementation: Most Common Value, Collision,
Markov, Compression, t-Tuple, Longest Repeated Substring (LRS), Multi Most
Common in Window, Lag Prediction, MultiMMC, and LZ78Y. For each dataset, we
select one saved reading per supplied identity label, concatenate the full
binary readings in fixed identity and coordinate order, and repeat for
1,000 seeded selections. Thresholds and saved bit values are unchanged.
We retain each estimator's output separately and compare with 1,000
length-matched independent fair-bit controls. Minima, percentiles and medians
describe the sampled selections; the minimum is not over all combinations.

| Estimator | Statistic or predictor examined |
|---|---|
| Most Common Value | Frequency of the most common symbol |
| Collision | Waiting times to repeated symbols |
| Markov | Binary symbol and transition probabilities |
| Compression | Recurrence of blocks in the compression statistic |
| t-Tuple | Frequencies of repeated length-t tuples |
| LRS | Repeated substrings and the associated collision statistic |
| Multi Most Common in Window | Predictions from several recent-window sizes |
| Lag Prediction | Predictions from previous symbols at several lags |
| MultiMMC | Predictions from several Markov context lengths |
| LZ78Y | Dictionary-based predictions from previous contexts |

These are exploratory stream diagnostics. The one-reading-per-identity
inputs in the current datasets contain fewer than the one million samples
specified by SP 800-90B. Increasing the trial count does not increase the
number of independent people or the input length of a run. A source model
justifying biometric vector concatenation as an entropy-source process has
not been established, and coordinate order can affect the outputs without
changing matching distances. We therefore do not interpret these rates,
their minimum, or their product with the template length as certified
template entropy, fuzzy min-entropy, or key-security bounds. The suite is
SP 800-90B, distinct from the SP 800-22 randomness suite cited in the original
paper.

Reference: [SP 800-90B](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90B.pdf),
sections 3.1.1, 3.2.2 and 6.3.1-6.3.10.
