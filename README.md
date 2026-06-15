# RADS: Relational Adversarial Deviation Score

This repository provides a manuscript-aligned reference implementation. Exact reproduction of the reported numerical results requires the original aligned Apple/S&P 500 dataset and final experimental split, which are not included in the available files.

> Relational Adversarial Deviation Score for Robust Multi-Modal Stock Market
> Prediction under Data Poisoning Attacks

It implements RADS as the paper defines it, not as a single residual outlier
flag. The scalar RADS score is learned from:

1. **S-NDS:** cosine distance between a BERT headline embedding and the
   normal-news reference centroid.
2. **T-NDS:** squared residual from an LSTM trained on normal temporal
   sequences.
3. **NDS:** the negative Isolation Forest normality score for numerical market
   features.

XGBoost fuses these three values into an attack-risk probability. During stock
forecasting, that probability is concatenated with the dual cross-attention
representation. Samples are not removed or filtered at inference time.

## Repository structure

```text
configs/paper.yaml              Paper hyperparameters
data/supplied/                  Data supplied with the manuscript
docs/MANUSCRIPT_CODE_MAP.md     Equations and tables mapped to source files
src/rads/gan.py                 Dense headline GAN from Table 2
src/rads/embeddings.py          Frozen BERT [CLS] extraction
src/rads/temporal.py            Normal-only temporal LSTM
src/rads/rads.py                S-NDS, T-NDS, NDS, and XGBoost fusion
src/rads/model.py               Transformer-LSTM and dual cross-attention
src/rads/pipeline.py            End-to-end chronological experiment
src/rads/ablation.py            Table 4 model ablations
src/rads/component_ablation.py  Table 5 RADS component ablations
src/rads/cross_asset.py         Table 7 cross-entity transfer
tests/                          Unit tests for formulas and time alignment
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[deep,excel,dev]"
```

The first BERT run downloads `bert-base-uncased`. For an offline experiment,
download the model in advance and set `bert_model` in `configs/paper.yaml` to
its local directory.

## Expected input

The end-to-end experiment accepts CSV or Excel data with one row per aligned
news/market observation:

| Column | Meaning |
|---|---|
| `date` | Observation date |
| `headline` | Financial headline |
| chosen price column | Close price to forecast |
| `is_fake` | `0` for real/normal, `1` for generated/manipulated |

Optional volume data can be incorporated through `load_market_data`.

## Run the full model

The supplied `stock.xlsx` contains attack labels and can exercise the pipeline
with either numerical series:

```bash
rads-run \
  --data data/supplied/stock.xlsx \
  --price-column gold_close_price \
  --config configs/paper.yaml
```

Outputs are written under `outputs/<price-column>/full/`.

## Run the paper ablations

```bash
rads-ablation \
  --data data/supplied/stock.xlsx \
  --price-column gold_close_price \
  --config configs/paper.yaml
```

This evaluates the full model, no cross-attention, no text transformer, and no
RADS. `evaluate_rads_component_ablation` evaluates full RADS and removal of
each individual deviation component.

## Run a cross-asset experiment

This command trains all learned components on the source entity and evaluates
the target entity in the later held-out period:

```bash
rads-cross-asset \
  --data path/to/labeled_SYF_WFC.csv \
  --source-column SYF \
  --target-column WFC
```

Run the reverse direction by swapping the two column names.

## Reproducibility and leakage controls

- Splits are chronological; time-series windows are never randomly shuffled
  across train and test.
- The normal-news centroid and Isolation Forest are fitted on training data
  labeled as normal.
- The temporal LSTM is fitted only on normal training sequences.
- BERT is frozen and can be cached.
- The GAN is a training-only module and is not part of inference.
- Seeds are fixed in Python, NumPy, TensorFlow, XGBoost, and Isolation Forest
  where supported.

## Important data limitation

The files supplied with the manuscript do **not** include the exact aligned
Apple and S&P 500 OHLCV/news dataset used for Tables 4-6, nor the 111,971-item
news corpus used before selecting the balanced 1,000-headline subset.
Therefore, this repository does not fabricate the paper's reported numbers.
It implements the stated method and includes the supplied datasets, but exact
numerical reproduction requires the original aligned data and the final
real/fake headline split.

The supplied SYF/WFC file has no attack labels. It is suitable for
within-entity and cross-entity forecasting only after a labeled adversarial
headline set is aligned to those dates.

## Tests

```bash
pytest
ruff check src tests
```

## Citation

Use the manuscript title above and the final published bibliographic details
when they become available.

## Disclaimer

This software is provided for research and educational purposes only. It is
provided "as is", without warranties or guarantees of accuracy, reliability,
fitness for a particular purpose, profitability, or uninterrupted operation.

The authors are not responsible for financial losses, trading decisions,
commercial damages, data loss, security incidents, or other consequences
arising from the use or misuse of this software.

Commercial use is permitted under the MIT License. Users are responsible for
independently validating the software and ensuring compliance with applicable
laws, regulations, third-party licenses, dataset licenses, and pretrained-model
licenses.

## License

MIT. Dataset and pretrained-model licenses remain governed by their original
providers.
