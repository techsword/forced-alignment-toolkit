# forced-alignment-toolkit

Using forced alignment timestamps to process time-series data.

## Preliminary goal

The preliminary goal of this repo is to create a re-usable tool that can transform the (hidden-state) layer output of audio transformer models like wav2vec2 into something similar to a BERT model layerwise output.

## Installation

Requires Python 3.10+. Editable install pulls in `torch`/`torchaudio` (>=2.6) and `transformers` (>=4.48):

```bash
pip install -e .
```

Or install from git:

```bash
pip install git+https://github.com/techsword/forced-alignment-toolkit.git
```

### Torch / GPU note

`pip install -e .` resolves `torch>=2.6,<3` and `torchaudio>=2.6,<3` from PyPI. On Linux the default PyPI wheel is CUDA-enabled (it also runs on CPU-only hosts). To pin a specific CUDA or CPU-only build, install `torch`/`torchaudio` from https://pytorch.org/get-started/locally/ *before* installing this package.

## Usage

The toolkit pools wav2vec2 hidden states into segments defined by a forced-alignment TextGrid. The `.wav` file and its `.TextGrid` file must sit in the same directory.

```python
from falt import extract_and_save_processed_activations

extract_and_save_processed_activations(
    modelname="facebook/wav2vec2-base",
    datapath="examples/wavs",
    savepath="examples/activations",
    slicing_tier="phones",  # "words", "phones", "utterance", or None
    overwrite=True,
)
```

This globs every `.wav` under `datapath`, extracts hidden states, slices them by the chosen tier, and saves the result to a `.pt` file under `savepath`. The saved file contains a list of `(labels, slicing_tier, activations)` tuples.

<!-- ============================================================
     DRAFT (for maintainer review) — first-use model download note.
     Wording below is a draft; the maintainer finalizes it.
     ============================================================ -->
> **DRAFT (for maintainer review).** The wav2vec2 checkpoint (`facebook/wav2vec2-base`) is downloaded from Hugging Face on first use and cached locally. Later runs with the same `modelname` reuse the cache and do not download again.

<!-- ============================================================ -->

For finer control, use the lower-level functions directly:

```python
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from falt import extract_activations
from falt.falt_process import process_array

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

activations = extract_activations("examples/wavs/A2_0.wav", model, feature_extractor)
labels, tier, sliced = process_array(
    activations.filename, activations.hidden_state_activations, slicing_tier="phones"
)
```

## Example files

<!-- ============================================================
     DRAFT (for maintainer review) — examples/ layout description.
     Wording below is a draft; the maintainer finalizes it.
     ============================================================ -->
> **DRAFT (for maintainer review).** `examples/` layout:
> - `examples/wavs/` — example audio (`.wav`) and the matching forced-alignment TextGrids (`.TextGrid`).
> - `examples/activations/` — output directory written by the examples (created on first run; not tracked by git).

<!-- ============================================================ -->

Example data structure can be found under `examples`. The examples are taken from the THCHS-30 dataset.
```
@misc{THCHS30_2015,
  title={THCHS-30 : A Free Chinese Speech Corpus},
  author={Dong Wang, Xuewei Zhang, Zhiyong Zhang},
  year={2015},
  url={http://arxiv.org/abs/1512.01882}
}
```
