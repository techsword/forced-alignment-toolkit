# Example data attribution

The example audio and forced-alignment TextGrids in `wavs/` are redistributed
here so that `python -m falt.generate_activations` runs out of the box.

## Audio

- **Source**: LibriSpeech `dev-clean` split.
- **License**: CC BY 4.0.
- **Citation**: Panayotov et al. (2015), *Librispeech: An ASR corpus based on
  public domain audio books*, ICASSP.
- **Archive**: OpenSLR SLR12, <https://www.openslr.org/12>.
- **Hugging Face mirror**: <https://huggingface.co/datasets/openslr/librispeech_asr>.

The audio is 16 kHz mono WAV, converted from the original FLAC.

## Alignments (TextGrids)

- **Source**: [`gilkeyio/librispeech-alignments`](https://huggingface.co/datasets/gilkeyio/librispeech-alignments)
  (CC BY 4.0).
- **Method**: Montreal Forced Aligner (MFA).
- **Citation**: McAuliffe et al. (2017), *Montreal Forced Aligner: Trainable
  Text-Speech Alignment Using Kaldi*, Interspeech.

Each `.TextGrid` has two IntervalTiers, `phones` and `words`, built from the
start/end alignments in that dataset. Empty (non-speech) regions are labelled
`[SIL]`. Tier `xmax` equals the WAV duration.

## Files

| File | Utterance (LibriSpeech id) | Duration |
| --- | --- | --- |
| `3752-4944-0058.wav` | "i'll report this to the government" | 2.05 s |
| `2428-83699-0004.wav` | "the whole thing was a trifle odd" | 1.88 s |
| `1988-147956-0003.wav` | "now why is that otto" | 2.50 s |

Each `.wav` has a same-basename `.TextGrid`.
