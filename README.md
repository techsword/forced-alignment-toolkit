# forced-alignment-toolkit
Using forced alignment timestamp to process time-series data

## Preliminary goal

The preliminary goal of this repo is to create a re-usable tool that can transform the (hidden-state) layer output of audio transformer models like wav2vec2 into something similar to a BERT model layerwise output.

## Example files

Example data structure can be found under `examples`. The examples are taken from the THCHS-30 dataset.
```
@misc{THCHS30_2015,
  title={THCHS-30 : A Free Chinese Speech Corpus},
  author={Dong Wang, Xuewei Zhang, Zhiyong Zhang},
  year={2015},
  url={http://arxiv.org/abs/1512.01882}
}
```