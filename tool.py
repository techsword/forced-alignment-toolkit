import os
import glob
from pydoc import text
import numpy as np
import pandas as pd
import torch

import textgrids
from generate_activations import Activations
from collections import namedtuple

# Create namedtuple object to store the sliced activations
SlicedActivations = namedtuple("SlicedActivations", ["slicename", "hidden_state_activations"])

def load_activations(activations_file):
    # Load pre-saved activations
    if not os.path.exists(activations_file):
        raise FileNotFoundError(f"File {activations_file} not found")
    activations = torch.load(activations_file)
    return activations

def read_textgrid_file(textgrid_file):

def transform_wavfile_to_textgridfile(wavfile):
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(wavfile))[0]
    spkid, uttid = filename.split("_")
    spknum = spkid[1:]
    spklet = spkid[0]

    textgrid_file = f"examples/textgrids/TH{spklet}{str(int(spknum)+100)[1:]}-{str(int(uttid)+1000)[1:]}.TextGrid"

    return textgrid_file

def main():
    activations_file = "examples/activations/activations.pt"
    activations = load_activations(activations_file)
    for activation in activations:
        print(activation.filename)
        print(activation.hidden_state_activations.shape)
        textgridfile = transform_wavfile_to_textgridfile(activation.filename)
        tg = textgrids.TextGrid(textgridfile)
        wordtier = tg['words']
        sliced_activations = []
        for i, word in enumerate(wordtier):
            # print(word.text)
            # Turn xmins and xmaxs into wav2vec2 timesteps
            xmin_frame = int(word.xmin / 0.02)
            xmax_frame = int(word.xmax / 0.02)
            sliced_activations.append(SlicedActivations(slicename=word.text, hidden_state_activations=activation.hidden_state_activations[:, :, xmin_frame:xmax_frame].mean(-2)))

        word_activations = np.stack([x[1] for x in sliced_activations], axis = -2)
        


        

        break