import glob
import os
from collections import namedtuple

import numpy as np
import textgrids
import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from falt_process import process_array

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create namedtuple object to store the extracted activations
Activations = namedtuple("Activations", ["filename", "hidden_state_activations"])
SlicedActivations = namedtuple(
    "SlicedActivations", ["slicename", "hidden_state_activations"]
)


# Load audio file
def extract_activations(
    audio_file: os.PathLike | str,
    model: Wav2Vec2Model,
    feature_extractor: Wav2Vec2FeatureExtractor,
) -> Activations:
    """
    Extracts activations from an audio file using a specified model and feature extractor.
    Args:
        audio_file (os.PathLike | str): Path to the audio file.
        model (Wav2Vec2Model): The model used to generate activations.
        feature_extractor (Wav2Vec2FeatureExtractor): The feature extractor used to process the audio input.
    Returns:
        Activations: An object containing the filename and hidden state activations.
    """

    audio_input, sr = torchaudio.load(audio_file)
    audio_input = audio_input.to(device)

    # Extract features
    input_values = feature_extractor(
        audio_input.squeeze(), return_tensors="pt", sampling_rate=sr
    ).input_values.to(device)

    output = model.forward(input_values, output_hidden_states=True)

    return Activations(
        filename=audio_file,
        hidden_state_activations=torch.stack(output.hidden_states)
        .detach()
        .cpu()
        .numpy(),
    )


def slice_activations(
    activation: Activations, **kwargs
):
    """
    Slice activations based on the specified slicing tier.

    Parameters:
    activation (object): An object containing hidden_state_activations and filename attributes.
    **kwargs: Additional keyword arguments.
        - datapath (str): Path to the directory containing the TextGrid files. Default is "examples/wavs".
        - slicing_tier (str): The tier to slice the activations by. Can be "words", "phones", "utterance", or None.

    Returns:
    object: Sliced activations based on the slicing tier. If slicing_tier is None, returns the original activation.
            If slicing_tier is "words" or "phones", returns a list of SlicedActivations objects.
            If slicing_tier is "utterance", returns a single SlicedActivations object.

    Raises:
    AssertionError: If the shape of hidden_state_activations is not (13, 1, num_frames, 768).
    ValueError: If slicing_tier is not one of "words", "phones", "utterance", or None.
    """
    # First make confirm the activation shape is (13, 1, num_frames, 768)
    try:
        assert np.moveaxis(activation.hidden_state_activations, -2, -1).shape[:-1] == (
            13,
            1,
            768,
        )
    except AssertionError:
        raise AssertionError(
            "The hidden_state_activations shape is not (13, 1, num_frames, 768)"
        )
    # Unpack and set default values from kwargs
    slicing_tier = None if "slicing_tier" not in kwargs else kwargs["slicing_tier"]

    if slicing_tier is None:
        return (
            list(range(activation.hidden_state_activations.shape[-2])),
            "no_slicing",
            activation.hidden_state_activations,
        )
    elif slicing_tier == "words" or slicing_tier == "phones":
        # Load textgrid file
        textgridfile = activation.filename.replace(".wav", ".TextGrid")
        tg = textgrids.TextGrid(textgridfile)
        wordtier = tg[slicing_tier]
        segment_label, sliced_activations = [], []
        for i, word in enumerate(wordtier):
            # print(word.text)
            # Turn xmins and xmaxs into wav2vec2 timesteps
            xmin_frame = int(word.xmin / 0.02)
            xmax_frame = int(word.xmax / 0.02)
            if xmin_frame == xmax_frame:
                sliced_activations.append(
                    activation.hidden_state_activations[:, :, xmin_frame]
                )
            else:
                sliced_activations.append(
                    activation.hidden_state_activations[
                        :, :, xmin_frame:xmax_frame
                    ].mean(-2)
                )
            segment_label.append(word.text)
        return segment_label, slicing_tier, np.stack(sliced_activations, axis = -2)
    elif slicing_tier == "utterance":
        # return SlicedActivations(
        #     slicename=activation.filename,
        #     hidden_state_activations=activation.hidden_state_activations.mean(-2),
        # )
        return (
            activation.filename,
            slicing_tier,
            activation.hidden_state_activations.mean(-2).unsqueeze(-2),
        )

    else:
        raise ValueError(
            "slicing_tier must be either 'words', 'phones', 'utterance' or None"
        )


def save_activations(**kwargs):
    """
    Save activations from a Wav2Vec2 model for a dataset of audio files.
    Keyword Arguments:
    modelname (str): The name of the pre-trained Wav2Vec2 model to use. Defaults to "facebook/wav2vec2-base".
    datapath (str): The path to the directory containing the audio files. Defaults to "examples/".
    savepath (str): The path to the directory where the activations will be saved. Defaults to "examples/activations".
    overwrite (bool): If True, overwrite existing activation files. Defaults to False.
    slicing_params (dict): Additional parameters for slicing activations. Defaults to None.
    """

    # Set default modelname and paths
    modelname = (
        "facebook/wav2vec2-base" if "modelname" not in kwargs else kwargs["modelname"]
    )
    datapath = "examples/" if "datapath" not in kwargs else kwargs["datapath"]

    savepath = (
        "examples/activations" if "savepath" not in kwargs else kwargs["savepath"]
    )

    slicing_tier = None if "slicing_tier" not in kwargs else kwargs["slicing_tier"]

    output_file = f"{savepath}/{modelname.replace('/', '-')}-{os.path.dirname(datapath)}-{slicing_tier}.pt"

    if os.path.exists(output_file) and not kwargs.get("overwrite", False):
        print(f"Activations already exist at {output_file}. Skipping...")
        return

    # Load model
    model = Wav2Vec2Model.from_pretrained(modelname).to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(modelname)

    # Load audio files
    audio_files = glob.glob(f"{datapath}/**/*.wav", recursive=True)

    all_activations = []
    for audio_file in tqdm(audio_files):
        activations = extract_activations(audio_file, model, feature_extractor)
        activations = process_array(*activations, **kwargs)
        # activations = slice_activations(activations, **kwargs)
        all_activations.append(activations)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    torch.save(all_activations, output_file)
    print(f"Saved activations to {output_file}")


if __name__ == "__main__":
    kwargs = {
        "modelname": "facebook/wav2vec2-base",
        "datapath": "examples/",
        "slicing_tier": "phones",
        "savepath": "examples/activations",
        "overwrite": True,
    }
    save_activations(**kwargs)
