import glob
import os
from collections import namedtuple

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .falt_process import process_array

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create namedtuple object to store the extracted activations
Activations = namedtuple("Activations", ["filename", "hidden_state_activations"])


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


def extract_and_save_processed_activations(**kwargs):
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
    datasetname = "examples" if "datasetname" not in kwargs else kwargs["datasetname"]

    slicing_tier = None if "slicing_tier" not in kwargs else kwargs["slicing_tier"]

    savepath = os.path.realpath(savepath)
    datapath = os.path.realpath(datapath)
    output_file = f"{savepath}/{modelname.replace('/', '-')}-{datasetname}-{slicing_tier}.pt" if "output_file" not in kwargs else kwargs["output_file"]

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
        all_activations.append(activations)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    torch.save(all_activations, output_file)
    print(f"Saved activations to {output_file}")


if __name__ == "__main__":
    # Demo: run from the repository root with
    #   python -m falt.generate_activations
    # Paths are repo-relative; the model is downloaded from Hugging Face on first use.
    extract_and_save_processed_activations(
        modelname="facebook/wav2vec2-base",
        datapath="examples/wavs",
        slicing_tier="phones",
        savepath="examples/activations",
        overwrite=True,
    )
