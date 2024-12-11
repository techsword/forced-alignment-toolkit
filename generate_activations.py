import glob
import os
from collections import namedtuple

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create namedtuple object to store the extracted activations
Activations = namedtuple("Activations", ["filename", "hidden_state_activations"])


# Load audio file
def extract_activations(audio_file, model, feature_extractor):
    audio_input, sr = torchaudio.load(audio_file)
    audio_input = audio_input.to(device)

    # Extract features
    input_values = feature_extractor(
        audio_input.squeeze(), return_tensors="pt", sampling_rate=sr
    ).input_values.to(device)

    output = model.forward(input_values, output_hidden_states=True)

    return Activations(
        filename=os.path.basename(audio_file),
        hidden_state_activations=torch.stack(output.hidden_states)
        .detach()
        .cpu()
        .numpy(),
    )


def main():
    # Set default modelname and paths
    modelname = "facebook/wav2vec2-base"
    datapath = "examples/wavs"
    savepath = "examples/activations"

    # Load model
    model = Wav2Vec2Model.from_pretrained(modelname).to(device)
    model.eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(modelname)

    # Load audio files
    audio_files = glob.glob(f"{datapath}/*.wav")

    all_activations = []
    for audio_file in tqdm(audio_files):
        activations = extract_activations(audio_file, model, feature_extractor)
        all_activations.append(activations)

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    torch.save(all_activations, f"{savepath}/activations.pt")
    print(f"Saved activations to {savepath}/activations.pt")


if __name__ == "__main__":
    main()
