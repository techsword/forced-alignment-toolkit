import numpy as np
import textgrids


def process_array(filename: str, array: np.ndarray, **kwargs) -> tuple:
    """
    Processes the given array based on the slicing tier specified in kwargs.

    Args:
        filename (str): The filename of the corresponding audio file.
        array (np.ndarray): The input array with shape
            (n_layers, n_channels, n_frames, hidden_size). Any positive sizes
            are accepted. For example, a 13-layer model with one channel,
            100 frames, and a hidden size of 768 has shape (13, 1, 100, 768).

    Keyword Args:
        slicing_tier (str, optional): The tier to slice the array by.
            Can be 'words', 'phones', 'utterance', or None. Defaults to None.

    Raises:
        ValueError: If the input array is not 4-D or if any dimension is zero.
        NotImplementedError: If the slicing_tier is not 'words', 'phones', 'utterance', or None.

    Returns:
        tuple: A tuple containing:
            - list: A list of segment labels or frame indices.
            - str: The slicing tier used.
            - np.ndarray: The processed array.
    """
    if array.ndim != 4:
        raise ValueError(
            "Expected a 4-D array with shape "
            "(n_layers, n_channels, n_frames, hidden_size), "
            f"but got {array.ndim} dimension(s)."
        )
    if any(dim == 0 for dim in array.shape):
        raise ValueError(
            "Expected positive dimensions with shape "
            "(n_layers, n_channels, n_frames, hidden_size), "
            f"but got shape {array.shape}."
        )
    # Unpack and set default values from kwargs
    slicing_tier = None if "slicing_tier" not in kwargs else kwargs["slicing_tier"]

    if slicing_tier is None:
        return (
            list(range(array.shape[-2])),
            "no_slicing",
            array,
        )
    elif slicing_tier == "words" or slicing_tier == "phones":
        # Load textgrid file
        textgridfile = filename.replace(".wav", ".TextGrid")
        tg = textgrids.TextGrid(textgridfile)
        wordtier = tg[slicing_tier]
        segment_label, sliced_activations = [], []
        for i, word in enumerate(wordtier):
            # print(word.text)
            # Turn xmins and xmaxs into wav2vec2 timesteps
            xmin_frame = int(word.xmin / 0.02)
            xmax_frame = int(word.xmax / 0.02)
            if xmin_frame == xmax_frame:
                sliced_activations.append(array[:, :, xmin_frame])
            else:
                sliced_activations.append(array[:, :, xmin_frame:xmax_frame].mean(-2))
            segment_label.append(word.text)
        return segment_label, slicing_tier, np.stack(sliced_activations, axis=-2)
    elif slicing_tier == "utterance":
        return (
            [filename],
            slicing_tier,
            np.expand_dims(array.mean(-2), axis=-2),
        )

    else:
        raise NotImplementedError(
            "slicing_tier must be either 'words', 'phones', 'utterance' or None"
        )
