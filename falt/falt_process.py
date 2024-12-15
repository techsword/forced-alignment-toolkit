import numpy as np
import textgrids


def process_array(array: np.array, filename: str, **kwargs) -> tuple:
    """
    Processes the given array based on the slicing tier specified in kwargs.

        array (np.array): The input array with shape (13, 1, num_frames, 768).
        filename (str): The filename of the corresponding audio file.

    Args:
        array (np.array): _description_
        filename (str): _description_

    Keyword Args:
        slicing_tier (str, optional): The tier to slice the array by.
            Can be 'words', 'phones', 'utterance', or None. Defaults to None.

    Raises:
        ValueError: If the shape of the input array is not (13, 1, num_frames, 768).
        NotImplementedError: If the slicing_tier is not 'words', 'phones', 'utterance', or None.

    Returns:
        tuple: A tuple containing:
            - list: A list of segment labels or frame indices.
            - str: The slicing tier used.
            - np.array: The processed array.
    """
    # First make confirm the activation shape is (13, 1, num_frames, 768)
    try:
        assert np.moveaxis(array, -2, -1).shape[:-1] == (
            13,
            1,
            768,
        )
    except AssertionError as exc:
        raise ValueError(
            "The hidden_state_activations shape is not (13, 1, num_frames, 768)"
        ) from exc
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
        # return SlicedActivations(
        #     slicename=filename,
        #     hidden_state_activations=array.mean(-2),
        # )
        return (
            [filename],
            slicing_tier,
            array.mean(-2).unsqueeze(-2),
        )

    else:
        raise NotImplementedError(
            "slicing_tier must be either 'words', 'phones', 'utterance' or None"
        )
