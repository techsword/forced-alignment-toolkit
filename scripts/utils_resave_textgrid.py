"""Standalone maintenance script: resave THA TextGrids with wavfile-style names.

This is a one-off utility, not part of the installed ``falt`` package. It was
moved out of ``falt/utils.py`` because nothing in the package imports it.

TODO(maintainer): revisit this script. The source directory it expects,
``examples/textgrids/``, was deleted as stale example data. Update the
``textgrid_file`` path below to point at confirmed-good TextGrid sources
before using it.
"""

import os
import glob

import textgrids


def resave_THA_textgrid_with_wavfile_name():
    # Get all the wav files in the examples directory
    wavfiles = glob.glob("examples/wavs/*.wav")

    # Resave the TextGrid files with the new filename
    for wavfile in wavfiles:
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(wavfile))[0]
        spkid, uttid = filename.split("_")
        spknum = spkid[1:]
        spklet = spkid[0]

        # TODO(external): the source TextGrids previously lived under
        # examples/textgrids/, which was deleted as stale example data.
        textgrid_file = f"examples/textgrids/TH{spklet}{str(int(spknum)+100)[1:]}-{str(int(uttid)+1000)[1:]}.TextGrid"

        new_textgrid_file = f"{os.path.splitext(wavfile)[0]}.TextGrid"

        # Load the existing TextGrid file
        tg = textgrids.TextGrid(textgrid_file)

        # Save the TextGrid file with the new filename
        tg.write(new_textgrid_file)

    print(f"Resaved {len(wavfiles)} TextGrid files with the new filename")
