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

        textgrid_file = f"examples/textgrids/TH{spklet}{str(int(spknum)+100)[1:]}-{str(int(uttid)+1000)[1:]}.TextGrid"

        new_textgrid_file = f"{os.path.splitext(wavfile)[0]}.TextGrid"

        # Load the existing TextGrid file
        tg = textgrids.TextGrid(textgrid_file)
        
        
        # Save the TextGrid file with the new filename
        tg.write(new_textgrid_file)
        
    print(f"Resaved {len(wavfiles)} TextGrid files with the new filename")
