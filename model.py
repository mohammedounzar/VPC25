#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>

import librosa

def anonymize(input_audio_path):
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy array
        numpy array representing the anonymized audio wavefrom
    sr : int
        sampling rate of the anonymized audio
    """

    # Read the source audio file

    # Apply your anonymization algorithm
    
    # Output:
    audio = ... # numpy array representing the anonymized audio wavefrom
    sr = ... # sampling rate of the anonymized audio
    return audio, sr