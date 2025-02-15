#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################
import librosa

def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`, 
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """

    arr_audio,fr=librosa.load(input_audio_path,sr=None)
    audio_shifted=librosa.effects.pitch_shift(arr_audio,sr=fr,n_steps=3.0,scale=False)
    audio_stretched = librosa.effects.time_stretch(audio_shifted,rate=1)
    # sf.write(output_file_name, audio_stretched, sr, subtype='PCM_24')

    # Read the source audio file

    # Apply your anonymization algorithm
    
    # Output:
    audio = audio_stretched
    sr = fr
    
    return audio, sr