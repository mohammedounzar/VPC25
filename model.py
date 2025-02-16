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
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf

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
    """
    Denoise an audio file using spectral gating.
    
    :param input_file: Path to the input audio file.
    :param output_file: Path to save the denoised audio file.
    :param noise_start: Start time (in seconds) of the noise sample in the audio.
    :param noise_end: End time (in seconds) of the noise sample in the audio.
    """
    # Load the audio file

    arr_audio,fr=librosa.load(input_audio_path,sr=None)
    audio_shifted=librosa.effects.pitch_shift(arr_audio,sr=fr,n_steps=3.0,scale=False)
    audio_stretched = librosa.effects.time_stretch(audio_shifted,rate=1)
    # sf.write(output_file_name, audio_stretched, sr, subtype='PCM_24')

    # Read the source audio file
    noise_sample = audio_stretched[int(0 * fr) : int(1 * fr)]
    
    # Perform noise reduction
    y_denoised = nr.reduce_noise(y=audio_stretched, sr=fr, y_noise=noise_sample, prop_decrease=1.0)

    # Apply your anonymization algorithm

    # Output:
    audio = y_denoised
    sr = fr
    
    return audio, sr