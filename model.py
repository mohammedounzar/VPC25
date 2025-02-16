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

import numpy as np
import scipy
from scipy.io import wavfile

def apply_mcadams(sig, fs, winLengthinms=25, shiftLengthinms=10, lp_order=16, mcadams_factor=0.7):
    """
    Apply McAdams transformation to the audio.
    
    :param sig: Input audio signal.
    :param fs: Sample rate of the audio.
    :param winLengthinms: Window length in milliseconds.
    :param shiftLengthinms: Shift length in milliseconds.
    :param lp_order: Order of the LPC model.
    :param mcadams_factor: McAdams transformation factor.
    :return: Anonymized audio signal.
    """
    winlen = int(winLengthinms * 0.001 * fs)
    shift = int(shiftLengthinms * 0.001 * fs)
    length_sig = len(sig)

    NFFT = 2 ** int(np.ceil(np.log2(winlen)))
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    Nframes = 1 + int((length_sig - winlen) / shift)

    sig_rec = np.zeros(length_sig)

    for m in range(Nframes):
        index = range(m * shift, min(m * shift + winlen, length_sig))
        frame = sig[index] * win

        # Compute LPC coefficients
        a_lpc = librosa.lpc(frame, order=lp_order)

        poles = scipy.signal.tf2zpk([1], a_lpc)[1]
        ind_imag = np.where(np.isreal(poles) == False)[0]
        ind_imag_con = ind_imag[np.arange(0, len(ind_imag), 2)]

        # Apply McAdams transformation
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams_factor
        new_angles = np.clip(new_angles, 0, np.pi)

        new_poles = poles.copy()
        for k in range(len(ind_imag_con)):
            new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
            new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])

        a_lpc_new = np.real(np.poly(new_poles))
        res = scipy.signal.lfilter(a_lpc, [1], frame)
        frame_rec = scipy.signal.lfilter([1], a_lpc_new, res)
        frame_rec = frame_rec * win

        outindex = range(m * shift, m * shift + len(frame_rec))
        sig_rec[outindex] += frame_rec

    return sig_rec

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

    # Load the audio file
    sig, fs = librosa.load(input_audio_path, sr=None)

    # Apply McAdams transformation
    sig_anonymized = apply_mcadams(sig, fs, mcadams_factor=0.7)

    # Normalize the audio to avoid clipping
    sig_anonymized = sig_anonymized / np.max(np.abs(sig_anonymized))

    noise_sample = np.float32(sig_anonymized)[int(0 * fs) : int(1 * fs)]
    
    # Perform noise reduction
    y_denoised = nr.reduce_noise(y=np.float32(sig_anonymized), sr=fs, y_noise=noise_sample, prop_decrease=1.0)

    # Apply your anonymization algorithm

    # Output:
    audio = y_denoised
    sr = fs
    
    return audio, sr