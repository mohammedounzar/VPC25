# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################
import librosa
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf
import numpy as np
import scipy.signal
def apply_mcadams(sig, fs, winLengthinms=25, shiftLengthinms=10, lp_order=16, mcadams_factor=0.65):
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
@@ -26,18 +83,31 @@ def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
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
    audio = audio_stretched
    audio = y_denoised
    sr = fr

    return audio, sr