#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################



def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    output_audio_path : str
        path to the anonymized audio file in one ".wav" format. Should have the same name as the source audio file but save in the anonymized_audio/ directory
    """

    # Read the source audio file

    # Apply your anonymization algorithm and save the anonymized audio file in .wav format in the anonymized_audio/ directory with the same file name as the source audio file
    
    # Output:
    output_audio_path = ... # path to the anonymized audio file (in one ".wav" format, should have the same name as the source audio file but save in the anonymized_audio/ directory)
    
    
    return output_audio_path