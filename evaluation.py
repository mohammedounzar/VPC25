import os
import logging
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from jiwer import wer
from model import anonymize
import torch
import time
logging.getLogger("transformers").setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup logging to only output to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def transcribe_audio(audio_path):
    """
    Transcribe audio from a .wav file path.
    """
    if not os.path.exists(audio_path):
        error_msg = f"Audio file not found: {audio_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device = device)
        transcription = asr_pipeline(audio_path)
        return transcription['text'].lower()
    except Exception as e:
        error_msg = f"Error transcribing {audio_path}: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

def compute_we(input_audio_path, anonymized_audio_path):
    """
    Compute Word Error Rate (WE = WER * N) given the input and anonymized audio.
    """
    original = transcribe_audio(input_audio_path)
    anonymized = transcribe_audio(anonymized_audio_path)

    if original is None or anonymized is None:
        error_msg = f"Failed transcription for {input_audio_path}. Stopping evaluation."
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    words = len(original.split())
    we = wer(original, anonymized) * words
    return we, words

def compute_total_eer(input_dir_path, anonymized_dir_path):
    """
    Placeholder function for EER computation.
    """
    # Implement EER computation logic here.
    return 0.0

def evaluate(input_directory, output_directory, anonymization_algorithm):
    """
    Evaluate the anonymization algorithm by computing WER and EER.
    """
    if not os.path.exists(input_directory):
        error_msg = f"Input directory does not exist: {input_directory}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not os.path.exists(output_directory):
        logging.info(f"Creating output directory: {output_directory}")
        os.makedirs(output_directory)

    audio_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    if not audio_files:
        error_msg = f"No audio files found in input directory: {input_directory}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    total_wer = 0
    total_words = 0

    start = time.time()

    for filename in tqdm(audio_files, desc="Anonymizing Audio Files"):
        input_audio_path = os.path.join(input_directory, filename)
        anonymized_audio_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_anonymized.wav")

        try:
            anonymized_audio, sr = anonymization_algorithm(input_audio_path)
            sf.write(anonymized_audio_path, anonymized_audio, sr)
        except Exception as e:
            error_msg = f"Error processing {filename}: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        we, reference_length = compute_we(input_audio_path, anonymized_audio_path)
        total_wer += we
        total_words += reference_length

    eer = compute_total_eer(input_directory, output_directory)

    if total_words == 0:
        error_msg = "No valid reference transcriptions found. Cannot compute WER."
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    
    end = time.time()

    avg_wer = total_wer / total_words
    results = pd.DataFrame([{"WER": avg_wer, "EER": eer, "Runtime (s)": end - start}])
    results.to_csv("results.csv", index=False)

    logging.info("Evaluation completed successfully. Results saved to results.csv.")

if __name__ == "__main__":
    input_directory = "source_audio/"
    output_directory = "anonymized_audio/"

    try:
        evaluate(input_directory, output_directory, anonymize)
    except Exception as e:
        logging.critical(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")
        exit(1)
