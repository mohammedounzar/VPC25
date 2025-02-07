##############################################
# DO NOT MODIFY THIS FILE
##############################################


import logging
import torch
import numpy as np
import os
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from jiwer import wer
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torchaudio
from speechbrain.inference import SpeakerRecognition
from model import anonymize
import soundfile as sf
import sys

import warnings
warnings.simplefilter("ignore", FutureWarning)

# Setup logging to only output to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("transformers").setLevel(logging.ERROR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model once (optimize for repeated evaluations)
VERIFICATION_MODEL = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

def compute_total_eer(evaluation_data_path):
    """Calculate EER for an anonymization model with robust error handling."""
    
    original_dir = Path(original_dir)
    anonymized_dir = Path(anonymized_dir)

    if not original_dir.exists():
        error_msg = f"Original directory does not exist: {original_dir}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if not anonymized_dir.exists():
        error_msg = f"Anonymized directory does not exist: {anonymized_dir}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    original_files = []
    anonymized_files = []

    # Walk through directories to find matching speaker/utterance pairs
    for orig_path in original_dir.rglob("*.wav"):
        rel_path = orig_path.relative_to(original_dir)
        ano_path = anonymized_dir / rel_path

        if ano_path.exists():
            original_files.append(orig_path)
            anonymized_files.append(ano_path)
        else:
            logging.warning(f"Anonymized file not found: {ano_path}")

    if not original_files or not anonymized_files:
        error_msg = "No valid audio file pairs found in the provided directories."
        logging.error(error_msg)
        raise ValueError(error_msg)

    def _get_embeddings(file_list):
        """Extract embeddings with error handling."""
        embeddings = []
        for file in tqdm(file_list, desc="Computing embeddings"):
            try:
                signal, sr = torchaudio.load(file)
                
                if signal.shape[1] == 0:
                    logging.warning(f"Empty audio file: {file}")
                    continue

                if signal.shape[0] > 1:  # Convert to mono
                    signal = torch.mean(signal, dim=0, keepdim=True)

                emb = VERIFICATION_MODEL.encode_batch(signal).squeeze().cpu().numpy()
                embeddings.append(emb)
            
            except Exception as e:
                logging.error(f"Error computing embeddings for {file}: {e}")
        
        if not embeddings:
            error_msg = "Failed to compute embeddings for any audio file."
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        return np.array(embeddings)

    try:
        orig_embeddings = _get_embeddings(original_files)
        ano_embeddings = _get_embeddings(anonymized_files)
    except RuntimeError as e:
        raise RuntimeError(f"Embedding extraction failed: {e}")

    if orig_embeddings.shape[0] != ano_embeddings.shape[0]:
        error_msg = "Mismatch in the number of embeddings for original and anonymized audio."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Compute genuine scores
    try:
        genuine_scores = np.sum(orig_embeddings * ano_embeddings, axis=1) / (
            np.linalg.norm(orig_embeddings, axis=1) * np.linalg.norm(ano_embeddings, axis=1)
        )
    except Exception as e:
        logging.error(f"Error computing genuine scores: {e}")
        raise RuntimeError("Failed to compute genuine scores.")

    # Compute impostor scores
    impostor_scores = []
    rng = np.random.default_rng(seed=42)  # For reproducibility
    num_impostor_pairs = min(len(orig_embeddings), 1000)  # Limit to avoid excessive computation
    
    for _ in tqdm(range(num_impostor_pairs), desc="Calculating impostors"):
        try:
            i, j = rng.choice(len(orig_embeddings), 2, replace=False)
            score = np.dot(orig_embeddings[i], ano_embeddings[j]) / (
                np.linalg.norm(orig_embeddings[i]) * np.linalg.norm(ano_embeddings[j])
            )
            impostor_scores.append(score)
        except Exception as e:
            logging.error(f"Error computing impostor score: {e}")

    if not impostor_scores:
        error_msg = "Failed to compute impostor scores."
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # Compute EER
    try:
        y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
        y_score = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer
    except Exception as e:
        logging.error(f"Error computing EER: {e}")
        raise RuntimeError("Failed to compute EER.")


def transcribe_audio(audio_path):
    """
    Transcribe audio from a .wav file path.
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        error_msg = f"Audio file not found: {audio_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=device)
        transcription = asr_pipeline(str(audio_path))
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

def evaluate(evaluation_data_path, anonymization_algorithm):
    """
    Evaluate the anonymization algorithm by computing WER and EER.
    """

    enrollment_directory = os.path.join(evaluation_data_path, "Enrollment")
    trial_directory = os.path.join(evaluation_data_path, "Trial")

    if not os.path.exists(enrollment_directory):
        error_msg = f"Enrollment directory does not exist: {enrollment_directory}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not os.path.exists(trial_directory):
        logging.info(f"Creating output directory: {trial_directory}")
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    total_wer = 0
    total_words = 0
    start = time.time()

    # Process Enrollment and Trial directories
    for subset_directory in [enrollment_directory, trial_directory]:
        subset_name = os.path.basename(subset_directory)  # "Enrollment" or "Trial"

        for speaker in os.listdir(subset_directory):
            speaker_path = os.path.join(subset_directory, speaker)
            if not os.path.isdir(speaker_path):
                continue

            # Ensure the anonymized directory exists
            anonymized_dir = os.path.join(speaker_path, "anonymized")
            os.makedirs(anonymized_dir, exist_ok=True)

            # Collect all .wav files
            audio_files = [
                f for f in os.listdir(speaker_path) 
                if f.lower().endswith('.wav')
            ]

            # If there are no .wav files, log and continue
            if not audio_files:
                logging.warning(f"No audio files found for speaker: {speaker_path}")
                continue

            for filename in tqdm(audio_files, desc=f"Processing {subset_name}/{speaker}"):
                input_audio_path = os.path.join(speaker_path, filename)

                # Ensure file is strictly .wav (reject other formats)
                if not filename.endswith('.wav'):
                    error_msg = f"Invalid file format detected: {filename}. Only .wav files are allowed."
                    logging.error(error_msg)
                    raise ValueError(error_msg)

                # Generate anonymized file path
                anonymized_audio_path = os.path.join(anonymized_dir, f"anon_{filename}")

                try:
                    # Anonymization process
                    anonymized_audio, sr = anonymization_algorithm(input_audio_path)
                    sf.write(anonymized_audio_path, anonymized_audio, sr)
                except Exception as e:
                    logging.error(f"Error anonymizing {filename}: {e}")
                    continue

                try:
                    # Compute WER
                    we, reference_length = compute_we(input_audio_path, anonymized_audio_path)
                    total_wer += we
                    total_words += reference_length
                except Exception as e:
                    logging.error(f"Error computing WER for {filename}: {e}")
                    continue

    eer = 0 #compute_total_eer(evaluation_data_path)

    end = time.time()

    avg_wer = total_wer / total_words
    results = pd.DataFrame([{"WER": avg_wer, "EER": eer, "Runtime (s)": end - start}])
    results.to_csv("results.csv", index=False)

    logging.info("Evaluation completed successfully. Results saved to results.csv.")

if __name__ == "__main__":
    try:
        evaluation_data_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation_data/"
        evaluate(evaluation_data_path, anonymize)
    except Exception as e:
        logging.critical(f"Evaluation failed: {e}")
        exit(1)


