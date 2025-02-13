# **Voice Privacy Challenge**

Welcome to the **Voice Privacy Challenge**! Your task is to develop a model that anonymizes audio while preserving intelligibility and naturalness. This repository provides the necessary setup, evaluation script, and rules for participation.

---

## **📂 Repository Structure**

```
evaluation_data/       # Directory containing enrollment and trial audio data
│── Enrollment/        # Speaker audio files for enrollment
│   ├── speaker1/      # Directory for Speaker 1
│   │   ├── 1272-128104-0000.wav  # Original enrollment utterance
│   │   ├── ...
│   │   └── anonymized/            # Anonymized versions of the above audio files (Will be automatically created based on your anonymization algorithm when the evaluation script is run)
│   │       ├── anon_1272-128104-0000.wav
│   │       ├── ...
│   ├── speaker2/
│   ├── speaker3/
│   ├── speaker4/
│   └── ...
│
│── Trial/             # Speaker audio files for testing (trial phase)
│   ├── speaker1/
│   │   ├── 1272-128104-0003.wav  # Trial utterances (different from enrollment)
│   │   ├── ...
│   │   └── anonymized/           # Anonymized versions of the above audio files (Will be automatically created based on your anonymization algorithm when the evaluation script is run)
│   │       ├── anon_1272-128104-0003.wav
│   │       ├── ...
│   ├── speaker2/
│   ├── speaker3/
│   ├── speaker4/
│   └── ...
│
parameters/            # Directory to store model parameters (participants should add their own)
evaluation.py          # DO NOT MODIFY - Evaluates your model and generates results.csv
model.py               # MODIFY - Implement your anonymization model here
README.md              # This file - contains all competition instructions
requirements.txt       # MODIFY - List your dependencies here
run.sh                 # DO NOT MODIFY - Runs the evaluation script
```

---

### **🗂 Understanding Enrollment and Trial Data**

In this challenge, participants work with **enrollment** and **trial** utterances, which follow a structure similar to speaker verification tasks.

- **Enrollment Utterances** (Stored in `Enrollment/`):
  - These are speech recordings associated with a particular speaker.
  - Each speaker has multiple enrollment utterances, which serve as reference data.
  - The anonymization system must ensure that any transformed enrollment utterance still preserves the necessary speech characteristics, except for the speaker's identity.

- **Trial Utterances** (Stored in `Trial/`):
  - These are new speech recordings from the same speakers but contain different utterances.
  - These utterances are anonymized and later compared against enrollment utterances.
  - The anonymization system must ensure that the same speaker's trial utterances still match their anonymized enrollment utterances while preventing identification of the original speaker.

### **🔑 Key Properties**
- Each **speaker in Enrollment and Trial is the same**, meaning `speaker1` in `Enrollment/` is the same as `speaker1` in `Trial/`, but their audio files differ.
- The anonymized versions of a speaker’s **trial utterances must match the anonymized version of their enrollment utterances**, maintaining consistency in the "pseudo-speaker" identity.
- The anonymization system should **not alter linguistic content** but should make it impossible to link the anonymized voice back to the original speaker.
---
## **🚀 Getting Started**

### **1️⃣ Fork the Repository**

Before cloning, you need to **fork** this repository to your own GitHub account. Follow these steps:

1. Navigate to the repository on GitHub.
2. In the top-right corner, click the **Fork** button.
3. This creates a copy of the repository under your GitHub account.

---

### **2️⃣ Clone Your Forked Repository**

Once you've forked the repository, clone it to your local machine:

```sh
# Replace <YOUR_GITHUB_USERNAME> with your actual GitHub username
git clone https://github.com/<YOUR_GITHUB_USERNAME>/VPC25.git
cd VPC25
```

This ensures you're working on your own version of the repository while still being able to pull updates from the original source.

---

### **3️⃣ Set Up Your Environment**

This project requires **Python 3.12**. Ensure you have it installed before proceeding.

#### **Check your Python version:**

```sh
python3 --version
```

or on Windows (PowerShell):

```powershell
python --version
```

If you don't have Python 3.12, download it from [python.org](https://www.python.org/downloads/).

#### **Install FFmpeg (Required for Audio Processing)**

To process audio files, **FFmpeg** must be installed. Follow these steps based on your system:

##### **Linux**
```sh
sudo apt update && sudo apt install ffmpeg
```

##### **macOS**
```sh
brew install ffmpeg
```

##### **Windows**
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) (recommended: Windows build from gyan.dev).
2. Extract it to a folder (e.g., `C:\ffmpeg`).
3. Add `C:\ffmpeg\bin` to your **system PATH** to make FFmpeg accessible from the command line.
4. Verify installation by running:
   ```sh
   ffmpeg -version
   ```

#### **Create a Virtual Environment**

These instructions should be followed inside the **`VPC25/`** folder exactly as written. Do not modify the command examples, including the virtual environment name.

##### **Linux/macOS**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

##### **Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
```

This ensures all dependencies are installed inside an isolated environment.

#### **Activate the Virtual Environment Each Time You Work on the Project**

Each time you start working on the project, you should activate the virtual environment:

##### **Linux/macOS**
```sh
source .venv/bin/activate
```

##### **Windows (PowerShell)**
```powershell
.venv\Scripts\Activate
```

For more details on virtual environments in Python, refer to:
- [Python Virtual Environments Documentation](https://docs.python.org/3/tutorial/venv.html)
- [Real Python Guide to Virtual Environments](https://realpython.com/python-virtual-environments-a-primer/)

---

### **4️⃣ Implement Your Model**

- Modify **`model.py`** to implement your anonymization approach.
- Store any necessary model parameters in the **`parameters/`** directory.
- Add any additional dependencies to **`requirements.txt`**.

⚠️ **DO NOT modify**:

- `evaluation.py`
- `run.sh`

---

### **5️⃣ Add Your Source Audio Files**

- Place your test audio files inside **`evaluation_data/`**.
- The evaluation script will process these files automatically.

---

### **6️⃣ Run the Evaluation**

To test your model, execute:

```sh
bash run.sh
```

This will:

1. Set up and activate the virtual environment (if not already done).
2. Ensure dependencies are installed.
3. Process the source audio.
4. Generate anonymized audio files.
5. Output evaluation results to **`results.csv`**.

**Important:**
- **Windows users** must use **Git Bash** to run this command, as PowerShell and Command Prompt do not support shell scripts properly.
- **Windows and macOS users** might need to run `run.sh` with **administrator privileges** to avoid permission issues with symbolic links.

---

## **📊 Evaluation Metrics**

The evaluation script will measure:

- **Equal Error Rate (EER):** This metric, derived from an Automatic Speaker Verification (ASV) system, measures the system's ability to differentiate between speech from the same speaker and different speakers. A higher EER indicates better privacy protection, as it means the system is less likely to correctly identify the speaker.
- **Word Error Rate (WER):** This metric is calculated using an Automatic Speech Recognition (ASR) system and measures how well the anonymized speech preserves linguistic content. A lower WER indicates better utility, meaning the anonymized speech is still easily understood by the ASR system.
- **Processing time:** Measure the effeciency of the anonymization algorithm.

Results are stored in **`results.csv`**.

---

## **📜 Rules & Guidelines**

✅ **You MUST:**

- Implement your model in `model.py`.
- List dependencies in `requirements.txt`.
- Store model parameters in `parameters/`.
- Run evaluation using `run.sh`.

❌ **You MUST NOT:**

- Delete or modify `evaluation.py` or `run.sh`.
- Remove or alter existing directories.

Good luck! 🚀🎧
