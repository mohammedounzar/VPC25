# **Voice Privacy Challenge**

Welcome to the **Voice Privacy Challenge**! Your task is to develop a model that anonymizes audio while preserving intelligibility and naturalness. This repository provides the necessary setup, evaluation script, and rules for participation.

---

## **📂 Repository Structure**

```
anonymized_audio/      # Output directory where anonymized audio files will be saved
source_audio/          # Directory for original audio files (participants can add their own, provided they are in .wav format and include at least three files)
parameters/            # Directory to store model parameters (participants should add their own)
evaluation.py          # DO NOT MODIFY - Evaluates your model and generates results.csv
model.py               # MODIFY - Implement your anonymization model here
README.md              # This file - contains all competition instructions
requirements.txt       # MODIFY - List your dependencies here
run.sh                 # DO NOT MODIFY - Runs the evaluation script
```

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
git clone https://github.com/<YOUR_GITHUB_USERNAME>/voice-privacy-challenge.git
cd voice-privacy-challenge
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

#### **Create a Virtual Environment**

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

- Place your test audio files inside **`source_audio/`**.
- The evaluation script will process these files automatically.

---

### **6️⃣ Run the Evaluation**

To test your model, execute:

```sh
sh run.sh
```

This will:

1. Set up and activate the virtual environment (if not already done).
2. Ensure dependencies are installed.
3. Process the source audio.
4. Generate anonymized audio in **`anonymized_audio/`**.
5. Output evaluation results to **`results.csv`**.

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

---

## **📬 Need Help?**

If you have questions, feel free to contact <Contact>.

Good luck! 🚀🎧
