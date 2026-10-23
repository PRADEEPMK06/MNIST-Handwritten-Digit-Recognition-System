# CodeAlpha_HandwrittenCharacterRecognition

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2EA44F?style=for-the-badge)

**CodeAlpha Machine Learning Internship - Task 3: Handwritten Character Recognition**

A CNN-based system that recognizes handwritten **digits (0-9)** and **letters (A-Z)** from images, trained on the combined **MNIST** + **EMNIST (byclass, uppercase-only)** datasets.

## Objective

Identify handwritten characters or alphabets using image processing and deep learning, extendable to full word/sentence recognition.

## Approach

- **Datasets:** MNIST (digits) + EMNIST `byclass` split filtered to uppercase-only (A-Z), merged into one 36-class problem.
- **Model:** A Convolutional Neural Network with two conv blocks (Conv -> BatchNorm -> Conv -> BatchNorm -> MaxPool -> Dropout), followed by a dense classifier head.
- **Training:** Data augmentation (rotation, shift, zoom) on a stratified train/validation split, with early stopping, learning-rate reduction on plateau, and class-weight balancing (EMNIST `byclass` isn't sample-balanced per letter).
- **Evaluation:** Accuracy, per-class precision/recall/F1, and a confusion matrix.

## Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- models/                  # trained model + plots saved here after training
|-- data/                    # persisted app prediction history (prediction_history.json)
`-- src/
    |-- labels.py            # shared label map (0-9 -> digits, 10-35 -> A-Z)
    |-- data_loader.py       # loads & merges MNIST + EMNIST
    |-- model.py             # CNN architecture
    |-- train.py             # training script
    |-- evaluate.py          # test-set evaluation + confusion matrix
    |-- predict.py           # predict a single character image (CLI)
    `-- app.py               # Streamlit web app (draw + upload UI)
```

## Setup

```bash
git clone https://github.com/<your-username>/CodeAlpha_HandwrittenCharacterRecognition.git
cd CodeAlpha_HandwrittenCharacterRecognition

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note on datasets:** `data_loader.py` downloads MNIST automatically via `tensorflow.keras.datasets`, and EMNIST (`byclass` split) via `tensorflow-datasets`, on first run. This needs an active internet connection - works out-of-the-box on your laptop or in Google Colab. `byclass` is the full, unbalanced EMNIST split (~500MB+) and gets cached locally afterward, so only the first run pays the download cost.

## Usage

**1. Train the model**
```bash
cd src
python train.py --epochs 20 --batch-size 128
```
The best model (by validation accuracy) is saved to `models/handwritten_char_cnn.keras`, and training curves are saved to `models/training_history.png`.

**2. Evaluate on the test set**
```bash
python evaluate.py
```
Prints a classification report and saves a confusion matrix to `models/confusion_matrix.png`.

**3. Predict a single character (command line)**
```bash
python predict.py --image ../path/to/your_character.png
```
Works with either dark-text-on-light or light-text-on-dark images - it auto-detects and inverts if needed.

**4. Web app (recommended for demos)**
```bash
streamlit run app.py
```
Opens a browser UI where you draw a character (or upload an image) and instantly see:
- The predicted character with a confidence circle
- A bar breakdown of the top 3 predictions
- A persistent "homework pile" history of every character graded, all-time
- A report-card chart of average confidence per character

This is the easiest way to show the project in your video walkthrough.

## Interactive demo (Streamlit)

A browser-based demo where you draw a character (or upload an image) and get a live prediction:

```bash
streamlit run app.py
```

This opens at `http://localhost:8501`. You need a trained model first (`python src/train.py`) - the app will tell you clearly if it can't find one yet at `models/handwritten_char_cnn.keras`.

Design-wise, it's built around the actual data convention rather than a generic template: MNIST/EMNIST images are literally white strokes on a black background, so the drawing board itself keeps that convention, while the surrounding page uses a light, easy-to-read layout - the top prediction gets circled like a teacher grading homework.

## Expected Results

| Dataset                          | Typical Test Accuracy |
|-----------------------------------|:---------------------:|
| MNIST (digits only)                | ~99%                  |
| EMNIST `byclass`, uppercase only   | ~90%+                 |
| Combined (this project)            | ~93-96%                |

*(Exact numbers depend on epochs, augmentation settings, and hardware - retrain with `python src/train.py` and check `models/training_history.png` / `evaluate.py`'s report for your actual run.)*

## Future Extensions

- Extend to full word/sentence recognition using a **CRNN** (CNN + RNN + CTC loss) for sequence modeling.
- Fine-tune on a custom handwriting sample for personalized recognition.
- Add lowercase letter support as a separate, explicitly-labeled class set (rather than merging with uppercase).

## CodeAlpha Internship

This project was completed as part of the **CodeAlpha Machine Learning Internship** (Task 3 of 4).

- Internship info: [www.codealpha.tech](https://www.codealpha.tech)
- Task: Handwritten Character Recognition
