# MNIST-Handwritten-Digit-Recognition-System

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2EA44F?style=for-the-badge)

## Handwritten Digit Recognition System

A deep learning-based handwritten digit recognition system that identifies handwritten digits **(0–9)** from images using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Objective

The objective of this project is to develop a machine learning system capable of accurately recognizing handwritten digits using image processing and deep learning techniques.

## Approach

- **Dataset:** MNIST handwritten digit dataset, containing 60,000 training images and 10,000 test images.
- **Model:** A Convolutional Neural Network (CNN) designed to extract image features and classify handwritten digits into 10 classes (0–9).
- **Preprocessing:** Image normalization and reshaping to prepare images for model training and prediction.
- **Training:** Train the CNN using the training dataset and validate its performance.
- **Evaluation:** Evaluate the trained model using test accuracy, loss, and classification performance.
- **Prediction:** Use the trained model to recognize handwritten digits from input images.

## Project Structure

```text
MNIST-Handwritten-Digit-Recognition-System/
├── README.md
├── requirements.txt
├── .gitignore
├── models/
│   └── trained model files
├── data/
│   └── MNIST dataset (if stored locally)
└── src/
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    └── model.py
```

*Adjust the folder and file names to match your actual repository.*

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/PRADEEPMK06/MNIST-Handwritten-Digit-Recognition-System.git
cd MNIST-Handwritten-Digit-Recognition-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model

```bash
python src/train.py
```

### 2. Evaluate the model

```bash
python src/evaluate.py
```

### 3. Predict a handwritten digit

```bash
python src/predict.py --image path/to/your_digit.png
```

*Ensure these commands match the scripts and arguments implemented in your repository.*

## Dataset

This project uses the **MNIST handwritten digit dataset**, a standard benchmark dataset in machine learning and computer vision.

- **Training images:** 60,000
- **Testing images:** 10,000
- **Image dimensions:** 28 × 28 pixels
- **Number of classes:** 10
- **Classes:** Digits 0–9
- **Image format:** Grayscale

The dataset can be loaded using TensorFlow/Keras, depending on the implementation.

## Expected Results

A well-trained CNN on MNIST can typically achieve approximately **98–99% test accuracy**, depending on the model architecture, preprocessing, and training configuration.

| Metric | Description |
|---|---|
| Accuracy | Percentage of correctly classified digits |
| Loss | Measures prediction error |
| Confusion matrix | Shows correct and incorrect predictions for each digit |
| Classification report | Provides precision, recall, and F1-score |

*Use the actual results from your model evaluation when reporting performance.*

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- MNIST Dataset

## Future Enhancements

- Build an interactive web application using Streamlit.
- Add a drawing canvas for real-time digit prediction.
- Support image uploads for handwritten digit recognition.
- Display prediction confidence scores.
- Improve model performance with data augmentation.
- Deploy the application for online access.

## Author

**Pradeep M K**

## GitHub Repository

[MNIST-Handwritten-Digit-Recognition-System](https://github.com/PRADEEPMK06/MNIST-Handwritten-Digit-Recognition-System)
