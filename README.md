# 😄 Face Emotion Detection with OpenCV & CNN

This is a real-time facial emotion detection system built using OpenCV and TensorFlow. It detects faces from your webcam and predicts emotions like Happy, Sad, Angry, etc.

---

## 🔍 Features

- Real-time face emotion detection
- Trained CNN model using custom dataset
- Uses OpenCV for face detection (Haar Cascades)
- Fully offline – works with webcam feed
- Easy to use and extend

---

## 🧠 Emotions Supported

- Angry  
- Happy  
- Sad  
- Surprise  
- Neutral  
- Fear  
- Disgust

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vanshbadjate07/Face-Emotion-Detection.git
cd Face-Emotion-Detection
```

### 2. Create a Virtual Environment (Optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Info

- The dataset used is organized into `train/` and `test/` folders
- If dataset is not included, download it from [Kaggle FER-2013] or similar source and extract into `archive/` folder.

---

## 🚀 How to Use

### 1. Preprocess the Data (Optional – already saved)

```bash
python preprocess.py
```

### 2. Train the Model (Optional – already trained model provided)

```bash
python train.py
```

### 3. Run Real-Time Emotion Detection

```bash
python detect.py
```

---

## 🧠 Model

- Model is saved as `model/best_model.h5`
- CNN architecture with Conv2D, MaxPooling, Dense, Dropout
- Input: 48x48 grayscale images

---

## 📄 License

MIT License

---

## 👤 Author

**Vansh Badjate**  
[GitHub](https://github.com/vanshbadjate07/)  
[LinkedIn](https://www.linkedin.com/in/vansh-badjate1008/)
