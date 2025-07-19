# 😄 Face Emotion Detection with OpenCV & CNN

This is a real-time facial emotion detection system built using OpenCV and TensorFlow. It detects faces from your webcam and predicts emotions like Happy, Sad, Angry, etc.

---

## 🔍 Features

- Real-time face emotion detection
- Trained CNN model using a custom dataset
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

## 📦 Dataset & Model Files

To keep the repository lightweight, large files such as the dataset, preprocessed `.npy` files, and trained model weights are **not included** in this GitHub repo.

🔗 Download all missing files from here:  
👉 [Download Dataset & Model Files (Google Drive)](https://drive.google.com/drive/folders/1HiCgbI4HCDeMgOCid2qF_jDs4om-4dD7?usp=sharing)

This includes:
- `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`
- `final_emotion_model.h5`, `best_model.h5`
- Preprocessed `archive/` dataset if needed

Place them in the root folder of the project as follows:

```
Face-Emotion-Detection/
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
├── best_model.h5
├── final_emotion_model.h5
```

---

## 🚀 How to Use

### 1. Preprocess the Data (Optional – already saved)

```bash
python preprocess.py
```

### 2. Train the Model (Optional – already trained model provided)

```bash
python train_model.py
```

### 3. Run Real-Time Emotion Detection

```bash
python predict.py
```

---

## 🧠 Model Details

- Saved as `best_model.h5`
- CNN architecture with Conv2D, MaxPooling2D, Dense, and Dropout
- Input: 48x48 grayscale face images
- Optimized for categorical emotion classification

---

## 🧪 Requirements

- Python 3.6+
- OpenCV
- TensorFlow / Keras
- NumPy

Use the included `requirements.txt` to install all dependencies.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Vansh Badjate**  
[GitHub](https://github.com/vanshbadjate07/)  
[LinkedIn](https://www.linkedin.com/in/vansh-badjate1008/)
