# 🤖 TangkapTulis

A comprehensive deep learning project for handwriting recognition using TensorFlow and Keras, implemented in Jupyter notebooks with support for both local development and Google Colab.

---

## 📁 Project Structure

```
handwriting-recognition/
├── 📓 fp_beginner.ipynb       # Main implementation notebook
├── 🤖 model/                  # Pre-trained models directory
├── 🐍 environment.yml         # Conda environment configuration
├── 📊 data/                   # Dataset directory (after download)
├── 📒 kaggle.json             # Kaggle key
└── 📖 README.md               # Project documentation
```

---

## 🗃️ Dataset

This project uses the **Handwriting Recognition Dataset** from Kaggle:

> **📊 Dataset Info:**
> - **Source:** https://www.kaggle.com/datasets/landlord/handwriting-recognition
> - **Type:** Handwritten characters and digits
> - **Purpose:** Training deep learning recognition models

### 📥 Downloading the Dataset

#### 🖥️ Option 1: Manual Download
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/landlord/handwriting-recognition)
2. Download and extract to your project directory
3. Update data paths in the notebook

#### ☁️ Option 2: Google Colab (Recommended)
For seamless integration with Google Colab:

```python
# Upload Kaggle credentials
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json file

# Setup Kaggle API
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and extract dataset
!kaggle datasets download -d landlord/handwriting-recognition
!unzip handwriting-recognition.zip -d data/
```

> **📋 Prerequisites:** Kaggle account with API token (`kaggle.json`)

---

## 🚀 Environment Setup

### 🐍 Using Conda (Local Development)

1. **Create environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate environment:**
   ```bash
   conda activate handwriting-recognition
   ```

### ☁️ Using Google Colab
No setup required! Just upload the notebook and run the dataset download code above.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| 🐍 **Python** | 3.10 | Core language |
| 🧠 **TensorFlow** | Latest + CUDA | Deep learning framework |
| 👁️ **OpenCV** | Latest | Computer vision |
| 🔢 **NumPy** | Latest | Numerical computing |
| 📊 **Pandas** | Latest | Data manipulation |
| 📈 **Matplotlib** | Latest | Visualization |
| 🎯 **Keras** | Latest | High-level neural networks |

---

## 🎯 Usage

### 🔥 Quick Start

1. **Open the main notebook:**
   ```bash
   jupyter notebook fp_beginner.ipynb
   ```
   *Or use Jupyter Lab:*
   ```bash
   jupyter lab
   ```

2. **Follow the notebook sections:**
   - 📊 **Data Loading & Preprocessing**
   - 🏗️ **Model Architecture Design**
   - 🎯 **Training Process**
   - 📈 **Performance Evaluation**
   - 🔮 **Prediction on New Samples**

### 🤖 Working with Pre-trained Models

Load existing models for immediate use:

```python
import tensorflow as tf

# Load Keras format (recommended)
model = tf.keras.models.load_model('model/model50v2.keras')

# Load HDF5 format (legacy)
model = tf.keras.models.load_model('model/model50v2.h5')

# Alternative model
alt_model = tf.keras.models.load_model('model/model50v1.h5')
```

---

## 🚀 Getting Started

1. **📥 Clone/Download** this repository
2. **🔧 Setup** environment (Conda or Colab)
3. **📊 Download** dataset using preferred method
4. **🚀 Launch** `fp_beginner.ipynb`
5. **🎯 Follow** notebook instructions step-by-step

---

## 📈 Model Performance

The project includes multiple pre-trained models:

- **`model26.keras`** - Main production model
- **`model26.h5`** - Same model in HDF5 format  
- **`modelaa.h5`** - Alternative architecture

> **💡 Tip:** Start with `model26.keras` for best compatibility and performance.
