# Handwriting Recognition with TensorFlow

This project contains a Jupyter notebook implementation for handwriting recognition using TensorFlow and deep learning techniques.

## Project Structure

```
handwriting-recognition/
├── fp_beginner.ipynb          # Main implementation notebook
├── model/                     # Pre-trained models directory
│   ├── model26.h5            # Main model (HDF5 format)
│   ├── model26.keras         # Main model (Keras format)
│   └── modelaa.h5            # Alternative model version
├── environment.yml           # Conda environment configuration
└── README.md                # Project documentation
```

## Environment Setup

This project uses a Conda environment. To set up the environment:

1. **Create the environment** using the provided environment.yml file:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment**:
   ```bash
   conda activate handwriting-recognition
   ```
## Dataset

This project uses the **Handwriting Recognition Dataset** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/landlord/handwriting-recognition
- **Description**: A comprehensive dataset containing handwritten characters and digits for training recognition models

### Downloading the Dataset

#### Option 1: Manual Download
1. Download the dataset from the Kaggle link above
2. Extract the files to your project directory
3. Update the data paths in the notebook accordingly

#### Option 2: Using Google Colab
If you're using Google Colab, you can download the dataset directly using the following code:

```python
# Di Google Colab
from google.colab import files
uploaded = files.upload()  # Upload file kaggle.json Anda

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset IAM Handwriting
!kaggle datasets download -d landlord/handwriting-recognition
!unzip handwriting-recognition.zip -d data/
```

**Prerequisites for Google Colab:**
1. Have a Kaggle account and API token (kaggle.json)
2. Upload your `kaggle.json` file when prompted
3. The dataset will be extracted to the `data/` directory

This project uses the **Handwriting Recognition Dataset** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/landlord/handwriting-recognition
- **Description**: A comprehensive dataset containing handwritten characters and digits for training recognition models

To use this dataset:
1. Download the dataset from the Kaggle link above
2. Extract the files to your project directory
3. Update the data paths in the notebook accordingly

## Dependencies

The project requires the following main dependencies:

- **Python 3.10**
- **TensorFlow** with CUDA support
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Plotting and visualization
- **Keras** - High-level neural networks API

## Usage

The main implementation is in `fp_beginner.ipynb`. Open this notebook using Jupyter to:

1. **Train the handwriting recognition model**
2. **Evaluate model performance**
3. **Make predictions on new handwriting samples**

### Running the Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Then navigate to and open `fp_beginner.ipynb`.

## Models

Pre-trained models are available in the `model/` directory:

- **`model26.h5`** / **`model26.keras`**: Main model in different formats
- **`modelaa.h5`**: Alternative model version

### Loading Models

```python
# Load Keras format model
model = tf.keras.models.load_model('model/model26.keras')

# Load HDF5 format model
model = tf.keras.models.load_model('model/model26.h5')
```

## Getting Started

1. Clone or download this repository
2. Set up the Conda environment as described above
3. Launch Jupyter and open `fp_beginner.ipynb`
4. Follow the notebook instructions to train or use the pre-trained models

## Features

- **Deep Learning Architecture**: Utilizes TensorFlow/Keras for neural network implementation
- **Image Processing**: OpenCV integration for preprocessing handwriting images
- **Multiple Model Formats**: Support for both HDF5 and Keras native formats
- **Interactive Notebook**: Step-by-step implementation with explanations

## Requirements

- CUDA-compatible GPU (recommended for training)
- Minimum 8GB RAM
- Python 3.10 or compatible version