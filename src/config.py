import os
import torch
import numpy as np

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "handwriting_synthesis_model.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Generation parameters
DEFAULT_BIAS = 10.0
MAX_GEN_LENGTH = 2000

# Normalization constants (calculated from training split of dataset)
TRAIN_MEAN = np.array([0.22312906, -0.0032649], dtype=np.float32)
TRAIN_STD = np.array([1.5291843, 1.3642025], dtype=np.float32)

# Predefined vocabulary characters (all unique characters present in sentences.txt, plus padding space)
VOCAB_CHARS = " !\"#'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
