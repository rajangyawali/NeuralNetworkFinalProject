import torch
import os

DATASET_PATH ="lfw_dataset/"
ATTRIBUTES_PATH = "lfw_attributes.txt"


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda:1" else False

BASE_OUTPUT = "output"
NUM_WORKERS = 2

# Hyperparameters
BATCH_SIZE = 200
LEARNING_RATE = 0.001
NUM_EPOCHS = 200

INPUT_CHANNEL = 3
LATENT_DIM = 8100

TYPE = 'Normal-VAE'
if TYPE == 'Normal-VAE':
    BETA = 1
elif TYPE == 'Beta-VAE':
    BETA = 2
else:
    BETA = 2
    
TRAINING = True

NAME = f"{TYPE} ModelV2 with beta={BETA}, latent dims={LATENT_DIM}, epochs={NUM_EPOCHS}, LR={LEARNING_RATE}"
print("*******************************************************************************")
print("*************************** Experimenting *************************************")
print("*******************************************************************************")
print("*******************************************************************************")
print(f"\t{NAME}")
print("*******************************************************************************")
print("*******************************************************************************")