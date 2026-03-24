import os
import sys
import torch
from monai.transforms import (
    AsDiscrete,
)
import multiprocessing

DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else min(4, multiprocessing.cpu_count())

from DosePrediction.utils.runtime import resolve_data_root, resolve_output_dir, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
Device = "cuda" if torch.cuda.is_available() else "cpu"
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
TRAIN_SIZE = int(os.environ.get("DOSE_PREDICTION_TRAIN_SIZE", 200))
VAL_SIZE = int(os.environ.get("DOSE_PREDICTION_VAL_SIZE", 100))
LEARNING_RATE = float(os.environ.get("DOSE_PREDICTION_LEARNING_RATE", 2e-4))
BATCH_SIZE = int(os.environ.get("DOSE_PREDICTION_BATCH_SIZE", 1))
SW_BATCH_SIZE = int(os.environ.get("DOSE_PREDICTION_SW_BATCH_SIZE", 1))
NUM_WORKERS = int(os.environ.get("DOSE_PREDICTION_NUM_WORKERS", DEFAULT_NUM_WORKERS))
CACHE_RATE = float(os.environ.get("DOSE_PREDICTION_CACHE_RATE", 1.0))
LAMBDA_VOXEL = 100
IMAGE_SIZE = int(os.environ.get("DOSE_PREDICTION_IMAGE_SIZE", 128))
CHANNEL_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = int(os.environ.get("DOSE_PREDICTION_NUM_EPOCHS", 200))
LOAD_MODEL = False
SAVE_MODEL = True
PRETRAIN = False
TRAIN_VAL_DIR = os.path.normpath('provided-data/nifti-val-pats/pt_*')
CHECKPOINT_MODEL_DIR = resolve_output_dir("DosePrediction", "ablation1_dose")
CHECKPOINT_MODEL_DIR_DOSE_SHARED = resolve_output_dir("DosePrediction", "dose_shared")
CHECKPOINT_MODEL_DIR_DOSE_GAN = resolve_output_dir("DosePrediction", "dose_gan")
CHECKPOINT_MODEL_DIR_BASE = resolve_output_dir("DosePrediction", "base_dose_shared")
CHECKPOINT_MODEL_DIR_BASE_FINAL = resolve_output_dir("DosePrediction", "final_baseline")
CHECKPOINT_MODEL_DIR_DOSE_SHARED_SIMPLE = resolve_output_dir("DosePrediction", "simple_dose_shared")
CHECKPOINT_MODEL_DIR_FINAL = resolve_output_dir("DosePrediction", "final")
CHECKPOINT_MODEL_DIR_FINAL_32 = resolve_output_dir("DosePrediction", "final32")
CHECKPOINT_MODEL_DIR_FINAL_FTUNE = resolve_output_dir("DosePrediction", "final_refine")
CHECKPOINT_MODEL_DIR_FINAL_RAY = resolve_output_dir("DosePrediction", "final_ray")
CHECKPOINT_MODEL_DIR_FINAL_KFOLD = resolve_output_dir("DosePrediction", "final_kfold")
CHECKPOINT_MODEL_DIR_FINAL_LINKED = resolve_output_dir("DosePrediction", "linked")

CHECKPOINT_RESULT_DIR = resolve_output_dir("DosePrediction", "results")

TRAIN_DIR = 'provided-data/nifti-train-pats/pt_*'
VAL_DIR = 'provided-data/nifti-test-pats/pt_*'
MAIN_PATH = resolve_data_root()

OAR_NAMES = [
    'Brainstem',
    'SpinalCord',
    'RightParotid',
    'LeftParotid',
    'Esophagus',
    'Larynx',
    'Mandible'
]
PTV_NAMES = ['PTV70',
             'PTV63',
             'PTV56']

post_label = AsDiscrete(to_onehot=len(OAR_NAMES) + 1)
post_pred = AsDiscrete(argmax=True, to_onehot=len(OAR_NAMES) + 1)
