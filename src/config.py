"""Default settings for the sleep staging workflow."""
import os
import torch


class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'train')
    TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed', 'test')
    RAW_TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train')
    RAW_TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'test')
    INPUT_MODE = 'preprocessed'
    RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')
    CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

    SAMPLING_RATE = 100
    EPOCH_DURATION = 30
    EPOCH_LENGTH = SAMPLING_RATE * EPOCH_DURATION
    LOW_FREQ = 0.5
    HIGH_FREQ = 30.0
    NOTCH_FREQ = None
    FILTER_ORDER = 4

    LABEL_MAP = {
        'W': 0, 'Sleep stage W': 0, 'Wake': 0,
        '1': 1, 'Sleep stage 1': 1, 'N1': 1,
        '2': 2, 'Sleep stage 2': 2, 'N2': 2,
        '3': 3, 'Sleep stage 3': 3, '4': 3, 'Sleep stage 4': 3, 'N3': 3,
        'R': 4, 'Sleep stage R': 4, 'REM': 4,
    }
    REVERSE_LABEL_MAP = {
        0: 'Sleep stage W', 1: 'Sleep stage 1', 2: 'Sleep stage 2',
        3: 'Sleep stage 3', 4: 'Sleep stage R',
    }
    NUM_CLASSES = 5
    EVAL_CLASSES = [1, 2, 3, 4]

    SEQ_LEN = 20  # Twenty contiguous epochs provide ten minutes of context.
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    VAL_RATIO = 0.15

    CNN_DROPOUT = 0.5
    ATTN_DROPOUT = 0.1
    FC_DROPOUT = 0.3
    FEATURE_DIM = 256
    LSTM_HIDDEN = 128
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 8
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.05
    USE_CLASS_WEIGHTS = True

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    USE_DATA_AUGMENT = True
    TTA_N = 1  # Deterministic inference; augmentation is an explicit experiment.


def get_config():
    return Config()
