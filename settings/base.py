import sys
import logging

from utilities.path_utils import get_project_base_dir, get_absolute_path

ENVIRONMENT = 'PC'
USE_VIDEO_EMBEDDINGS = True

# --------------------------- Paths ---------------------------
PROJECT_BASE_DIR = get_project_base_dir()

TRAINING_DATASET_PATH = get_absolute_path('data/vatex/vatex_training.json')
VALIDATION_DATASET_PATH = get_absolute_path('data/vatex/vatex_validation.json')
TEST_DATASET_PATH = get_absolute_path('data/vatex/vatex_test.json')

PROCESSED_TRAINING_DATA_PATH = get_absolute_path('data/pickle/training_dataset.pkl')
PROCESSED_VALIDATION_DATA_PATH = get_absolute_path('data/pickle/validation_dataset.pkl')
PROCESSED_TEST_DATA_PATH = get_absolute_path('data/pickle/test_dataset.pkl')
PARALLEL_CORPUS_PATH = get_absolute_path('data/pickle/parallel_corpus.pkl')

TRAINING_EMBEDDINGS_DIR = get_absolute_path('data/vatex/embeddings/training')
VALIDATION_EMBEDDINGS_DIR = get_absolute_path('data/vatex/embeddings/validation')
TEST_VIDEO_EMBEDDINGS_DIR = get_absolute_path('data/vatex/embeddings/test')

MODEL_SAVE_DIR = 'saved'

# --------------------------- Data-related settings ---------------------------
TOTAL_CAPTIONS_PER_VIDEO = 20
PARALLEL_CAPTION_PAIRS_PER_VIDEO = 5

MAX_SOURCE_SEQUENCE_LENGTH = 300
MAX_TARGET_SEQUENCE_LENGTH = 300

MAX_NUM_SOURCE_TOKENIZER_WORDS = 30000  # smallest possible value
MAX_NUM_TARGET_TOKENIZER_WORDS = 23000
MAX_VIDEO_EMBEDDING_LENGTH = 30000

if USE_VIDEO_EMBEDDINGS:
    FINAL_NUM_ENCODER_TOKENS = max(MAX_VIDEO_EMBEDDING_LENGTH, MAX_NUM_SOURCE_TOKENIZER_WORDS)
else:
    FINAL_NUM_ENCODER_TOKENS = MAX_NUM_SOURCE_TOKENIZER_WORDS

TRAINING_SEQUENCE_PAIRS_LIMIT = 116955
VALIDATION_SEQUENCE_PAIRS_LIMIT = 1500
TEST_SEQUENCE_PAIRS_LIMIT = 1300

SEQUENCE_START_TOKEN = '<SOS>'
SEQUENCE_END_TOKEN = '<EOS>'

SAMPLE_VIDEO_ID = '__NrybzYzUg_000415_000425'

I3D_SEGMENT_LENGTH = 2048

# --------------------------- Hyperparameter settings ---------------------------
BATCH_SIZE = 32
EPOCHS = 4
LATENT_DIM = 256
LEARNING_RATE = 0.001

USE_MULTIPROCESSING = False
WORKERS = 1

# ReduceLROnPlateau settings
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_PATIENCE = 1

# --------------------------- Logging settings ---------------------------
LOGGING_STREAM = sys.stdout
LOGGING_LEVEL = logging.DEBUG
