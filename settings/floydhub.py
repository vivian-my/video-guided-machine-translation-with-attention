import multiprocessing

ENVIRONMENT = 'FLOYDHUB'
USE_VIDEO_EMBEDDINGS = True

# --------------------------- Paths ---------------------------
TRAINING_DATASET_PATH = '/vatex_json/vatex_training.json'
VALIDATION_DATASET_PATH = '/vatex_json/vatex_validation.json'
TEST_DATASET_PATH = '/vatex_json/vatex_test.json'

PROCESSED_TRAINING_DATA_PATH = '/vatex_proc/training_dataset.pkl'
PROCESSED_VALIDATION_DATA_PATH = '/vatex_proc/validation_dataset.pkl'
PROCESSED_TEST_DATA_PATH = '/vatex_proc/test_dataset.pkl'
PARALLEL_CORPUS_PATH = '/vatex_proc/parallel_corpus.pkl'

TRAINING_EMBEDDINGS_DIR = '/vatex_embeddings/'
VALIDATION_EMBEDDINGS_DIR = '/vatex_embeddings/'
TEST_VIDEO_EMBEDDINGS_DIR = '/vatex_embeddings/'

# --------------------------- Data-related settings ---------------------------
MAX_NUM_SOURCE_TOKENIZER_WORDS = 30000  # 29534
MAX_NUM_TARGET_TOKENIZER_WORDS = 23000  # 22505
MAX_VIDEO_EMBEDDING_LENGTH = 30000

if USE_VIDEO_EMBEDDINGS:
    FINAL_NUM_ENCODER_TOKENS = max(MAX_VIDEO_EMBEDDING_LENGTH, MAX_NUM_SOURCE_TOKENIZER_WORDS)
else:
    FINAL_NUM_ENCODER_TOKENS = MAX_NUM_SOURCE_TOKENIZER_WORDS

TRAINING_SEQUENCE_PAIRS_LIMIT = 116955
VALIDATION_SEQUENCE_PAIRS_LIMIT = 15000
TEST_SEQUENCE_PAIRS_LIMIT = 13000

# --------------------------- Hyperparameter settings ---------------------------
BATCH_SIZE = 64
EPOCHS = 15
LATENT_DIM = 256
LEARNING_RATE = 0.001

USE_MULTIPROCESSING = True
WORKERS = multiprocessing.cpu_count()
