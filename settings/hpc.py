import multiprocessing

ENVIRONMENT = 'HPC'
USE_VIDEO_EMBEDDINGS = False

# --------------------------- Paths ---------------------------
TRAINING_DATASET_PATH = '/nobackup/sc16ps/vatex/dataset/vatex_training.json'
VALIDATION_DATASET_PATH = '/nobackup/sc16ps/vatex/dataset/vatex_validation.json'
TEST_DATASET_PATH = '/nobackup/sc16ps/vatex/dataset/vatex_test.json'

PROCESSED_TRAINING_DATA_PATH = '/nobackup/sc16ps/datasets/training_dataset.pkl'
PROCESSED_VALIDATION_DATA_PATH = '/nobackup/sc16ps/datasets/validation_dataset.pkl'
PROCESSED_TEST_DATA_PATH = '/nobackup/sc16ps/datasets/test_dataset.pkl'
PARALLEL_CORPUS_PATH = '/nobackup/sc16ps/datasets/parallel_corpus.pkl'

TRAINING_EMBEDDINGS_DIR = '/nobackup/sc16ps/vatex/embeddings/training'
VALIDATION_EMBEDDINGS_DIR = '/nobackup/sc16ps/vatex/embeddings/validation'
TEST_VIDEO_EMBEDDINGS_DIR = '/nobackup/sc16ps/vatex/embeddings/test'

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

