import os
import re
import string
from typing import List, Tuple, Union, Iterator
from unicodedata import normalize

import inflect as inflect
import jieba
import numpy as np
from simple_settings import settings
from zhon import hanzi

from dataset_type import DatasetType
from .contractions_utils import expand_contractions
from .path_utils import get_video_embeddings_directory

regex_en_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
regex_ch_punctuation = re.compile('[%s]' % re.escape(hanzi.punctuation))
regex_printable = re.compile('[^%s]' % re.escape(string.printable))

inflect_engine = inflect.engine()
jieba.setLogLevel(jieba.logging.WARNING)


def replace_numbers(sequence: str) -> List[str]:
    """
    Replace all integer number occurrences in a sequence with textual representation.

    :param sequence: string
    :return: list of tokens in string with numbers replaced by corresponding textual representation
    """
    new_words = []
    for ch in sequence:
        if ch.isdigit():
            new_word = inflect_engine.number_to_words(ch)
            new_words.append(new_word)
        else:
            new_words.append(ch)
    return new_words


def clean_english_sequence(sequence: str) -> str:
    """
    Pre-process an English sequence.

    :param sequence: English sequence
    :return: clean English sequence
    """

    # Normalize unicode characters
    sequence = normalize('NFD', sequence)

    # Convert sequence to lowercase
    sequence = sequence.lower()

    # Tokenize on white space
    sequence_tokens = sequence.split()

    # Replace contractions with their expansions (e.g., aren't --> are not)
    sequence_tokens = expand_contractions(' '.join(sequence_tokens)).split()

    # Remove punctuation from each token
    sequence_tokens = [regex_en_punctuation.sub('', token) for token in sequence_tokens]

    # Remove non-printable characters form each token
    sequence_tokens = [regex_printable.sub('', token) for token in sequence_tokens]

    # Replace number tokens (<=10) with their word equivalents
    for index, word in enumerate(sequence_tokens):
        if word.isdigit() and int(word) <= 10:
            sequence_tokens[index] = ' '.join(replace_numbers(word))

    # Convert list of tokens to string
    return ' '.join(sequence_tokens)


def clean_chinese_sequence(sequence):
    """
    Pre-process a Chinese sequence.

    :param sequence: Chinese sequence
    :return: clean Chinese sequence
    """

    # Normalize Unicode characters
    sequence = normalize('NFD', sequence)

    # Segment a Chinese string into separate words
    sequence = jieba.cut(sequence)

    # Remove Chinese punctuation marks
    sequence = [regex_ch_punctuation.sub('', w) for w in sequence]

    # Remove tokens with numbers in them
    sequence = [word for word in sequence if word.isalpha()]
    return ' '.join(sequence)


def load_raw_video_embedding(dataset_type: 'DatasetType', video_id: str) -> np.ndarray:
    """
    Load a video embedding from disk.

    :param dataset_type: type of the dataset in which to look for a video
    :param video_id: id of video for which to load an embedding
    :return: loaded video embedding
    """

    embeddings_dir = get_video_embeddings_directory(dataset_type)
    embedding_path = os.path.join(embeddings_dir, f'{video_id}.npy')
    video_embedding = np.load(embedding_path)

    return video_embedding


def get_video_embedding(dataset_type: 'DatasetType', video_id: str) -> np.ndarray:
    """
    Get a video embedding in a suitable shape.

    :param dataset_type: type of the dataset in which to look for a video
    :param video_id: id of video for which to load an embedding
    :return: loaded and reshaped video embedding
    """

    video_embedding = load_raw_video_embedding(dataset_type, video_id)

    # Reshape the embedding into 1D
    video_embedding = video_embedding.ravel()

    # Ensure the embedding is not larger than allowed
    max_length_in_multiples = int(settings.MAX_VIDEO_EMBEDDING_LENGTH / settings.I3D_SEGMENT_LENGTH)

    # Slice at the boundary
    video_embedding = video_embedding[:max_length_in_multiples * settings.I3D_SEGMENT_LENGTH]

    # Pad embedding with zeros to make it of consistent size,
    # so it can form a part of the NumPy array corresponding to a batch in the data generator
    padding_length = settings.MAX_VIDEO_EMBEDDING_LENGTH - video_embedding.shape[0]
    video_embedding = np.pad(video_embedding, (0, padding_length), 'constant')

    return video_embedding


def get_input_target_texts(parallel_corpus: np.ndarray) -> Tuple[List[str], List[str]]:
    """
    Given a parallel corpus as a NumPy array, extract input and target texts.

    :param parallel_corpus: parallel corpus as a NumPy array
    :return: list of input texts and list of target texts
    """

    input_texts = parallel_corpus[:, 0]
    target_texts = parallel_corpus[:, 1]
    return input_texts, target_texts


def find_max_sequence_length(sequences: List[str]) -> int:
    """
    Given a list of sequences, return the number of tokens in the longest sequence.

    :param sequences: list of sequence strings
    :return: number of tokens in the longest sequence string
    """

    return max(len(sequence.split()) for sequence in sequences)


def append_start_end_tokens(sequence: str) -> str:
    """
    Append a start and an end token to a string.

    :param sequence: sequence as a string
    :return: sequence with start and end tokens appended to it
    """

    return f'{settings.SEQUENCE_START_TOKEN} {sequence.strip()} {settings.SEQUENCE_END_TOKEN}'


def remove_start_end_tokens(sequence: Union[str, List[str]], return_as_string: bool = True) \
        -> Union[str, Iterator[str]]:
    """
    Remove start and end tokens from a string or a list of tokens.

    :param sequence: a string or a list of tokens
    :param return_as_string: whether to return the result as a string (if not, return as a list of tokens)
    :return: sequence with start and end tokens removed
    """

    if isinstance(sequence, str):
        sequence = sequence.split()

    is_not_start_end_token = lambda x: x != settings.SEQUENCE_START_TOKEN and x != settings.SEQUENCE_END_TOKEN
    sequence = filter(is_not_start_end_token, sequence)

    if return_as_string:
        return ' '.join(sequence)
    else:
        return sequence
