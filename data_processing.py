import json
import pickle
from math import ceil
from typing import List, Dict, Union, Any

import numpy as np
from simple_settings import settings
from tqdm import tqdm

from utilities.data_processing_utils import clean_chinese_sequence, clean_english_sequence, append_start_end_tokens
from utilities.path_utils import create_directory


def load_raw_dataset(filepath: str, max_num_caption_pairs: int) -> List[Dict[str,str]]:
    """
    Load a training, validation, or testing partition of the VATEX dataset from a JSON file.

    :param filepath: location of the JSON file on disk
    :param max_num_caption_pairs: limit for how many caption pairs should be loaded
    :return: dataset loaded from filepath, with its total size limited to a multiple of 5 closest to
     sequence_limit sentences
    """

    print(f'Loading raw dataset from {filepath}')
    with open(filepath, mode='rt', encoding='utf-8') as f:
        dataset = json.load(f)
    

    # Since there are 5 parallel captions per video, this initially may store a few more captions than needed if
    # max_num_sequences is not a multiple of 5 — those are removed subsequently
    entry_limit = ceil(max_num_caption_pairs / settings.PARALLEL_CAPTION_PAIRS_PER_VIDEO)
    dataset = dataset[:entry_limit]
    
    

    print(f'Loaded {len(dataset) * settings.TOTAL_CAPTIONS_PER_VIDEO} captions '
          f'from {len(dataset)} distinct videos')

    return dataset[:entry_limit]


def extract_parallel_captions(dataset: List[Dict]) -> List[Dict]:
    """
    Extract only the parallel English-Chinese caption pairs from the loaded dataset, discarding the rest.

    :param dataset: list with entries of the form:
        {'chCap': ['一个男人正准备爬上被冰雪覆盖着的那一侧山坡。', ...],
         'enCap': ['A man is getting ready to climb the side of a mountain covered in '
                   'snow and ice.', ...]
         'videoID': '6_4kjPiQr7w_000191_000201'},
    :return: dataset containing only parallel English-Chinese captions for each video
    """
    
    print('Extracting parallel translations')
    count_before = count_sequence_pairs(dataset)
   
    for video_caption_dict in dataset:
        # Out of the 10 video caption pairs, only the last 5 pairs are parallel translations.
        # Keep only the parallel caption pairs.
        video_caption_dict.update({'chCap': video_caption_dict['chCap'][5:],
                                   'enCap': video_caption_dict['enCap'][5:]})

    count_after = count_sequence_pairs(dataset)
    print(f'Extracted {count_after} parallel caption pairs from {count_before} pairs in the original dataset')

    return dataset


def reduce_dataset_size(dataset: Dict[int, Dict], max_num_sequence_pairs: int) -> Dict[int, Dict]:
    """
    Reduce dataset size to at most sequence_limit parallel sentences.

    :param dataset: dictionary with items of the form:
        8: {'ch_seq': '<SOS> 一 个 穿着 黑色 衣服 的 男人 正 在 逗 他 的 狗 <EOS>',
           'en_seq': '<SOS> a kneeling man is training his german shepard <EOS>',
           'video_id': '7WOty3qVdts_000000_000010'}
    :param max_num_sequence_pairs: limit for how many sentence pairs can be kept
    :return: dataset containing at most sequence_limit parallel sentence pairs
    """

    dataset_size = len(dataset)
    if dataset_size > max_num_sequence_pairs:
        num_entries_remove = dataset_size - max_num_sequence_pairs
        entries_remove = list(range(dataset_size - num_entries_remove, dataset_size))
        for key in entries_remove:
            dataset.pop(key)

    return dataset


def clean_sequences(dataset: List[Dict]) -> List[Dict]:
    """
    Clean every English and Chinese sequence in the corpus: tokenise, remove punctuation marks, etc.

    :param dataset: dataset containing unmodified English and Chinese sequences as found in the original JSON
    :return: dataset containing pre-processed English and Chinese sequences
    """

    print('Processing source and target sequences (tokenising, removing punctuation marks, etc.):')
    for dct in tqdm(dataset):
        en_cap_list = []
        for index, sequence in enumerate(dct['enCap']):
            clean_sequence = clean_english_sequence(sequence)
            clean_sequence = append_start_end_tokens(clean_sequence)
            # print(1, sequence)
            # print(2, clean_sequence)
            en_cap_list.append(clean_sequence)
        dct['enCap'] = en_cap_list

        ch_cap_list = []
        for index, sequence in enumerate(dct['chCap']):
            clean_sequence = clean_chinese_sequence(sequence)
            clean_sequence = append_start_end_tokens(clean_sequence)
            # print(1, sequence)
            # print(2, clean_sequence)
            ch_cap_list.append(clean_sequence)
        dct['chCap'] = ch_cap_list

    return dataset


def filter_sequences_by_length(dataset: List[Dict], max_chinese_seq_length: int,
                               max_english_seq_length: int) -> List[Dict]:
    """
    Filter English and Chinese sequences by length, removing sequence pairs that have more than
     the specified number of tokens for sequence of either language.

    :param dataset: dataset containing English and Chinese sequences of unlimited length
    :param max_chinese_seq_length: maximum length for a Chinese sequence
    :param max_english_seq_length: maximum length for an English sequence
    :return: dataset containing English and Chinese sequences not longer than the specified length
    """

    print('Filtering sequences by sentence length')
    print(f'Max Chinese sequence length allowed: {max_chinese_seq_length}')
    print(f'Max English sequence length allowed: {max_english_seq_length}')
    num_sequences_before = count_sequence_pairs(dataset)
    print('Number of sequences before filtering out long sequences', num_sequences_before)

    # Add 2 to maximum allowed sequence length account to account for start and end tokens appended earlier
    max_chinese_seq_length += 2
    max_english_seq_length += 2

    for d in dataset:
        indices_to_remove = []
        for index, (en_seq, ch_seq) in enumerate(zip(d['enCap'], d['chCap'])):
            # Sentence lengths in words
            en_sequence_length = len(en_seq.split())
            ch_sequence_length = len(ch_seq.split())

            if (ch_sequence_length > max_chinese_seq_length) or (
                    en_sequence_length > max_english_seq_length):
                # print((en_seq, ch_seq))
                indices_to_remove.append(index)

        # Reverse list to be able to remove by index while iterating
        for index in sorted(indices_to_remove, key=int, reverse=True):
            # Remove a parallel translation if either English or Chinese sentence is too long
            # If English sentence is too long, the corresponding Chinese sentence is also removed, and vice-versa
            del d['enCap'][index]
            del d['chCap'][index]

    num_sequences_after = count_sequence_pairs(dataset)
    print(f'Filtering completed: {num_sequences_before - num_sequences_after} sequence(s) removed')
    print(f'Number of sequences after filtering out long sentences: {num_sequences_after}')

    return dataset


def convert_sequence_lists_to_arrays(dataset: List[Dict]) -> None:
    """
    For each video, convert its respective Python lists of English and Chinese captions to NumPy arrays,
    for faster processing later on.

    :param dataset: dataset containing English and Chinese sequences stored in Python lists
    :return: dataset containing English and Chinese sequences stored in NumPy arrays
    """

    for video_caption_dict in dataset:
        video_caption_dict['enCap'] = np.array(video_caption_dict['enCap'])
        video_caption_dict['chCap'] = np.array(video_caption_dict['chCap'])



def count_sequence_pairs(dataset: List[Dict]) -> int:
    """
    Count the number of English-Chinese sequence pairs in the dataset.

    :param dataset: dataset
    :return: number of sequence pairs
    """
    num_english_sequences = 0

    for video_caption_dict in dataset:
        # There is always an equal number of English and Chinese sequences, so counting the number of sequences
        # for one language is sufficient
        num_english_sequences += len(video_caption_dict.get('enCap'))

    return num_english_sequences


def transform_dataset(dataset: List[Dict]) -> Dict[int, Dict[str, Any]]:
    """
    Create a dictionary where the keys are sequential caption-pair IDs, and the values are inner dictionaries with the
    keys 'video_id', 'ch_seq', and 'en_seq'.

    :param dataset: dataset in the form of list of dictionaries
    :return: dataset in a dictionary where each video is assigned a unique integer ID
    """

    outer_dict = {}
    overall_index = 0
    for video_caption_dict in dataset:
        for index, [ch_seq, en_seq] in enumerate(zip(video_caption_dict['chCap'], video_caption_dict['enCap'])):
            inner_dict = {'video_id': video_caption_dict['videoID'],
                          'ch_seq': ch_seq,
                          'en_seq': en_seq}

            outer_dict[overall_index] = inner_dict
            overall_index += 1

    return outer_dict


def save_transformed_dataset(dataset, filepath) -> None:
    """
    Store pre-processed dataset in the pickle serialisation format.

    :param dataset: pre-processed dataset
    :param filepath: filepath where the dataset should be saved
    :return: None
    """

    # Create required directory if it doesn't already exist
    create_directory(filepath)
    with open(filepath, 'wb') as fp:
        pickle.dump(dataset, fp)
    print(f'Saved pre-processed dataset: {filepath}')


def preprocess_dataset(raw_dataset_path: str, clean_dataset_path: str, sequence_limit: int) -> None:
    """
    Complete all the steps of pre-processing for a single dataset (training, validation, or testing).

    :param raw_dataset_path: filepath where raw dataset is stored
    :param clean_dataset_path: filepath where pre-processed dataset should be saved
    :param sequence_limit: limit for how many sentence pairs should be loaded at most (others are discarded)
    :return: None
    """
    raw_dataset = load_raw_dataset(raw_dataset_path, sequence_limit)
    dataset = extract_parallel_captions(raw_dataset)
    dataset = clean_sequences(dataset)
    filter_sequences_by_length(dataset, settings.MAX_SOURCE_SEQUENCE_LENGTH, settings.MAX_TARGET_SEQUENCE_LENGTH)
    convert_sequence_lists_to_arrays(dataset)
    
    transformed_dataset = transform_dataset(dataset)
    transformed_dataset = reduce_dataset_size(transformed_dataset, sequence_limit)

    save_transformed_dataset(transformed_dataset, clean_dataset_path)



def construct_parallel_corpus(datasets_info: List[Dict[str, Any]], save_path: str) -> np.ndarray:
    """
    Construct a parallel corpus, bringing together sequences  from training and validation datasets.
    This step is needed in order to initialise tokenisers later, as one dataset may contain tokens
    not present in the other dataset.

    :param datasets_info: list of dictionaries containing the filepath of each dataset and a limit for the number of
    sequences
    :param save_path: filepath where the new parallel corpus should be saved :return: parallel corpus as a
    NumPy array
    """
    parallel_corpus = []

    for dataset_info in datasets_info:
        print(dataset_info)
        with open(dataset_info['filepath'], 'rb') as fp:
            dataset = pickle.load(fp)
            # In case loaded dataset is bigger than the specified sequence limit for the current run, resize
            dataset = reduce_dataset_size(dataset, dataset_info['limit'])
            pairs = []
            for video_caption_dict in dataset.values():
                pair = (video_caption_dict['ch_seq'], video_caption_dict['en_seq'])
                pairs.append(pair)
            print(f'Loaded {len(pairs)} parallel sentences from {dataset_info["filepath"]}')
            parallel_corpus.extend(pairs)

    parallel_corpus_array = np.array(parallel_corpus)

    with open(save_path, 'wb') as fp:
         pickle.dump(parallel_corpus_array, fp)

    print(f'Got a total of {parallel_corpus_array.shape[0]} parallel sentence pairs')
    print(f'Combined from:{[dataset["filepath"] for dataset in datasets_info]}')
    print(f'Saved to: {save_path}')

    return parallel_corpus_array


def preprocess_all_datasets() -> None:
    """
    Prepare all the data required for model training (training, validation, and test datasets).

    :return: None
    """

    # Prepare training dataset
    preprocess_dataset(settings.TRAINING_DATASET_PATH, settings.PROCESSED_TRAINING_DATA_PATH,
                       settings.TRAINING_SEQUENCE_PAIRS_LIMIT)

    # Prepare validation dataset
    preprocess_dataset(settings.VALIDATION_DATASET_PATH, settings.PROCESSED_VALIDATION_DATA_PATH,
                       settings.VALIDATION_SEQUENCE_PAIRS_LIMIT)

    # Prepare test dataset
    preprocess_dataset(settings.TEST_DATASET_PATH, settings.PROCESSED_TEST_DATA_PATH,settings.TEST_SEQUENCE_PAIRS_LIMIT)
