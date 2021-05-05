import pickle
from typing import List, Dict, Union, Tuple

import numpy as np
from tensorflow.keras.utils import Sequence
from simple_settings import settings

from dataset_type import DatasetType
from utilities.data_processing_utils import get_video_embedding
from utilities.printing_utils import print_asterisk_separator


class DataGenerator(Sequence):
    """
    A class which generates batches of data that are used for model training and testing.
    """

    def __init__(self, batch_size: int, dimensions: Dict, dataset_type: 'DatasetType', input_token_index: Dict,
                 target_token_index: Dict, use_video_embeddings: bool, shuffle: bool = False) -> None:
        """
        Initialise a data generator.

        :param batch_size: size of the batch that will be generated
        :param dimensions: dictionary of dimensions used for creating NumPy array
        :param dataset_type: type of the dataset (TRAINING, VALIDATION, or TEST)
        :param input_token_index: input token index obtained from a Keras tokeniser
        :param target_token_index: target token index obtained from a Keras tokeniser
        :param use_video_embeddings: whether video embeddings are used
        :param shuffle: whether indexes in each batch are shuffled after epoch ends
        :return: None
        """

        self.dimensions = dimensions
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.dataset = self.load_dataset()
        self.dataset_item_ids = range(len(self.dataset))
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.use_video_embeddings = use_video_embeddings
        self.shuffle = shuffle
        self.on_epoch_end()

        if settings.ENVIRONMENT == 'PC':
            # Get sample video embedding used for testing on a desktop machine
            self.sample_video_embedding = get_video_embedding(DatasetType.TRAINING, settings.SAMPLE_VIDEO_ID)

        if self.use_video_embeddings:
            # Add one because the first input in a sequence is now always a video embedding
            self.dimensions['max_encoder_seq_length'] = self.dimensions['max_encoder_seq_length'] + 1

        self.dimensions['final_num_encoder_tokens'] = settings.FINAL_NUM_ENCODER_TOKENS

        print(f'DataGenerator.__init__() called!')
        self.print_generator_info()

    def print_generator_info(self) -> None:
        """
        Print information related to this data generator.

        :return: None
        """
        print(f'Dataset type: {self.dataset_type.name}')
        print(f'Use video embeddings: {self.use_video_embeddings}')
        print(f'Number of batches per epoch: {self.__len__()}')
        print(f'Batch size: {self.batch_size}')
        print(f'Total: {self.__len__()} * {self.batch_size} = {self.batch_size * self.__len__()}')
        print(f'Shuffle enabled? {self.shuffle}')
        print_asterisk_separator()

    def load_dataset(self) -> Dict:
        """
        Load a pickled dataset from disk, based on the dataset type of this data generator and
        the settings files.

        :return: loaded dataset as a dictionary
        """
        if self.dataset_type == DatasetType.TRAINING:
            dataset = pickle.load(open(settings.PROCESSED_TRAINING_DATA_PATH, 'rb'))
        elif self.dataset_type == DatasetType.VALIDATION:
            dataset = pickle.load(open(settings.PROCESSED_VALIDATION_DATA_PATH, 'rb'))
        elif self.dataset_type == DatasetType.TEST:
            dataset = pickle.load(open(settings.PROCESSED_TEST_DATA_PATH, 'rb'))
        else:
            raise ValueError('Invalid dataset type supplied')

        return dataset

    def __len__(self) -> int:
        """
        Get the number of batches per epoch produced by this data generator.

        :return: number of batches per epoch
        """
        return int(np.floor(len(self.dataset_item_ids) / self.batch_size))

    def __getitem__(self, batch_index: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate a batch.

        :param batch_index: index of the batch that will be returned
        :return: three NumPy arrays: [encoder input data, decoder input data], and decoder target data
        """

        # Generate indexes belonging in this batch
        item_ids_in_batch = self.get_item_ids_in_batch(batch_index)

        # Adapted from https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

        encoder_input_data = np.zeros(
            (self.batch_size, self.dimensions['max_encoder_seq_length'],
             self.dimensions['final_num_encoder_tokens']),
            dtype='float32')

        decoder_input_data = np.zeros(
            (self.batch_size, self.dimensions['max_decoder_seq_length'], self.dimensions['num_decoder_tokens']),
            dtype='float32')

        decoder_target_data = np.zeros(
            (self.batch_size, self.dimensions['max_decoder_seq_length'], self.dimensions['num_decoder_tokens']),
            dtype='float32')

        # encoder_input_data -- k x m x n NumPy array where:
        # k is the number of input sequences in this batch, i.e. batch size
        # m is the number of tokens of the longest individual sequence in the entire dataset
        # n is the size of the input vocabulary (number of different input tokens (chinese characters)
        # 	that are possible)

        for i, id in enumerate(item_ids_in_batch):
            input_text = self.dataset[id]['ch_seq']
            target_text = self.dataset[id]['en_seq']
            video_id = self.dataset[id]['video_id']

            if settings.ENVIRONMENT != 'PC':
                # Get the corresponding video embedding
                video_embedding = get_video_embedding(self.dataset_type, video_id)
            else:
                # Use a sample embedding for testing on a desktop machine
                video_embedding = self.sample_video_embedding

            input_words = input_text.split()

            if self.use_video_embeddings:
                for t, word in enumerate(input_words):  # for every word in input text
                    encoder_input_data[
                        # (t + 1) because the first token will be a video embedding — one-hot encoded words are
                        # shifted to the right by one
                        i, t + 1, self.input_token_index[
                            word]] = 1.  # the corresponding word column to 1

                encoder_input_data[i][0] = video_embedding
            else:
                for t, word in enumerate(input_words):  # for every word in input text
                    encoder_input_data[
                        i, t, self.input_token_index[
                            word]] = 1.  # the corresponding word column to 1

            target_words = target_text.split()
            for t, word in enumerate(target_words):  # for every word in target text
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self.target_token_index[word]] = 1.
                # dec_inp_data.append(word)
                if t > 0:
                    # Teacher forcing:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start token.
                    decoder_target_data[i, t - 1, self.target_token_index[word]] = 1.

        return [encoder_input_data, decoder_input_data], decoder_target_data

    def on_epoch_end(self) -> None:
        """
        Update the indexes in this data generator after each epoch.

        :return: None
        """
        self.indexes = np.arange(len(self.dataset_item_ids))

        # Shuffle the indexes, changing the order in which the model sees sequences
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_item_ids_in_batch(self, batch_index:int) -> List[Union[int, range]]:
        """
        Get IDs of items contained in the specified batch.

        :param batch_index: index of the batch for which to retrieve item IDs
        :return: IDs of items in the batch
        """
        indexes_in_batch = self.indexes[batch_index * self.batch_size:
                                        (batch_index + 1) * self.batch_size]

        # Get a corresponding list of dataset item IDs
        item_ids_in_batch = [self.dataset_item_ids[k] for k in indexes_in_batch]

        return item_ids_in_batch

    def generate_batch_for_testing(self, batch_index:int) -> Tuple:
        """
        Generate a batch for testing. In addition to the normal NumPy array, this includes the
        corresponding input and target texts, which can be used for evaluation.

        :param batch_index: index of the batch which will be retrieved for testing
        :return: a tuple containing three NumPy arrays ([encoder input data, decoder input data], decoder target data)
                 and a list of lists (input strings and target strings)

        """
        item_ids_in_batch = self.get_item_ids_in_batch(batch_index)
        input_texts = [self.dataset[id]['ch_seq'] for id in item_ids_in_batch]
        target_texts = [self.dataset[id]['en_seq'] for id in item_ids_in_batch]

        # Include input texts and target texts, which are not returned by the __getitem__ method
        return self.__getitem__(batch_index), [input_texts, target_texts]
