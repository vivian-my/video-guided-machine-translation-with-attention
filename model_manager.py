import pickle
from typing import Dict, Any, Union, List

from tensorflow import keras as keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from simple_settings import settings

from data_generator import DataGenerator
from data_processing import construct_parallel_corpus
from dataset_type import DatasetType
from model_2 import VmtModel
from utilities.data_processing_utils import get_input_target_texts, find_max_sequence_length, remove_start_end_tokens
from utilities.path_utils import generate_model_checkpoint_path, get_model_summary_path, create_directories, \
    generate_model_history_path
from utilities.printing_utils import print_asterisk_separator

DATASETS = [{'filepath': settings.PROCESSED_TRAINING_DATA_PATH,
             'limit': settings.TRAINING_SEQUENCE_PAIRS_LIMIT},
            {'filepath': settings.PROCESSED_VALIDATION_DATA_PATH,
             'limit': settings.VALIDATION_SEQUENCE_PAIRS_LIMIT},
            {'filepath': settings.PROCESSED_TEST_DATA_PATH,
             'limit': settings.TEST_SEQUENCE_PAIRS_LIMIT}]


class ModelManager:
    """
    A class that provides a way to interact with the VMT model and can be used to train and evaluate the VMT model.
    """

    def __init__(self, prepare_for_training: bool = False, prepare_for_sampling: bool = False, load_model: bool = False,
                 saved_model_path: str = None) -> None:
        """
        Initialise a model manager.

        :param prepare_for_training: whether the model manager should prepare a VMT model for training
        :param prepare_for_sampling: whether the model manager should prepare a VMT model for sampling
        :param load_model: whether a saved VMT model should be loaded
        :param saved_model_path: filepath of a saved VMT model which is to be loaded
        :return: None
        """

        self.message_count = 0

        parallel_corpus = construct_parallel_corpus(DATASETS, settings.PARALLEL_CORPUS_PATH)
        self.input_texts, self.target_texts = get_input_target_texts(parallel_corpus)
        self.echo('Loaded parallel caption corpus')

        self.english_tokenizer = self.prepare_english_tokenizer()
        self.chinese_tokenizer = self.prepare_chinese_tokenizer()
        self.input_token_index = self.chinese_tokenizer.word_index
        self.target_token_index = self.english_tokenizer.word_index
        self.tokenizer_info = self.get_tokenizer_info()
        self.print_parallel_corpus_info()
        self.dimensions = self.get_dimensions_from_tokenizers()

        self.echo('Prepared tokenisers')

        if load_model:
            self.custom_model = self.load_model_from_path(settings.LATENT_DIM, saved_model_path)
            self.echo(f'Loaded model from {saved_model_path}')

        else:
            self.custom_model = VmtModel(latent_dim=settings.LATENT_DIM,
                                         num_encoder_tokens=settings.FINAL_NUM_ENCODER_TOKENS,
                                         num_decoder_tokens=self.dimensions['num_decoder_tokens'])
            self.echo(f'Defined VMT model')

        self.entire_model = self.custom_model.get_entire_model()

        if prepare_for_training:
            self.training_generator = self.initialise_generator(DatasetType.TRAINING, settings.BATCH_SIZE,
                                                                settings.USE_VIDEO_EMBEDDINGS)
            self.validation_generator = self.initialise_generator(DatasetType.VALIDATION, settings.BATCH_SIZE,
                                                                  settings.USE_VIDEO_EMBEDDINGS)
            self.compile_model()
            self.echo('Compiled VMT model')

        if prepare_for_sampling:
            self.reverse_input_token_index = self.get_reverse_input_token_index()
            self.reverse_target_token_index = self.get_reverse_target_token_index()
            self.test_generator = self.initialise_generator(DatasetType.TEST, settings.BATCH_SIZE,
                                                            settings.USE_VIDEO_EMBEDDINGS)
            self.sampling_encoder = self.custom_model.get_sampling_encoder()
            self.sampling_decoder = self.custom_model.get_sampling_decoder()
            self.echo('Prepared for sampling')

    def prepare_english_tokenizer(self) -> Tokenizer:
        """
        Fit a Keras tokeniser on the relevant target texts.

        :return: tokeniser that has been fit on English target texts
        """

        english_tokenizer = Tokenizer(num_words=settings.MAX_NUM_TARGET_TOKENIZER_WORDS, lower=False,
                                      char_level=False, filters=' ', oov_token='unk')
        english_tokenizer.fit_on_texts(self.target_texts)
        return english_tokenizer

    def prepare_chinese_tokenizer(self) -> Tokenizer:
        """
        Fit a Keras tokeniser on the relevant source texts.

        :return: tokeniser that has been fit on Chinese source texts
        """

        chinese_tokenizer = Tokenizer(num_words=settings.MAX_NUM_SOURCE_TOKENIZER_WORDS, lower=False,
                                      char_level=False, filters=' ', oov_token='unk')
        chinese_tokenizer.fit_on_texts(self.input_texts)
        
        print(chinese_tokenizer)
        return chinese_tokenizer

    def get_tokenizer_info(self) -> Dict[str, Union[int, str]]:
        """
        Get a dictionary containing information about the English and Chinese tokenisers.

        :return: dictionary that contains information about the source and target language tokenisers
        """

        english_vocab_size = len(self.english_tokenizer.word_index) + 1
        chinese_vocab_size = len(self.chinese_tokenizer.word_index) + 1

        max_english_seq_length = find_max_sequence_length(self.target_texts)
        max_chinese_seq_length = find_max_sequence_length(self.input_texts)

        tokenizer_info = {
            'num_input_texts': len(self.input_texts),
            'num_target_texts': len(self.target_texts),
            'english_vocab_size': english_vocab_size,
            'chinese_vocab_size': chinese_vocab_size,
            'max_english_seq_length': max_english_seq_length,
            'max_chinese_seq_length': max_chinese_seq_length,
        }
        
        return tokenizer_info

    def print_parallel_corpus_info(self) -> None:
        """
        Print information about the obtained parallel corpus, for logging and debugging purposes.

        :return: None
        """

        print_asterisk_separator()
        print('Information about the parallel corpus:')
        print(f'Number of input sequences: {self.tokenizer_info["num_input_texts"]}')
        print(f'Number of target sequences: {self.tokenizer_info["num_target_texts"]}')
        print(f'English vocabulary size: {self.tokenizer_info["english_vocab_size"]}')
        print(f'Chinese vocabulary size: {self.tokenizer_info["chinese_vocab_size"]}')
        print(f'Max English sequence length: {self.tokenizer_info["max_english_seq_length"]}')
        print(f'Max Chinese sequence length: {self.tokenizer_info["max_chinese_seq_length"]}')
        print_asterisk_separator()

    def get_dimensions_from_tokenizers(self) -> Dict[str, int]:
        """
        Get the tokeniser information needed for generating batches of data.

        :return: dictionary containing information about dimensions needed for batch generation
        """

        dimensions = {"max_encoder_seq_length": self.tokenizer_info["max_chinese_seq_length"],
                      "max_decoder_seq_length": self.tokenizer_info["max_english_seq_length"],
                      "num_encoder_tokens": self.tokenizer_info["chinese_vocab_size"],
                      "num_decoder_tokens": self.tokenizer_info["english_vocab_size"]}

        return dimensions

    def initialise_generator(self, dataset_type: 'DatasetType', batch_size: int,
                             use_video_embeddings: bool) -> DataGenerator:
        """
        Initialise a new data generator.

        :param dataset_type: type of the dataset: training, validation, or test
        :param batch_size: the size of batches that the data generator will generate
        :param use_video_embeddings: whether the data output by the data generator should include video embeddings
        :return: a new data generator
        """

        dimensions = self.get_dimensions_from_tokenizers()

        generator = DataGenerator(batch_size=batch_size,
                                  dimensions=dimensions,
                                  dataset_type=dataset_type,
                                  input_token_index=self.input_token_index,
                                  target_token_index=self.target_token_index,
                                  use_video_embeddings=use_video_embeddings)
        
        
        return generator

    def compile_model(self) -> None:
        """
        Compile the VMT model.

        :return: None
        """

        optimizer = keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE)
        self.entire_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self) -> None:
        """
        Train the VMT model.

        :return: None
        """

        # Set up model checkpointing
        model_checkpoint_path = generate_model_checkpoint_path()
        model_summary_path = get_model_summary_path()
        create_directories(model_checkpoint_path, model_summary_path)
        checkpointer = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)

        # Set up a callback to reduce learning rate when after loss stopps improving
        reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=settings.REDUCE_LR_FACTOR,
                                               patience=settings.REDUCE_LR_PATIENCE, verbose=1, mode='auto',
                                               min_delta=0.0001, cooldown=0, min_lr=0)

        # Save the model architecture as a PNG file plot_model(self.entire_model, to_file=model_summary_path)
        plot_model(self.entire_model, to_file=model_summary_path)

        print(self.entire_model.summary())
        # Train model
        self.echo('Calling model.fit_generator!')
        history = self.entire_model.fit_generator(generator=self.training_generator,
                                                  validation_data=self.validation_generator,
                                                  use_multiprocessing=settings.USE_MULTIPROCESSING,
                                                  workers=settings.WORKERS,
                                                  epochs=settings.EPOCHS,
                                                  verbose=1,
                                                  callbacks=[checkpointer, reduce_lr_callback])
        self.echo('Training completed')

        history_save_path = generate_model_history_path()
        create_directories(history_save_path)
        #with open(history_save_path, 'wb') as fp:
            #pickle.dump(history, fp)

        self.echo(f'Training history saved to {history_save_path}')

    def get_reverse_input_token_index(self) -> Dict[int, str]:
        """
        Get a reverse index for source language tokens.

        :return: dictionary that maps indexes to source language tokens
        """

        return dict((index, token) for token, index in self.input_token_index.items())

    def get_reverse_target_token_index(self) -> Dict[int, str]:
        """
        Get a reverse index for target language tokens.

        :return: dictionary that maps indexes to target language tokens
        """
        return dict((index, token) for token, index in self.target_token_index.items())

    @staticmethod
    def load_model_from_path(latent_dim: int, filepath: str) -> VmtModel:
        """
        Load a saved VMT model from disk.

        :param latent_dim: number of units in the hidden state of the LSTM
        :param filepath: filepath where the VMT model is saved
        :return: loaded VMT model
        """

        return VmtModel(latent_dim=latent_dim,
                        filepath=filepath)

    def decode_sequence(self, input_seq: np.ndarray) -> List:
        """
        Translate an input sequence.

        :param input_seq: an input sequence that has been encoded
        :return: list of tokens in produced translation
        """

        # Encode the input as state vectors.
        # states_value = self.sampling_encoder.predict(input_seq)
        encoder_output, state_h, state_c = self.sampling_encoder.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.tokenizer_info["english_vocab_size"]))

        # Populate the first token of target sequence with the start token
        target_seq[0, 0, self.target_token_index[settings.SEQUENCE_START_TOKEN]] = 1.

        # Adapted from https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, hidden_state, cell_state = self.sampling_decoder.predict(
                [target_seq, state_h, state_c, encoder_output])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.reverse_target_token_index[sampled_token_index]
            decoded_sentence.append(sampled_word)

            # Exit condition: either hit max length or find sequence end token.
            if (sampled_word == settings.SEQUENCE_END_TOKEN or
                    len(decoded_sentence) > self.tokenizer_info["max_english_seq_length"]):
                stop_condition = True

            # Update the target sequence
            target_seq = np.zeros((1, 1, self.tokenizer_info["english_vocab_size"]))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [hidden_state, cell_state]

        return decoded_sentence

    def echo(self, message: str) -> None:
        """
        Output a progress message with an indication that it comes from the ModelManager class.

        :param message: message that will be output
        :return: None
        """

        self.message_count += 1
        print(f'[ModelManager: {self.message_count}] {message}')

    def translate_test_batch(self, batch_index: int, print_results: bool = False) -> List[Dict[str, Any]]:
        """
        Translate a single batch from the test dataset.

        :param batch_index: index of batch in the test dataset that should be translated
        :param print_results: whether the results should be printed as translations are produced
        :return: list of dictionaries where each entry in a dictionary contains an input sequence, a target sequence,
                 and a generated sequence
        """

        batch_info = self.test_generator.generate_batch_for_testing(batch_index)
        encoder_input_data = batch_info[0][0][0]
        input_texts = batch_info[1][0]
        target_texts = batch_info[1][1]

        results = []
        for idx, input_data in enumerate(encoder_input_data):
            input_sequence_enc = encoder_input_data[idx:idx + 1]
            input_sequence = input_texts[idx]
            target_sequence = target_texts[idx]
            decoded_sentence = self.decode_sequence(input_sequence_enc)

            input_sequence = remove_start_end_tokens(input_sequence)
            target_sequence = remove_start_end_tokens(target_sequence)
            decoded_sentence = remove_start_end_tokens(decoded_sentence)

            if print_results:
                print('-')
                print('Input sequence:', input_sequence)
                print('Target sequence:', target_sequence)
                print('Decoded sequence (token list):', decoded_sentence)
                print('Decoded sequence (string):', ' '.join(decoded_sentence))

            result = {'input_sequence': input_sequence,
                      'target_sequence': target_sequence,
                      'decoded_sequence': decoded_sentence}
            results.append(result)
        return results

    def translate_test_dataset(self) -> List[Dict[str, Any]]:
        """
        Translate the entire test dataset, one batch at a time.

        :return: list of dictionaries where each entry in a dictionary contains an input sequence, a target sequence,
                 and a generated sequence
        """
        batches_per_epoch = len(self.test_generator)

        results = []
        for idx in range(batches_per_epoch):
            self.echo(f'Translating test batch {idx + 1}/{batches_per_epoch}')
            batch_results = self.translate_test_batch(idx)
            #print(len(results))
            results.extend(batch_results)
            #print(results)
        return results
