import logging
from typing import Dict, List, Any

import click
import pandas as pd
from simple_settings import settings

from data_processing import preprocess_all_datasets
from model_manager import ModelManager
from utilities.evaluation_utils import calculate_bleu
from utilities.path_utils import generate_results_csv_path, create_directory

logging.basicConfig(stream=settings.LOGGING_STREAM, level=settings.LOGGING_LEVEL)


def echo(message: str) -> None:
    """
    Output a progress message with an indication that it comes from the "run.py" script.

    :return: None
    """

    click.echo(f'[run.py] {message}')


def output_results_to_csv(results: List[Dict[str, Any]]) -> None:
    """
    Save the results of a translation evaluation to a CSV file.

    :param results: a dictionary where each entry has an input sequence, target sequence, and decoded sequence.
    :return: None
    """

    df = pd.DataFrame(results)
    filepath = generate_results_csv_path()
    create_directory(filepath)
    df.to_csv(filepath, encoding='utf-8', index=False,
              columns=['input_sequence', 'target_sequence', 'decoded_sequence'])
    echo(f'Results saved to {filepath}')


@click.command()
def preprocess() -> None:
    """
    Pre-process all the required data: training, validation, and  test datasets.

    :return: None
    """

    echo('Pre-processing training, validation, and  test datasets')
    preprocess_all_datasets()
    echo('Pre-processing completed')


@click.command()
def train() -> None:
    """
    Train the model, using one of the settings modules chosen with a command-line argument.

    :return: None
    """

    echo('Initialising model')
    model_manager = ModelManager(prepare_for_training=True)
    echo('Starting model training')
    model_manager.train_model()
    echo('Model training completed')


@click.command()
@click.option('--model-path', '-m', required=True, type=click.Path(exists=True))
@click.option('--batch-num', '-b', required=True, type=int)
@click.option('--output-csv', '-o', default=False)
@click.option('--calculate_score', '-c', default=True)
def translate_single_batch(model_path: str, batch_num: int, output_csv: bool, calculate_score: bool) -> None:
    """
    Translate a single specified batch from the test dataset.

    :param model_path: path of saved model to be loaded and used to perform the translation
    :param batch_num: number of the batch from the test dataset to process
    :param output_csv: whether to output translation results to a CSV file
    :param calculate_score: whether to calculate the BLEU score for the translations produced
    :return: None
    """

    echo(f'Loading saved model from {model_path}')
    model_manager = ModelManager(prepare_for_sampling=True,
                                 load_model=True,
                                 saved_model_path=model_path)

    echo(f'Translating a single batch (number {batch_num}) from the test dataset')
    results = model_manager.translate_test_batch(batch_num)
    echo('Completed')

    if output_csv:
        output_results_to_csv(results)

    if calculate_score:
        echo('Calculating BLEU score')
        bleu_score = calculate_bleu(results)
        print(f'BLEU score: {bleu_score}')


@click.command()
@click.option('--model-path', '-m', required=True, type=click.Path(exists=True))
@click.option('--output-csv', '-o', default=False)
@click.option('--calculate_score', '-c', default=True)
def translate_test_dataset(model_path: str, output_csv: bool, calculate_score: bool) -> None:
    """
    Translate the entire test dataset.

    :param model_path: path of saved model to be loaded and used to perform the translation
    :param output_csv: whether to output translation results to a CSV file
    :param calculate_score: whether to calculate the BLEU score for the translations produced
    :return: None
    """

    echo(f'Loading saved model from {model_path}')
    model_manager = ModelManager(prepare_for_sampling=True,
                                 load_model=True,
                                 saved_model_path=model_path)

    echo('Processing the entire test dataset')
    results = model_manager.translate_test_dataset()
    echo('Completed')

    if output_csv:
        output_results_to_csv(results)

    if calculate_score:
        echo('Calculating BLEU score')
        bleu_score = calculate_bleu(results)
        print(f'BLEU score: {bleu_score}')


@click.group()
def command_line_interface():
    pass


command_line_interface.add_command(preprocess)
command_line_interface.add_command(train)
command_line_interface.add_command(translate_single_batch)
command_line_interface.add_command(translate_test_dataset)

if __name__ == '__main__':
    command_line_interface()
