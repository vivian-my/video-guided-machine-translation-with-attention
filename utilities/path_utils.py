import os
import pathlib
from datetime import datetime
from os import listdir
from os.path import join, isfile
from typing import List

from simple_settings import settings

from dataset_type import DatasetType


def get_project_base_dir() -> str:
    """
    Get the absolute path of the project base directory.

    :return: absolute path of the project base directory
    """

    utilities_dir = pathlib.Path(__file__).parent.absolute()
    base_dir = os.path.abspath(os.path.join(utilities_dir, '..'))
    return base_dir


def get_absolute_path(path_in_project: str) -> str:
    """
    Get the absolute path of a relative path in the project directory.

    :param path_in_project: relative path in the project directory
    :return: absolute path equivalent of the relative path supplied
    """

    base_dir = get_project_base_dir()
    absolute_path = os.path.join(base_dir, path_in_project)
    return absolute_path


def create_directory(path: str) -> None:
    """
    Create the missing directories for the supplied filepath.

    :param path: path
    :return: None
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)


def create_directories(*paths: str) -> None:
    """
    Create the missing directories for all of the supplied filepaths.

    :param paths: paths
    :return: None
    """

    for path in paths:
        create_directory(path)


def generate_model_directory_name() -> str:
    """
    Generate a name for the model directory, based on the current date and time.

    :return: model directory name
    """
    return datetime.today().strftime('%Y-%m-%d_%H%M')


def get_model_save_directory() -> str:
    """
    Get the directory where the model should be saved at checkpoints.

    :return: model directory name
    """

    unique_directory_name = generate_model_directory_name()
    full_directory_name = os.path.join(settings.PROJECT_BASE_DIR, settings.MODEL_SAVE_DIR, unique_directory_name)
    return full_directory_name


def generate_model_checkpoint_path() -> str:
    """
    Generate the filepath for model checkpoints.

    :return: model checkpoint filepath
    """

    save_directory = get_model_save_directory()
    path = os.path.join(save_directory, 'model-{epoch:02d}_{val_loss:.3f}.hdf5')
    return path


def generate_model_history_path() -> str:
    """
    Generate the filepath for saving model history.

    :return: model history filepath
    """

    save_directory = get_model_save_directory()
    path = os.path.join(save_directory, 'history.pkl')

    return path


def generate_results_csv_path() -> str:
    """
    Generate the filepath for saving results of a translation run as a CSV.

    :return: path for the results CSV file
    """

    timestamp = datetime.today().strftime('%Y-%m-%d_%H%M')
    filename = f'results-{timestamp}.csv'
    filepath = get_absolute_path(f'output/{filename}')

    return filepath


def get_newest_saved_model_path(save_dir: str) -> str:
    """
    Get the latest saved model filepath in a specified directory.

    :param save_dir: directory where to look for saved models
    :return: path for the results CSV file
    """

    model_files = [join(save_dir, file) for file in listdir(save_dir) if
                   isfile(join(save_dir, file)) and file.endswith('.hdf5')]

    return max(model_files, key=os.path.getctime)


def get_model_summary_path() -> str:
    """
    Get the filepath where a model diagram should be saved.

    :return: filepath for a model summary
    """

    save_directory = get_model_save_directory()
    path = os.path.join(save_directory, 'model.png')
    return path


def get_video_embeddings_directory(dataset_type: 'DatasetType') -> str:
    """
    Given a dataset type, get the directory of the corresponding video embeddings.

    :param dataset_type: type of the dataset: training, validation, or test
    :return: directory where the corresponding video embeddings are stored
    """

    if dataset_type == DatasetType.TRAINING:
        directory = settings.TRAINING_EMBEDDINGS_DIR
    elif dataset_type == DatasetType.VALIDATION:
        directory = settings.VALIDATION_EMBEDDINGS_DIR
    elif dataset_type == DatasetType.TEST:
        directory = settings.TEST_VIDEO_EMBEDDINGS_DIR
    else:
        raise ValueError('Invalid dataset type supplied')

    return directory
