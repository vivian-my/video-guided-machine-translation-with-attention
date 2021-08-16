 # 0. About
The experimental machine translation system developed by Paulius Staugaitis is used in the "Multimodal Machine Translation through visuals" final year project.
The following documentation describes how to set up the environment and run the translation software.

# 1. Setup
## 1.1 Obtaining the Required Data
The VaTeX dataset should be downloaded from https://eric-xw.github.io/vatex-website/download.html

In particular, the following files are needed:
* Training dataset: https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json
* Validation dataset: https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json
* I3D video features for the training and validation datasets: https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json

JSON files may be renamed to use shorter names, and the video feature archive should be unzipped (caution: there is a
 large number of files in the archive). The values in the included settings files in the `settings` directory provide
 examples of how the directories and files can be structured.

To replicate results described in the final report, the original training dataset should be split into two datasets 
(new training and new test), as detailed in Subsection 4.1.3 "Dataset Splitting" of the report. Although for testing
purposes, any files with the same structure will work.

The filepaths of the datasets should be entered in the settings files, described next.

## 1.2 Configuring the Settings Files
Depending on the environment that the translation system is going to be run in, one or more settings files should
 be configured. 
 
`settings/base.py` should always be configured, as any other "child settings" files "inherit" from it. If the
 translation system is run on a personal computer, configuring just `settings/base.py` is enough. This settings
 module specifies default values that are used to quickly test whether the code is working correctly while developing
 the system (i.e. not for generating meaningful results). For instance, it only uses 96 sequence pairs for
 training the model instead of the 116955 sequence pairs that `settings/google_cloud.py` uses.
 
A second settings module should be configured if one wants to train the translation system on Google Cloud 
(`settings/google_cloud.py`, Floydhub (`settings/floydhub.py`), or Leeds University HPC cluster (`settings/hpc.py`).
Every settings value can be adjusted as needed. E.g. it's not necessary to always use the full dataset. 

## 1.3 How Settings Modules Work
The reasoning behind using settings modules is provided in Subsection 4.3.6 "Managing Settings" of the final report.
When the command-line interface is used, one or two settings modules should be specified in the command by setting the
 environment variable `SIMPLE_SETTINGS` to an appropriate value. Specifying no settings modules will result 
 in an error, as the entire system depends on the values inside them.

If two modules are specified, the duplicate values contained in the second module override the values in the first
module. For example, the value `TRAINING_SEQUENCE_PAIRS_LIMIT = 116955` in `settings/google_cloud.py` would override
the value `TRAINING_SEQUENCE_PAIRS_LIMIT = 96` in `settings/base.py` when the following command is used:
`SIMPLE_SETTINGS=settings.base,settings.google_cloud python run.py preprocess`

The following examples illustrate this:
```
# 1) Run the script on a personal computer 
SIMPLE_SETTINGS=settings.base python run.py preprocess
# 2) Run the script on Google Cloud
SIMPLE_SETTINGS=settings.base,settings.google_cloud python run.py preprocess
```
The first command starts the model training process on a personal computer, using values from `settings/base.py`.
The second command starts the model training process on Google Cloud, using a combination of values from both 
`settings/base.py` and `settings/google_cloud.py`

The above command format is suitable to be used when the current working directory is the base directory of the
project (i.e.the directory that contains the `run.py` file).

## 1.4 Setting up the Virtual Environment
Python 3.6.8 and pip 20.1 was used during the development. It is recommended to use the same Python version.

First, a new virtual environment should be created:
```
python3 -m venv <myenvname>
```

The new virtual environment should then be activated:
```
source myenvname/bin/activate
```

Finally, the required packages should be installed:
```
pip install -r requirements.txt
```

# 2. Running the Software
After the set-up is completed, `run.py` can be used to run the translation system.
The boolean variable `USE_VIDEO_EMBEDDINGS` in the settings determines whether **NMT-1** or **VMT-1** is run.

The examples that follow use the `settings/base.py` module but they can be modified for different environments,
as described above. Sample arguments are used in the examples (test batch at index 0 is chosen).

The help page for each command choice (see 2.1.2) provides more details about its parameters.

## 2.1. Displaying CLI Help
### 2.1.1. General Help
```
SIMPLE_SETTINGS=settings.base python run.py --help
```
### 2.1.2. Command-Specific Help

```
# Replace "train" with the relevant command
SIMPLE_SETTINGS=settings.base python run.py train --help
```


## 2.2. Pre-processing Data
```
SIMPLE_SETTINGS=settings.base python run.py preprocess
```

## 2.3. Training Model
```
SIMPLE_SETTINGS=settings.base python run.py train
```

## 2.4. Evaluating Translation Quality
### 2.4.1. Evaluating Translation Quality on One Batch
```
SIMPLE_SETTINGS=settings.base python run.py translate_single_batch -m /home/paul/PycharmProjects/fyp/saved/2020-05-04_0022/model-04_3.231.hdf5 -b 0

```

### 2.4.2. Evaluating Translation Quality on the Test Dataset
```
SIMPLE_SETTINGS=settings.base python run.py translate_test_dataset -m /home/paul/PycharmProjects/fyp/saved/2020-05-04_0022/model-04_3.231.hdf5
```
