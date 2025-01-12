from typing import Tuple, Optional

import torch
import torchaudio

import os
import hashlib
import re
import glob

import random
import math

from tensorflow.python.util import compat
from .utils import _download

DATA_URL = ['http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
            'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz']

OFFICIAL_TEST_URL = ['http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz',
                     'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz']

WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185
SR = 16000

def prepare_words_list(wanted_words: list) -> list:
    """Prepends common tokens to the custom word list.

    Args:
        wanted_words: List of strings containing the custom words.

    Returns:
        List with the standard silence and unknown tokens added.
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words

def which_set(filename: str, 
              validation_percentage: int, 
              testing_percentage: int) -> str:
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
        filename: File path of the data sample.
        validation_percentage: How much of the data set to use for validation.
        testing_percentage: How much of the data set to use for testing.

    Returns:
        String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                      (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

def prepare_data_index(data_dir: list, 
                       silence_percentage: int, 
                       unknown_percentage: int,
                       wanted_words:int, 
                       validation_percentage: int,
                       testing_percentage: int) -> Tuple[dict, dict]:
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
        wanted_words_index[wanted_word] = index + 2
    
    data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    
    # Look through all the subfolders to find audio samples
    search_path = glob.glob(os.path.join(data_dir, '*', '*.wav'))
    for wav_path in search_path:
        _, word = os.path.split(os.path.dirname(wav_path))
        word = word.lower()
        # Treat the '_background_noise_' folder as a special case, since we expect
        # it to contain long audio samples we mix in to improve training.
        if word == BACKGROUND_NOISE_DIR_NAME:
            continue
        all_words[word] = True
        set_index = which_set(wav_path, validation_percentage, testing_percentage)
        # If it's a known class, store its detail, otherwise add it to the list
        # we'll use to train the unknown label.
        if word in wanted_words_index:
            data_index[set_index].append({'label': word, 'file': wav_path})
        else:
            unknown_index[set_index].append({'label': word, 'file': wav_path})
    
    if not all_words:
        raise Exception('No .wavs found at ' + search_path)
    
    for index, wanted_word in enumerate(wanted_words):
        if wanted_word not in all_words:
            raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
        set_size = len(data_index[set_index])
        silence_size = int(math.ceil(set_size * silence_percentage / 100))
        for _ in range(silence_size):
            data_index[set_index].append({
                'label': SILENCE_LABEL,
              'file': silence_wav_path
            })
      
      # Pick some unknowns to add to each partition of the data set.
        random.shuffle(unknown_index[set_index])
        unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
        data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
        random.shuffle(data_index[set_index])
    
    # Prepare the rest of the result data structure.
    words_list = prepare_words_list(wanted_words)
    word_to_index = {}
    for word in all_words:
        if word in wanted_words_index:
            word_to_index[word] = wanted_words_index[word]
        else:
            word_to_index[word] = UNKNOWN_WORD_INDEX
    word_to_index[SILENCE_LABEL] = SILENCE_INDEX
    
    return data_index, word_to_index

def prepare_official_test(data_dir: str, 
                          wanted_words: list) -> Tuple[list, dict]:
    """
    In case of using the companion for evaluation. We also need to prepare it like we did which makes sure that
    everything will be synchronized. 
    Args:
    data_dir: str
        Data directory
    wanted_words: list
    """
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
        wanted_words_index[wanted_word] = index + 2
    wanted_words_index[SILENCE_LABEL] = SILENCE_INDEX
    wanted_words_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX

    test_data = []
    
    search_path = glob.glob(os.path.join(data_dir, '*', '*.wav'))
    for wav_path in search_path:
        _, word = os.path.split(os.path.dirname(wav_path))
        word = word.lower()
        test_data.append({'label': word, 'file': wav_path})
    
    return test_data, wanted_words_index

class SpeechCommands(torch.utils.data.Dataset):
    """
    This Dataset is equivalent to SPEECHCOMMANDS Dataset of Pytorch in the way of using.
    All the set up was based on the original paper, which you can find here: 
    <https://arxiv.org/pdf/1804.03209.pdf>
    In the section 7, the authors gave us the implementation for GSC 12 with 10 keywords
    ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'] and 2 additional keywords
    are '_silence_' and '_unknown_', which can be found here: 
    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py#L188>

    Our implementation almost based on this sample implementation, so you may find some familar codes here!

    Args
    root: str
        Default directory for using and downloading data.
    download: bool
        Whether download the file from our given url.
    version: int
        Version of Google Speech Commands dataset, includes [1, 2]
    subset: str
        Select a subset of the dataset ['training', 'validation', 'testing', 'official_testing']
    transform: 
        Data transformation.
    """

    def __init__(self, 
                 root: str,
                 download: bool = True,
                 version: int = 2,
                 subset: str = 'training',
                 wanted_words: Optional[list] = None) -> None:
        super().__init__()
        self.wanted_words = wanted_words if wanted_words else WORDS

        if subset != 'official_testing':
            if download:
                url = DATA_URL[version-1]
                filename = os.path.split(url)[-1]
                print('>> Downloading %s' % filename)
                _download(url, root)
            data_index, self.word_to_index = prepare_data_index(root, 
                                                                silence_percentage = 10,
                                                                unknown_percentage = 10,
                                                                wanted_words = self.wanted_words,
                                                                validation_percentage = 10,
                                                                testing_percentage = 10)
            self.dataset = data_index[subset]
        else:
            if download:
                url = OFFICIAL_TEST_URL[version-1]
                filename = os.path.split(url)[-1]
                print('>> Downloading %s' % filename)
                _download(url, root)
            self.dataset, self.word_to_index = prepare_official_test(root, 
                                                                     wanted_words = self.wanted_words)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]
        filepath = row['file']
        label = row['label']
        if label == SILENCE_LABEL:
            wav = torch.zeros([1, SR])
        else:
            wav, _ = torchaudio.load(filepath)
        return wav, self.word_to_index[label]

if __name__ == '__main__':
    # Sample using
    transform = torchaudio.transforms.MelSpectrogram(SR)

    train_dataset = SpeechCommands('./GSC_12', 
                                   download = True, 
                                   subset = 'training')
    val_dataset = SpeechCommands('./GSC_12', 
                                  download = True, 
                                  subset = 'validation')
    test_dataset = SpeechCommands('./GSC_12_test', 
                                  download = True, 
                                  subset = 'testing')

