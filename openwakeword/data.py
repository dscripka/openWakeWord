# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# imports
from math import comb
import os
import random
from tqdm import tqdm
from typing import List
import numpy as np
import torch
from speechbrain.dataio.dataio import read_audio
import torchaudio
import mutagen

# Load audio clips and structure into clips of the same length

def stack_clips(audio_data, clip_size=16000*2):
    """
    Takes an input list of 1D arrays (of different lengths), concatenates them together,
    and then extracts clips of a uniform size by dividing the combined array.
    Also converts the resulting array to 16-bit PCM format.

    Args:
        audio_data (List[ndarray]): A list of 1D numpy arrays to combine and stack
        clip_size (int): The desired total length of the uniform clip size (in samples)

    Returns:
        ndarray: A N by `clip_size` array with the audio data, converted to 16-bit PCM
    """

    # Combine all clips into single clip
    combined_data = np.hstack((audio_data))

    # Get chunks of the specified size
    new_examples = []
    for i in range(0, combined_data.shape[0], clip_size):
        chunk = combined_data[i:i+clip_size]
        if chunk.shape[0] != clip_size:
            chunk = np.hstack((chunk, np.zeros(clip_size - chunk.shape[0])))
        new_examples.append(chunk)
    
    # Convert to 16-bit PCM data
    X = (np.array(new_examples).astype(np.float32)*32767).astype(np.int16)

    return X

def load_audio_clips(files, clip_size=32000):
    """
    Takes the specified audio files and shapes them into an array of N by `clip_size`,
    where N is determined by the length of the audio files and `clip_size` at run time.
    
    Clips longer than `clip size` are truncated and extended into the N+1 row.
    Clips shorter than `clip_size` are combined with the previous or next clip
    (except for the last clip in `files`, which is ignored if it is too short.)
    
    Args:
        files (List[str]): A list of filepaths
        clip_size (int): The number of samples (of 16khz audio) for all of the rows in the array
        
    Returns:
        ndarray: A N by `clip_size` array with the audio data, converted to 16-bit PCM
    """
    
    # Load audio files
    audio_data = []
    for i in files:
        try:
            audio_data.append(read_audio(i))
        except ValueError:
            continue

    # Get shape of output array
    N = sum([i.shape[0] for i in audio_data])//clip_size
    X = np.empty((N, clip_size))
    
    # Add audio data to rows
    previous_row_remainder = None
    cnt = 0
    for row in audio_data:
        row = np.hstack((previous_row_remainder, row))
        while row.shape[0] >= clip_size:
            X[cnt, :] = row[0:clip_size]
            row = row[clip_size:]
            cnt += 1
        
        previous_row_remainder = row if row.size > 0 else None
            
    # Convert to 16-bit PCM data
    X = (X*32767).astype(np.int16)

    return X

# Dato I/O utils

def filter_audio_paths(target_dirs, min_length_secs, max_length_secs, duration_method="size"):
    """
    Gets the paths of wav files in a flat target directory, automatically filtering
    out files below/above the specified length (in seconds). Assumes that all
    wav files are sampled at 16khz, are single channel, and have 16-bit PCM data.

    Uses `os.scandir` in Python for highly efficient file system exploration,
    and doesn't require loading the files into memory for length estimation.

    Args:
        target_dir (List[str]): The target directories containing the wav files
        min_length_secs (float): The minimum length in seconds (otherwise the clip is skipped)
        max_length_secs (float): The maximum length in seconds (otherwise the clip is skipped)
        duration_method (str): Whether to use the file size ('size'), or header information ('header')
                               to estimate the duration of the audio file. 'size' is generally
                               much faster, but assumes that all files in the target directory
                               are the same type, sample rate, and bitrate.
    
    Returns:
        tuple: A list of strings corresponding to the paths of the wav files that met the length criteria,
               and a list of their durations (in seconds)
    """

    file_paths = []
    durations = []
    for target_dir in target_dirs:
        sizes = []
        dir_paths = []
        for i in os.scandir(target_dir):
            dir_paths.append(i.path)
            file_paths.append(i.path)
            sizes.append(i.stat().st_size)
        
        if duration_method == "size":
            durations.extend(estimate_clip_duration(dir_paths, sizes))

        elif duration_method == "header":
            durations.extend([get_clip_duration(i) for i in tqdm(dir_paths)])

    filtered = [(i,j) for i,j in zip(file_paths, durations) if j >= min_length_secs and j <= max_length_secs]
    return [i[0] for i in filtered], [i[1] for i in filtered]


def estimate_clip_duration(audio_files: list, sizes: list):
    """Estimates the duration of each audio file in a list.
    
    Assumes that all of the audio files have the same audio format,
    bit depth, and sample rate.

    Args:
        audio_file (str): A list of audio file paths
        sizes (int): The size of each audio file in bytes

    Returns:
        list: A list of durations (in seconds) for the audio files
    """

    # Determine file type by checking the first file
    details = torchaudio.info(audio_files[0])

    # Caculate any correction factors needed from the first file
    details = mutagen.File(audio_files[0])
    correction = 8*os.path.getsize(audio_files[0]) - details.info.bitrate*details.info.length

    # Estimate duration for all remaining clips from file size only
    durations = []
    for size in sizes:
        durations.append((size*8-correction)/details.info.bitrate)
    
    return durations

def get_clip_duration(clip):
    """Gets the duration of an audio clip in seconds from file header information"""
    metadata = torchaudio.info(clip)
    return metadata.num_frames/metadata.sample_rate

def get_wav_duration_from_filesize(size, nbytes=2):
    """
    Calculates the duration (in seconds) from a WAV file, assuming it contains 16 khz single-channel audio.
    The bit depth is user specified, and defaults to 2 for 16-bit PCM audio.

    Args:
        size (int): The file size in bytes
        nbytes (int): How many bytes for each data point in the audio (e.g., 16-bit is 2, 32-bit is 4, etc.)
    
    Returns:
        float: The duration of the audio file in seconds
    """
    return (size-44)/nbytes/16000


# Data augmentation utility function
def mix_clips_batch(
        foreground_clips: List[str],
        background_clips: List[str],
        combined_size: int,
        batch_size: int=32,
        snr_low: float=0,
        snr_high: float=0,
        start_index: List[int]=[],
        seed: int=None
    ):
    """
    Mixes foreground and background clips at a random SNR level in batches.
    
    References: https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html and
    https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.speech_augmentation.html#speechbrain.processing.speech_augmentation.AddNoise
    
    Args:
        foreground_clips (List[str]): A list of paths to the foreground clips
        background_clips (List[str]): A list of paths to the background clips (randomly selected for each foreground clip)
        combined_size (int): The total length (in samples) of the combined clip. If needed, the background clips are duplicated or truncated to reach this length.
        batch_size (int): The batch size
        snr_low (float): The low SNR level of the mixing in db
        snr_high (float): The high snr level of the mixing in db
        start_index (List[int]): The starting position (in samples) for the foreground clip to start in the background clip.
        seed (int): A random seed

    Returns:
        generator: Returns a generator that yields batches of mixed foreground/background audio
    """
    # Set random seed, if needed
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    # Set start indices, if needed
    if not start_index:
        start_index = [0]*batch_size
    
    for i in range(0, len(foreground_clips), batch_size):
        # Load foreground clips and truncate (if needed)
        foreground_clips_batch = [read_audio(i)[0:combined_size] for i in foreground_clips[i:i+batch_size]]
        
        # Load background clips and pad/truncate as needed
        background_clips_batch = [read_audio(i) for i in random.sample(background_clips, batch_size)]
        for ndx, background_clip in enumerate(background_clips_batch):
            if background_clip.shape[0] < combined_size:
                background_clips_batch[ndx] = background_clip.repeat(
                    np.ceil(combined_size/background_clip.shape[0])
                )[0:combined_size]
            elif background_clip.shape[0] > combined_size:
                r = np.random.randint(0, max(1, background_clip.shape[0] - combined_size))
                background_clips_batch[ndx] = background_clip[r:r + combined_size]
        
        # Mix clips at snr levels
        snrs_db = np.random.uniform(snr_low, snr_high, batch_size)
        mixed_clips_batch = []
        for fg, bg, snr, start in zip(foreground_clips_batch, background_clips_batch,
                                      snrs_db, start_index):
            fg_rms, bg_rms = fg.norm(p=2), bg.norm(p=2)
            snr = 10 ** (snr / 20)
            scale = snr * bg_rms / fg_rms
            start = min(start, combined_size - fg.shape[0])
            bg[start:start + fg.shape[0]] = bg[start:start + fg.shape[0]] + scale*fg
            mixed_clips_batch.append(bg / 2)
            
        mixed_clips_batch = torch.vstack(mixed_clips_batch)
        
        # Normalize clips only if max value is outside of [-1, 1]
        abs_max, _ = torch.max(
            torch.abs(mixed_clips_batch), dim=1, keepdim=True
        )
        mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

        # Convert to 16-bit PCM audio
        mixed_clips_batch = (mixed_clips_batch.numpy()*32767).astype(np.int16)

        yield mixed_clips_batch

# Load batches of data from mmaped numpy arrays
class mmap_batch_generator:
    """
    A generator class designed to dynamically build batches from mmaped numpy arrays.

    The generator will return tuples of (data, labels) with a batch size determined
    by the `n_per_class` initialization argument. When a mmaped numpy array has been
    fully interated over, it will restart at the zeroth index automatically.
    """
    def __init__(self, data_files, n_per_class, data_transform_funcs = {}, label_transform_funcs = {}):
        """
        Initialize the generator object

        Args:
            data_files (dict): A dictionary of labels (as keys) and on-disk numpy array paths (as values).
                               Keys should be integer strings representing class labels.
            n_per_class (dict): A dictionary with integer string labels (as keys) and number of example per batch
                               (as values).
            data_transform_funcs (dict): A dictionary of transformation functions to apply to each batch of per class
                                    data loaded from the mmaped array. For example, with an array of shape
                                    (batch, timesteps, features), if the goal is to half the timesteps per example,
                                    (effectively doubling the size of the batch) this function could be passed:
                                    
                                    lambda x: np.vstack(
                                        (x[:, 0:timesteps//2, :], x[:, timesteps//2:, :]
                                    ))

                                    The user should incorporate the effect of any transform on the values of the 
                                    `n_per_class` argument accordingly, in order to end of with the desired
                                    total batch size for each iteration of the generator.
            label_transform_funcs (dict): A dictionary of transformation functions to apply to each batch of labels.
                                          For example, strings can be mapped to integers or one-hot encoded,
                                          groups of classes can be merged together into one, etc.
                                    
        """
        # inputs
        self.data_files = data_files
        self.n_per_class = n_per_class
        self.data_transform_funcs = data_transform_funcs
        self.label_transform_funcs = label_transform_funcs
        
        # Get array mmaps and counter object
        self.data = {label:np.load(fl, mmap_mode='r') for label, fl in data_files.items()}
        self.data_counter = {label:0 for label in data_files.keys()}
        self.shapes = {label:self.data[label].shape for label in self.data.keys()}

        # Get estimated batches per epoch
        batch_size = sum([val for val in n_per_class.values()])
        batches_per_epoch = sum([i[0] for i in self.shapes.values()])//batch_size
        self.batch_per_epoch = batches_per_epoch
        print("Batches/steps per epoch:", batches_per_epoch)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # Build batch
        while True:
            X, y = [], []
            for label, n in self.n_per_class.items():
                # Restart at zeroth index if an array reaches the end
                if self.data_counter[label] >= self.shapes[label][0]:
                    self.data_counter[label] = 0
                    self.data[label] = np.load(self.data_files[label], mmap_mode='r')

                # Get data from mmaped file
                x = self.data[label][self.data_counter[label]:self.data_counter[label]+n]
                self.data_counter[label] += x.shape[0]

                # Transform data
                if self.data_transform_funcs and self.data_transform_funcs.get(label):
                    x = self.data_transform_funcs[label](x)

                # Make labels for data (following whatever the current shape of `x` is)
                y_batch = [label]*x.shape[0]

                # Transform labels
                if self.label_transform_funcs and self.label_transform_funcs.get(label):
                    y_batch = self.label_transform_funcs[label](y_batch)

                # Add data to batch
                X.append(x)
                y.extend(y_batch)

            return np.vstack(X), np.array(y)