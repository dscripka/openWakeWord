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
from multiprocessing.pool import ThreadPool
import os
import re
import logging
from functools import partial
from pathlib import Path
import random
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import itertools
import pronouncing
import torch
import audiomentations
import torch_audiomentations
from numpy.lib.format import open_memmap
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing.signal_processing import reverberate
import torchaudio
import mutagen
import acoustics


# Load audio clips and structure into clips of the same length
def stack_clips(audio_data, clip_size=16000*2):
    """
    Takes an input list of 1D arrays (of different lengths), concatenates them together,
    and then extracts clips of a uniform size by dividing the combined array.

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

    return np.array(new_examples)


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


# Convert clips with sox
def _convert_clip(input_file, output_file, backend="ffmpeg"):
    if backend == "sox":
        cmd = f"sox \"{input_file}\" -G -r 16000 -c 1 -b 16 \"{output_file}\""
    elif backend == "ffmpeg":
        cmd = f"ffmpeg -y -i \"{input_file}\" -ar 16000 \"{output_file}\""
    os.system(cmd)
    return None


def convert_clips(input_files, output_files, sr=16000, ncpu=1, backend="ffmpeg"):
    """
    Converts files in parallel with multithreading using Sox or ffmpeg.

    Intended to only convert input audio files into single-channel, 16 khz clips.

    Args:
        input_files (List[str]): A list of paths to input files
        output_files (List[str]): A list of paths to output files, corresponding 1:1 to the input files
        sr (int): The output sample rate of the converted clip
        ncpu (int): The number of CPUs to use for the conversion
        backend (str): The utilty to use for conversion, "sox" or "ffmpeg"

    Returns:
        None
    """
    # Setup ThreadPool object
    pool = ThreadPool(processes=ncpu)

    # Set backend for conversion
    f = partial(_convert_clip, backend=backend)

    # Submit jobs
    pool.starmap(f, [(i, j) for i, j in zip(input_files, output_files)])


def filter_audio_paths(target_dirs, min_length_secs, max_length_secs, duration_method="size", glob_filter=None):
    """
    Gets the paths of wav files in flat target directories, automatically filtering
    out files below/above the specified length (in seconds). Assumes that all
    wav files are sampled at 16khz, are single channel, and have 16-bit PCM data.

    Uses `os.scandir` in Python for highly efficient file system exploration,
    and doesn't require loading the files into memory for length estimation.

    Args:
        target_dir (List[str]): The target directories containing the audio files
        min_length_secs (float): The minimum length in seconds (otherwise the clip is skipped)
        max_length_secs (float): The maximum length in seconds (otherwise the clip is skipped)
        duration_method (str): Whether to use the file size ('size'), or header information ('header')
                               to estimate the duration of the audio file. 'size' is generally
                               much faster, but assumes that all files in the target directory
                               are the same type, sample rate, and bitrate. If None, durations are not calculated.
        glob_filter (str): A pathlib glob filter string to select specific files within the target directory

    Returns:
        tuple: A list of strings corresponding to the paths of the wav files that met the length criteria,
               and a list of their durations (in seconds)
    """

    file_paths = []
    durations = []
    for target_dir in target_dirs:
        sizes = []
        dir_paths = []
        if glob_filter:
            dir_paths = [str(i) for i in Path(target_dir).glob(glob_filter)]
            file_paths.extend(dir_paths)
            sizes.extend([os.path.getsize(i) for i in dir_paths])
        else:
            for i in tqdm(os.scandir(target_dir)):
                dir_paths.append(i.path)
                file_paths.append(i.path)
                sizes.append(i.stat().st_size)

        if duration_method == "size":
            durations.extend(estimate_clip_duration(dir_paths, sizes))

        elif duration_method == "header":
            durations.extend([get_clip_duration(i) for i in tqdm(dir_paths)])

    if durations != []:
        filtered = [(i, j) for i, j in zip(file_paths, durations) if j >= min_length_secs and j <= max_length_secs]
        return [i[0] for i in filtered], [i[1] for i in filtered]
    else:
        return file_paths, []


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


def estimate_mp3_duration(fpath):
    """Estimates the duration of an MP3 file from metadata and file-size.
    Is only accurate for 16000 khz sample rate audio with a relatively
    constant bit-rate.

    Args:
        fpath (str): The input path to the MP3 file

    Returns:
        float: The duration of the MP3 file in seconds
    """

    conversion_factors = {
        "16_khz_single_channel": 0.000333318208471784,
        "16_khz_stereo": 0.000333318208471784/2
    }

    duration_seconds = 0
    try:
        md = torchaudio.info(fpath)
    except RuntimeError:
        return duration_seconds

    nbytes = os.path.getsize(fpath)
    if md.num_channels == 1:
        if md.sample_rate == 16000:
            duration_seconds = nbytes*conversion_factors["16_khz_single_channel"]
    elif md.num_channels == 2:
        if md.sample_rate == 16000:
            duration_seconds = nbytes*conversion_factors["16_khz_stereo"]

    return duration_seconds


def get_clip_duration(clip):
    """Gets the duration of an audio clip in seconds from file header information"""
    try:
        metadata = torchaudio.info(clip)
    except RuntimeError:  # skip cases where file metadata can't be read
        return 0

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
        labels: List[int] = [],
        batch_size: int = 32,
        snr_low: float = 0,
        snr_high: float = 0,
        start_index: List[int] = [],
        foreground_durations: List[float] = [],
        foreground_truncate_strategy: str = "random",
        rirs: List[str] = [],
        rir_probability: int = 1,
        volume_augmentation: bool = True,
        generated_noise_augmentation: float = 0.0,
        shuffle: bool = True,
        return_sequence_labels: bool = False,
        return_background_clips: bool = False,
        return_background_clips_delay: Tuple[int, int] = (0, 0),
        seed: int = 0
        ):
    """
    Mixes foreground and background clips at a random SNR level in batches.

    References: https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html and
    https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.speech_augmentation.html#speechbrain.processing.speech_augmentation.AddNoise

    Args:
        foreground_clips (List[str]): A list of paths to the foreground clips
        background_clips (List[str]): A list of paths to the background clips (randomly selected for each
                                      foreground clip)
        combined_size (int): The total length (in samples) of the combined clip. If needed, the background
                             clips are duplicated or truncated to reach this length.
        labels (List[int]): A list of integer labels corresponding 1:1 for the foreground clips. Will be updated
                            as needed with foreground clips to ensure that mixed clips retain the proper labels.
        batch_size (int): The batch size
        snr_low (float): The low SNR level of the mixing in db
        snr_high (float): The high snr level of the mixing in db
        start_index (List[int]): The starting position (in samples) for the foreground clip to start in
                                 the background clip. If the foreground clip is longer than `combined_size`
                                 when starting at this point, the foreground clip will be truncated
                                 according to the `foreground_truncate_strategy` argument.
        foreground_durations (List[float]): The desired duration of each foreground clip (in seconds)
        foreground_truncate_strategy (str): The method used to truncate the foreground clip, if needed based on the
                                            `start_index`, `foreground_durations`, and `combined_size` arguments.
                                            See the options in the `truncate_clip` method.
        rirs (List[str]): A list of paths to room impulse response functions (RIR) to convolve with the
                          clips to simulate different recording environments. Applies a single random selection from the
                          list RIR file to the entire batch. If empty (the default), nothing is done.
        rir_probability (float): The probability (between 0 and 1) that the batch will be convolved with a RIR file.
        volume_augmentation (bool): Whether to randomly apply volume augmentation to the clips in the batch.
                                    This simply scales the data of each clip such that the maximum value is is between
                                    0.02 and 1.0 (the floor shouldn't be zero as beyond a certain point the audio data
                                    is no longer valid).
        generated_noise_augmentation: The probability of further mixing the mixed clip with generated random noise.
                                      Will be either "white", "brown", "blue", "pink", or "violet" noise, mixed at a
                                      random SNR between `snr_low` and `snr_high`.
        return_background_clips (bool): Whether to return the segment of the background clip that was mixed with each
                                        foreground clip in the batch.
        return_background_clips_delay (Tuple(int)): The lower and upper bound of a random delay (in samples)
                                           to apply to the segment of each returned backgroud clip mixed
                                           with each foreground clip in the batch. This is primarily intended to
                                           simulate the drift between input and output channels
                                           in audio devices, which means that the mixed audio is never
                                           exactly aligned with the two source clips.
        shuffle (bool): Whether to shuffle the foreground clips before mixing (default: True)
        return_sequence_labels (bool): Whether to return sequence labels (i.e., frame-level labels) for each clip
                                       based on the start/end positions of the foreground clip.
        seed (int): A random seed

    Returns:
        generator: Returns a generator that yields batches of mixed foreground/background audio, labels, and the
                   background segments used for each audio clip (or None is the
                   `return_backgroun_clips` argument is False)
    """
    # Set random seed, if needed
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    # Check and Set start indices, if needed
    if not start_index:
        start_index = [0]*batch_size
    else:
        if min(start_index) < 0:
            raise ValueError("Error! At least one value of the `start_index` argument is <0. Check your inputs.")

    # Make dummy labels
    if not labels:
        labels = [0]*len(foreground_clips)

    if shuffle:
        p = np.random.permutation(len(foreground_clips))
        foreground_clips = np.array(foreground_clips)[p].tolist()
        start_index = np.array(start_index)[p].tolist()
        labels = np.array(labels)[p].tolist()
        if foreground_durations:
            foreground_durations = np.array(foreground_durations)[p].tolist()

    for i in range(0, len(foreground_clips), batch_size):
        # Load foreground clips/start indices and truncate as needed
        sr = 16000
        start_index_batch = start_index[i:i+batch_size]
        foreground_clips_batch = [read_audio(j) for j in foreground_clips[i:i+batch_size]]
        foreground_clips_batch = [j[0] if len(j.shape) > 1 else j for j in foreground_clips_batch]
        if foreground_durations:
            foreground_clips_batch = [truncate_clip(j, int(k*sr), foreground_truncate_strategy)
                                      for j, k in zip(foreground_clips_batch, foreground_durations[i:i+batch_size])]
        labels_batch = np.array(labels[i:i+batch_size])

        # Load background clips and pad/truncate as needed
        background_clips_batch = [read_audio(j) for j in random.sample(background_clips, batch_size)]
        background_clips_batch = [j[0] if len(j.shape) > 1 else j for j in background_clips_batch]
        background_clips_batch_delayed = []
        delay = np.random.randint(return_background_clips_delay[0], return_background_clips_delay[1] + 1)
        for ndx, background_clip in enumerate(background_clips_batch):
            if background_clip.shape[0] < (combined_size + delay):
                repeated = background_clip.repeat(
                    np.ceil((combined_size + delay)/background_clip.shape[0]).astype(np.int32)
                )
                background_clips_batch[ndx] = repeated[0:combined_size]
                background_clips_batch_delayed.append(repeated[0+delay:combined_size + delay].clone())
            elif background_clip.shape[0] > (combined_size + delay):
                r = np.random.randint(0, max(1, background_clip.shape[0] - combined_size - delay))
                background_clips_batch[ndx] = background_clip[r:r + combined_size]
                background_clips_batch_delayed.append(background_clip[r+delay:r + combined_size + delay].clone())

        # Mix clips at snr levels
        snrs_db = np.random.uniform(snr_low, snr_high, batch_size)
        mixed_clips = []
        sequence_labels = []
        for fg, bg, snr, start in zip(foreground_clips_batch, background_clips_batch,
                                      snrs_db, start_index_batch):
            if bg.shape[0] != combined_size:
                raise ValueError(bg.shape)
            mixed_clip = mix_clip(fg, bg, snr, start)
            sequence_labels.append(get_frame_labels(combined_size, start, start+fg.shape[0]))

            if np.random.random() < generated_noise_augmentation:
                noise_color = ["white", "pink", "blue", "brown", "violet"]
                noise_clip = acoustics.generator.noise(combined_size, color=np.random.choice(noise_color))
                noise_clip = torch.from_numpy(noise_clip/noise_clip.max())
                mixed_clip = mix_clip(mixed_clip, noise_clip, np.random.choice(snrs_db), 0)

            mixed_clips.append(mixed_clip)

        mixed_clips_batch = torch.vstack(mixed_clips)
        sequence_labels_batch = torch.from_numpy(np.vstack(sequence_labels))

        # Apply reverberation to the batch (from a single RIR file)
        if rirs:
            if np.random.random() <= rir_probability:
                rir_waveform, sr = torchaudio.load(random.choice(rirs))
                if rir_waveform.shape[0] > 1:
                    rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]
                mixed_clips_batch = reverberate(mixed_clips_batch, rir_waveform, rescale_amp="avg")

        # Apply volume augmentation
        if volume_augmentation:
            volume_levels = np.random.uniform(0.02, 1.0, mixed_clips_batch.shape[0])
            mixed_clips_batch = (volume_levels/mixed_clips_batch.max(dim=1)[0])[..., None]*mixed_clips_batch
        else:
            # Normalize clips only if max value is outside of [-1, 1]
            abs_max, _ = torch.max(
                torch.abs(mixed_clips_batch), dim=1, keepdim=True
            )
            mixed_clips_batch = mixed_clips_batch / abs_max.clamp(min=1.0)

        # Convert to 16-bit PCM audio
        mixed_clips_batch = (mixed_clips_batch.numpy()*32767).astype(np.int16)

        # Remove any clips that are silent (happens rarely when mixing/reverberating)
        error_index = torch.from_numpy(np.where(mixed_clips_batch.max(dim=1) != 0)[0])
        mixed_clips_batch = mixed_clips_batch[error_index]
        labels_batch = labels_batch[error_index]
        sequence_labels_batch = sequence_labels_batch[error_index]

        if not return_background_clips:
            yield mixed_clips_batch, labels_batch if not return_sequence_labels else sequence_labels_batch, None
        else:
            background_clips_batch_delayed = (torch.vstack(background_clips_batch_delayed).numpy()
                                              * 32767).astype(np.int16)[error_index]
            yield (mixed_clips_batch,
                   labels_batch if not return_sequence_labels else sequence_labels_batch,
                   background_clips_batch_delayed)


def get_frame_labels(combined_size, start, end, buffer=1):
    sequence_label = np.zeros(np.ceil((combined_size-12400)/1280).astype(int))
    frame_positions = np.arange(12400, combined_size, 1280)
    start_frame = np.argmin(abs(frame_positions - start))
    end_frame = np.argmin(abs(frame_positions - end))
    sequence_label[start_frame:start_frame+2] = 1
    sequence_label[end_frame-1:end_frame+1] = 1
    return sequence_label


def mix_clip(fg, bg, snr, start):
    fg_rms, bg_rms = fg.norm(p=2), bg.norm(p=2)
    snr = 10 ** (snr / 20)
    scale = snr * bg_rms / fg_rms
    bg[start:start + fg.shape[0]] = bg[start:start + fg.shape[0]] + scale*fg
    return bg / 2


def truncate_clip(x, max_size, method="truncate_start"):
    """
    Truncates and audio clip with the specified method

    Args:
        x (nd.array): An array of audio data
        max_size (int): The maximum size (in samples)
        method (str): Can be one of four options:
            - "truncate_start": Truncate the start of the clip
            - "truncate_end": Truncate the end of the clip
            - "truncate_both": Truncate both the start and end of the clip
            - "random": Randomly select a segment of the right size from the clip

    Returns:
        nd.array: The truncated audio data
    """
    if x.shape[0] > max_size:
        if method == "truncate_start":
            x = x[x.shape[0] - max_size:]
        if method == "truncate_end":
            x = x[0:max_size]
        if method == "truncate_both":
            n = int(np.ceil(x.shape[0] - max_size)/2)
            x = x[n:-n][0:max_size]
        if method == "random":
            rn = np.random.randint(0, x.shape[0] - max_size)
            x = x[rn:rn + max_size]

    return x


# Reverberation data augmentation function
def apply_reverb(x, rir_files):
    """
    Applies reverberation to the input audio clips

    Args:
        x (nd.array): A numpy array of shape (batch, audio_samples) containing the audio clips
        rir_files (Union[str, list]): Either a path to an RIR (room impulse response) file or a list
                                      of RIR files. If a list, one file will be randomly chosen
                                      to apply to `x`

    Returns:
        nd.array: The reverberated audio clips
    """
    if isinstance(rir_files, str):
        rir_waveform, sr = torchaudio.load(rir_files[0])
    elif isinstance(rir_files, list):
        rir_waveform, sr = torchaudio.load(random.choice(rir_files))

    # Apply reverberation to the batch (from a single RIR file)
    if rir_waveform.shape[0] > 1:
        rir_waveform = rir_waveform[random.randint(0, rir_waveform.shape[0]-1), :]
    reverbed = reverberate(torch.from_numpy(x), rir_waveform, rescale_amp="avg")

    return reverbed.numpy()


# Alternate data augmentation method using audiomentations library (https://pypi.org/project/audiomentations/)
def augment_clips(
        clip_paths: List[str],
        total_length: int,
        sr: int = 16000,
        batch_size: int = 128,
        augmentation_probabilities: dict = {
            "SevenBandParametricEQ": 0.25,
            "TanhDistortion": 0.25,
            "PitchShift": 0.25,
            "BandStopFilter": 0.25,
            "AddColoredNoise": 0.25,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5
        },
        background_clip_paths: List[str] = [],
        RIR_paths: List[str] = []
        ):
    """
    Applies audio augmentations to the specified audio clips, returning a generator that applies
    the augmentations in batches to support very large quantities of input audio files.

    The augmentations (and probabilities) are chosen from experience based on training openWakeWord models, as well
    as for the efficiency of the augmentation. The individual probabilities of each augmentation may be adjusted
    with the "augmentation_probabilities" argument.

    Args:
        clip_paths (List[str]) = The input audio files (as paths) to augment. Note that these should be shorter
                                 than the "total_length" argument, else they will be truncated.
        total_length (int): The total length of audio files (in samples) after augmentation. All input clips
                            will be left-padded with silence to reach this size, with between 0 and 200 ms
                            of other audio after the end of the original input clip.
        sr (int): The sample size of the input audio files
        batch_size (int): The number of audio files to augment at once.
        augmentation_probabilities (dict): The individual probabilities of each augmentation. If all probabilities
                                           are zero, the input audio files will simply be padded with silence. THe
                                           default values are:

                                            {
                                                "SevenBandParametricEQ": 0.25,
                                                "TanhDistortion": 0.25,
                                                "PitchShift": 0.25,
                                                "BandStopFilter": 0.25,
                                                "AddColoredNoise": 0.25,
                                                "AddBackgroundNoise": 0.75,
                                                "Gain": 1.0,
                                                "RIR": 0.5
                                            }

        background_clip_paths (List[str]) = The paths to background audio files to mix with the input files
        RIR_paths (List[str]) = The paths to room impulse response functions (RIRs) to convolve with the input files,
                                producing a version of the input clip with different acoustic characteristics.

    Returns:
        ndarray: A batch of augmented audio clips of size (batch_size, total_length)
    """
    # Define augmentations

    # First pass augmentations that can't be done as a batch
    augment1 = audiomentations.Compose([
        audiomentations.SevenBandParametricEQ(min_gain_db=-6, max_gain_db=6, p=augmentation_probabilities["SevenBandParametricEQ"]),
        audiomentations.TanhDistortion(
            min_distortion=0.0001,
            max_distortion=0.10,
            p=augmentation_probabilities["TanhDistortion"]
        ),
    ])

    # Augmentations that can be done as a batch
    if background_clip_paths != []:
        augment2 = torch_audiomentations.Compose([
            torch_audiomentations.PitchShift(
                min_transpose_semitones=-3,
                max_transpose_semitones=3,
                p=augmentation_probabilities["PitchShift"],
                sample_rate=16000,
                mode="per_batch"
            ),
            torch_audiomentations.BandStopFilter(p=augmentation_probabilities["BandStopFilter"], mode="per_batch"),
            torch_audiomentations.AddColoredNoise(
                min_snr_in_db=10, max_snr_in_db=30,
                min_f_decay=-1, max_f_decay=2, p=augmentation_probabilities["AddColoredNoise"],
                mode="per_batch"
            ),
            torch_audiomentations.AddBackgroundNoise(
                p=augmentation_probabilities["AddBackgroundNoise"],
                background_paths=background_clip_paths,
                min_snr_in_db=-10,
                max_snr_in_db=15,
                mode="per_batch"
            ),
            torch_audiomentations.Gain(max_gain_in_db=0, p=augmentation_probabilities["Gain"]),
        ])
    else:
        augment2 = torch_audiomentations.Compose([
            torch_audiomentations.PitchShift(
                min_transpose_semitones=-3,
                max_transpose_semitones=3,
                p=augmentation_probabilities["PitchShift"],
                sample_rate=16000,
                mode="per_batch"
            ),
            torch_audiomentations.BandStopFilter(p=augmentation_probabilities["BandStopFilter"], mode="per_batch"),
            torch_audiomentations.AddColoredNoise(
                min_snr_in_db=10, max_snr_in_db=30,
                min_f_decay=-1, max_f_decay=2, p=augmentation_probabilities["AddColoredNoise"],
                mode="per_batch"
            ),
            torch_audiomentations.Gain(max_gain_in_db=0, p=augmentation_probabilities["Gain"]),
        ])

    # Iterate through all clips and augment them
    for i in range(0, len(clip_paths), batch_size):
        batch = clip_paths[i:i+batch_size]
        augmented_clips = []
        for clip in batch:
            clip_data, clip_sr = torchaudio.load(clip)
            clip_data = clip_data[0]
            if clip_data.shape[0] > total_length:
                clip_data = clip_data[0:total_length]

            if clip_sr != sr:
                raise ValueError("Error! Clip does not have the correct sample rate!")

            clip_data = create_fixed_size_clip(clip_data, total_length, clip_sr)

            # Do first pass augmentations
            augmented_clips.append(torch.from_numpy(augment1(samples=clip_data, sample_rate=sr)))

        # Do second pass augmentations
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        augmented_batch = augment2(samples=torch.vstack(augmented_clips).unsqueeze(dim=1).to(device), sample_rate=sr).squeeze(axis=1)

        # Do reverberation
        if augmentation_probabilities["RIR"] >= np.random.random() and RIR_paths != []:
            rir_waveform, sr = torchaudio.load(random.choice(RIR_paths))
            augmented_batch = reverberate(augmented_batch.cpu(), rir_waveform, rescale_amp="avg")

        # yield batch of 16-bit PCM audio data
        yield (augmented_batch.cpu().numpy()*32767).astype(np.int16)


def create_fixed_size_clip(x, n_samples, sr=16000, start=None, end_jitter=.200):
    """
    Create a fixed-length clip of the specified size by padding an input clip with zeros
    Optionally specify the start/end position of the input clip, or let it be chosen randomly.

    Args:
        x (ndarray): The input audio to pad to a fixed size
        n_samples (int): The total number of samples for the fixed length clip
        sr (int): The sample rate of the audio
        start (int): The start position of the clip in the fixed length output, in samples (default: None)
        end_jitter (float): The time (in seconds) from the end of the fixed length output
                            that the input clip should end, if `start` is None.

    Returns:
        ndarray: A new array of audio data of the specified length
    """
    dat = np.zeros(n_samples)
    end_jitter = int(np.random.uniform(0, end_jitter)*sr)
    if start is None:
        start = max(0, n_samples - (int(len(x))+end_jitter))

    if len(x) > n_samples:
        if np.random.random() >= 0.5:
            dat = x[0:n_samples].numpy()
        else:
            dat = x[-n_samples:].numpy()
    else:
        dat[start:start+len(x)] = x

    return dat


# Load batches of data from mmaped numpy arrays
class mmap_batch_generator:
    """
    A generator class designed to dynamically build batches from mmaped numpy arrays.

    The generator will return tuples of (data, labels) with a batch size determined
    by the `n_per_class` initialization argument. When a mmaped numpy array has been
    fully interated over, it will restart at the zeroth index automatically.
    """
    def __init__(self,
                 data_files: dict,
                 label_files: dict = {},
                 batch_size: int = 128,
                 n_per_class: dict = {},
                 data_transform_funcs: dict = {},
                 label_transform_funcs: dict = {}
                 ):
        """
        Initialize the generator object

        Args:
            data_files (dict): A dictionary of labels (as keys) and on-disk numpy array paths (as values).
                               Keys should be integer strings representing class labels.
            label_files (dict): A dictionary where the keys are the class labels and the values are the per-example
                                labels. The values must be the same shape as the correponding numpy data arrays
                                from the `data_files` argument.
            batch_size (int): The number of samples per batch
            n_per_class (dict): A dictionary with integer string labels (as keys) and number of example per batch
                               (as values). If None (the default), batch sizes for each class will be
                               automatically calculated based on the the input dataframe shapes and transformation
                               functions.

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
        self.label_files = label_files
        self.n_per_class = n_per_class
        self.data_transform_funcs = data_transform_funcs
        self.label_transform_funcs = label_transform_funcs

        # Get array mmaps and store their shapes (but load files < 1 GB total size into memory)
        self.data = {label: np.load(fl, mmap_mode='r') for label, fl in data_files.items()}
        self.labels = {label: np.load(fl) for label, fl in label_files.items()}
        self.data_counter = {label: 0 for label in data_files.keys()}
        self.original_shapes = {label: self.data[label].shape for label in self.data.keys()}
        self.shapes = {label: self.data[label].shape for label in self.data.keys()}

        # # Update effective shape of mmap array based on user-provided transforms (currently broken)
        # for lbl, f in self.data_transform_funcs.items():
        #     dummy_data = np.random.random((1, self.original_shapes[lbl][1], self.original_shapes[lbl][2]))
        #     new_shape = f(dummy_data).shape
        #     self.shapes[lbl] = (new_shape[0]*self.original_shapes[lbl][0], new_shape[1], new_shape[2])

        # Calculate batch sizes, if the user didn't specify them
        scale_factor = 1
        if not self.n_per_class:
            self.n_per_class = {}
            for lbl, shape in self.shapes.items():
                dummy_data = np.random.random((10, self.shapes[lbl][1], self.shapes[lbl][2]))
                if self.data_transform_funcs.get(lbl, None):
                    scale_factor = self.data_transform_funcs.get(lbl, None)(dummy_data).shape[0]/10

                ratio = self.shapes[lbl][0]/sum([i[0] for i in self.shapes.values()])
                self.n_per_class[lbl] = max(1, int(int(batch_size*ratio)/scale_factor))

            # Get estimated batches per epoch, including the effect of any user-provided transforms
            batch_size = sum([val*scale_factor for val in self.n_per_class.values()])
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

                # Get data from mmaped file
                x = self.data[label][self.data_counter[label]:self.data_counter[label]+n]
                self.data_counter[label] += x.shape[0]

                # Transform data
                if self.data_transform_funcs and self.data_transform_funcs.get(label):
                    x = self.data_transform_funcs[label](x)

                # Make labels for data (following whatever the current shape of `x` is)
                if self.label_files.get(label, None):
                    y_batch = self.labels[label][self.data_counter[label]:self.data_counter[label]+n]
                else:
                    y_batch = [label]*x.shape[0]

                # Transform labels
                if self.label_transform_funcs and self.label_transform_funcs.get(label):
                    y_batch = self.label_transform_funcs[label](y_batch)

                # Add data to batch
                X.append(x)
                y.extend(y_batch)

            return np.vstack(X), np.array(y)


# Function to remove empty rows from the end of a mmap array
def trim_mmap(mmap_path):
    """
    Trims blank rows from the end of a mmaped numpy array by creates new mmap array without the blank rows.
    Note that a copy is created and disk usage will briefly double as the function runs.

    Args:
        mmap_path (str): The path to mmap array file to trim

    Returns:
        None
    """
    # Identify the last full row in the mmaped file
    mmap_file1 = np.load(mmap_path, mmap_mode='r')
    i = -1
    while np.all(mmap_file1[i, :, :] == 0):
        i -= 1

    N_new = mmap_file1.shape[0] + i + 1

    # Create new mmap_file and copy over data in batches
    output_file2 = mmap_path.strip(".npy") + "2.npy"
    mmap_file2 = open_memmap(output_file2, mode='w+', dtype=np.float32,
                             shape=(N_new, mmap_file1.shape[1], mmap_file1.shape[2]))

    for i in tqdm(range(0, mmap_file1.shape[0], 1024), total=mmap_file1.shape[0]//1024, desc="Trimming empty rows"):
        if i + 1024 > N_new:
            mmap_file2[i:N_new] = mmap_file1[i:N_new].copy()
            mmap_file2.flush()
        else:
            mmap_file2[i:i+1024] = mmap_file1[i:i+1024].copy()
            mmap_file2.flush()

    # Remove old mmaped file
    os.remove(mmap_path)

    # Rename new mmap file to match original
    os.rename(output_file2, mmap_path)


# Generate words that sound similar ("adversarial") to the input phrase using phoneme overlap
def generate_adversarial_texts(input_text: str, N: int, include_partial_phrase: float = 0, include_input_words: float = 0):
    """
    Generate adversarial words and phrases based on phoneme overlap.
    Currently only works for english texts.
    Note that homophones are excluded, as this wouldn't actually be an adversarial example for the input text.

    Args:
        input_text (str): The target text for adversarial phrases
        N (int): The total number of adversarial texts to return. Uses sampling,
                 so not all possible combinations will be included and some duplicates
                 may be present.
        include_partial_phrase (float): The probability of returning a number of words less than the input
                                        text (but always between 1 and the number of input words)
        include_input_words (float): The probability of including individual input words in the adversarial
                                     texts when the input text consists of multiple words. For example,
                                     if the `input_text` was "ok google", then setting this value > 0.0
                                     will allow for adversarial texts like "ok noodle", versus the word "ok"
                                     never being present in the adversarial texts.

    Returns:
        list: A list of strings corresponding to words and phrases that are phonetically similar (but not identical)
              to the input text.
    """
    # Get phonemes for english vowels (CMUDICT labels)
    vowel_phones = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX"]

    word_phones = []
    input_text_phones = [pronouncing.phones_for_word(i) for i in input_text.split()]

    # Download phonemizer model for OOV words, if needed
    if [] in input_text_phones:
        phonemizer_mdl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "en_us_cmudict_forward.pt")
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")):
            os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources"))
        if not os.path.exists(phonemizer_mdl_path):
            logging.warning("Downloading phonemizer model from DeepPhonemizer library...")
            import requests
            file_url = "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt"
            r = requests.get(file_url, stream=True)
            with open(phonemizer_mdl_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2048):
                    if chunk:
                        f.write(chunk)

        # Create phonemizer object
        from dp.phonemizer import Phonemizer
        phonemizer = Phonemizer.from_checkpoint(phonemizer_mdl_path)

    for phones, word in zip(input_text_phones, input_text.split()):
        if phones != []:
            word_phones.extend(phones)
        elif phones == []:
            logging.warning(f"The word '{word}' was not found in the pronunciation dictionary! "
                            "Using the DeepPhonemizer library to predict the phonemes.")
            phones = phonemizer(word, lang='en_us')
            logging.warning(f"Phones for '{word}': {phones}")
            word_phones.append(re.sub(r"[\]|\[]", "", re.sub(r"\]\[", " ", phones)))
        elif isinstance(phones[0], list):
            logging.warning(f"There are multiple pronunciations for the word '{word}'.")
            word_phones.append(phones[0])

    # add all possible lexical stresses to vowels
    word_phones = [re.sub('|'.join(vowel_phones), lambda x: str(x.group(0)) + '[0|1|2]', re.sub(r'\d+', '', i)) for i in word_phones]

    adversarial_phrases = []
    for phones, word in zip(word_phones, input_text.split()):
        query_exps = []
        phones = phones.split()
        adversarial_words = []
        if len(phones) <= 2:
            query_exps.append(" ".join(phones))
        else:
            query_exps.extend(phoneme_replacement(phones, max_replace=max(0, len(phones)-2), replace_char="(.){1,3}"))

        for query in query_exps:
            matches = pronouncing.search(query)
            matches_phones = [pronouncing.phones_for_word(i)[0] for i in matches]
            allowed_matches = [i for i, j in zip(matches, matches_phones) if j != phones]
            adversarial_words.extend([i for i in allowed_matches if word.lower() != i])

        if adversarial_words != []:
            adversarial_phrases.append(adversarial_words)

    # Build combinations for final output
    adversarial_texts = []
    for i in range(N):
        txts = []
        for j, k in zip(adversarial_phrases, input_text.split()):
            if np.random.random() > (1 - include_input_words):
                txts.append(k)
            else:
                txts.append(np.random.choice(j))

        if include_partial_phrase is not None and len(input_text.split()) > 1 and np.random.random() <= include_partial_phrase:
            n_words = np.random.randint(1, len(input_text.split())+1)
            adversarial_texts.append(" ".join(np.random.choice(txts, size=n_words, replace=False)))
        else:
            adversarial_texts.append(" ".join(txts))

    # Remove any exact matches to input phrase
    adversarial_texts = [i for i in adversarial_texts if i != input_text]

    return adversarial_texts


def phoneme_replacement(input_chars, max_replace, replace_char='"(.){1,3}"'):
    results = []
    chars = list(input_chars)

    # iterate over the number of characters to replace (1 to max_replace)
    for r in range(1, max_replace+1):
        # get all combinations for a fixed r
        comb = itertools.combinations(range(len(chars)), r)
        for indices in comb:
            chars_copy = chars.copy()
            for i in indices:
                chars_copy[i] = replace_char
            results.append(' '.join(chars_copy))

    return results
