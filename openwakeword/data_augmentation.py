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
import random
from typing import List
from tqdm import tqdm
import numpy as np
from speechbrain.dataio.dataio import read_audio
import torch

# Data augmentation utility functions

def mix_clips_batch(
        foreground_clips: List[str],
        background_clips: List[str],
        combined_size: int,
        batch_size: int=32,
        snr_low: float=0,
        snr_high: float=0,
        start_index: List[int]=[],
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
    """
    
    # Set start indices, if needed
    if not start_index:
        start_index = [0]*batch_size
    
    mixed_clips = np.empty((len(foreground_clips), combined_size))
    for i in tqdm(list(range(0, len(foreground_clips), batch_size))):
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

        mixed_clips[i:i+mixed_clips_batch.shape[0], :] = mixed_clips_batch.numpy()
    
    return mixed_clips
