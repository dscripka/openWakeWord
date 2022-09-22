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


# Imports
import os
import onnxruntime as ort
import numpy as np
import scipy.io.wavfile
import pathlib
from collections import deque
from multiprocessing.pool import ThreadPool

# Base class for computing audio features using Google's speech_embedding model (https://tfhub.dev/google/speech_embedding/1)
class AudioFeatures():
    def __init__(self,
            melspec_onnx_model_path=os.path.join(pathlib.Path(__file__).parent.resolve(), "resources", "models", "melspectrogram.onnx"),
            embedding_onnx_model_path=os.path.join(pathlib.Path(__file__).parent.resolve(), "resources", "models", "embedding_model.onnx"),
            sr=16000,
            ncpu=1
        ):
        # Initialize the ONNX models
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = ncpu
        sessionOptions.intra_op_num_threads = ncpu
        self.melspec_model = ort.InferenceSession(melspec_onnx_model_path, sess_options=sessionOptions)
        self.embedding_model = ort.InferenceSession(embedding_onnx_model_path, sess_options=sessionOptions)
        self.onnx_execution_provider = self.melspec_model.get_providers()[0]

        # Create databuffers
        self.raw_data_buffer = deque(maxlen=sr*10)
        self.melspectrogram_buffer = np.zeros((0,32)) #n_frames x num_features
        self.melspectrogram_max_len = 10*97 # 97 is the number of frames in 1 second of 16hz audio
        self.feature_buffer = np.zeros((32,96))
        self.feature_buffer_max_len = 120 # ~10 seconds of feature buffer history
        
    def _get_melspectrogram(self, x, melspec_transform = lambda x: x/10 + 2):
        """Function to compute the mel-spectrogram of the provided audio samples."""
        x = x[None,] if len(x.shape) < 2 else x
        x = x.astype(np.float32) if x.dtype!=np.float32 else x 
        outputs = self.melspec_model.run(None, {'input': x})
        spec = np.squeeze(outputs[0])
        if melspec_transform:
            spec = melspec_transform(spec)  # Arbitrary adjustment to make result closer to original Google speech_embedding model
        
        return spec

    def _get_embeddings_from_melspec(self, melspec):
        if melspec.shape[0] != 1:
            melspec = melspec[None,]
        embedding = self.embedding_model.run(None, {'input_1': melspec})[0].squeeze()
        return embedding

    def _get_embeddings(self, x, window_size=76, step_size = 8, **kwargs):
        """Function to compute the embeddings of the provide audio samples."""
        spec = self._get_melspectrogram(x, **kwargs)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size: # truncate short windows
                windows.append(window)
        
        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        embedding = self.embedding_model.run(None, {'input_1': batch})[0].squeeze()
        return embedding

    def _get_melspectrogram_batch(self, x, batch_size=128, ncpu=1):
        """
        Compute the melspectrogram of the input audio samples in batches.
        
        Note that the optimal performance will depend in the interaction between the device, 
        batch size, and ncpu (if a CPU device is used). The user is encouraged
        to experiment with different values of these parameters to identify
        which combination is best for their data, as often differences of 1-4x are seen.
        
        Args:
            x (ndarray): A numpy array of 16 khz input audio data in shape (N, samples).
                        Assumes that all of the audio data is the same length (same number of samples).
            batch_size (int): The batch size to use when computing the melspectrogram
            ncpu (int): The number of CPUs to use when computing the melspectrogram. This argument has
                        no effect if the underlying model is executing on a GPU.
                    
        Returns:
            ndarray: A numpy array of shape (N, frames, melbins) containing the melspectrogram of
                    all N input audio examples
        """

        # Prepare ThreadPool object, if needed for multithreading
        pool = None
        if "CPU" in self.onnx_execution_provider:
            pool = ThreadPool(processes=ncpu)
        
        # Make batches
        n_frames = int(np.ceil(x.shape[1]/160-3))
        mel_bins = 32 # fixed by melspectrogram model
        melspecs = np.empty((x.shape[0], n_frames, mel_bins), dtype=np.float32)
        for i in range(0, max(batch_size, x.shape[0]), batch_size):
            batch = x[i:i+batch_size]
        
            if "CUDA" in self.onnx_execution_provider:
                result = self._get_melspectrogram(batch)

            elif pool:
                result = np.array(pool.map(self._get_melspectrogram, batch, chunksize=batch.shape[0]//ncpu))
                
            melspecs[i:i+batch_size, :, :] = result.squeeze()
        
        # Cleanup ThreadPool
        if pool:
            pool.close()
            
        return melspecs

    def _get_embeddings_batch(self, x, batch_size=128, ncpu=1):
        """
        Compute the embeddings of the input melspectrograms in batches.
        
        Note that the optimal performance will depend in the interaction between the device, 
        batch size, and ncpu (if a CPU device is used). The user is encouraged
        to experiment with different values of these parameters to identify
        which combination is best for their data, as often differences of 1-4x are seen.
        
        Args:
            x (ndarray): A numpy array of melspectrograms of shape (N, frames, melbins).
                        Assumes that all of the melspectrograms have the same shape.
            batch_size (int): The batch size to use when computing the embeddings
            ncpu (int): The number of CPUs to use when computing the embeddings. This argument has
                        no effect if the underlying model is executing on a GPU.
                    
        Returns:
            ndarray: A numpy array of shape (N, frames, embedding_dim) containing the embeddings of
                    all N input melspectrograms
        """

        # Ensure input is the correct shape
        if x.shape[1] < 76:
            raise ValueError("Embedding model requires the input melspectrograms to have at least 76 frames")

        # Prepare ThreadPool object, if needed for multithreading
        pool = None
        if "CPU" in self.onnx_execution_provider:
            pool = ThreadPool(processes=ncpu)
            
        # Calcuate array sizes and make batches
        n_frames = (x.shape[1] - 76)//8 + 1
        embedding_dim = 96 # fixed by embedding model
        embeddings = np.empty((x.shape[0], n_frames, embedding_dim), dtype=np.float32)
        
        batch = []
        ndcs = []
        for ndx, melspec in enumerate(x):
            window_size = 76
            for i in range(0, melspec.shape[0], 8):
                window = melspec[i:i+window_size]
                if window.shape[0] == window_size: # ignore windows that are too short (truncates end of clip)
                    batch.append(window)
            ndcs.append(ndx)
                    
            if len(batch) >= batch_size or ndx+1 == x.shape[0]:
                batch = np.array(batch).astype(np.float32)
                if "CUDA" in self.onnx_execution_provider:
                    result = self.embedding_model.run(None, {'input_1': batch})[0].squeeze()

                elif pool:
                    result = np.array(pool.map(self._get_embeddings_from_melspec, batch, chunksize=batch.shape[0]//ncpu))
                
                for j, ndx2 in zip(range(0, result.shape[0], n_frames), ndcs):
                    embeddings[ndx2, :, :] = result[j:j+n_frames]
                
                batch = []
                ndcs = []
        
        # Cleanup ThreadPool
        if pool:
            pool.close()
        
        return embeddings

    def embed_clips(self, x, batch_size=128, ncpu=1):
        """
        Compute the embeddings of the input audio clips in batches.
        
        Note that the optimal performance will depend in the interaction between the device, 
        batch size, and ncpu (if a CPU device is used). The user is encouraged
        to experiment with different values of these parameters to identify
        which combination is best for their data, as often differences of 1-4x are seen.
        
        Args:
            x (ndarray): A numpy array of 16 khz input audio data in shape (N, samples).
                        Assumes that all of the audio data is the same length (same number of samples).
            batch_size (int): The batch size to use when computing the embeddings
            ncpu (int): The number of CPUs to use when computing the melspectrogram. This argument has
                        no effect if the underlying model is executing on a GPU.
                    
        Returns:
            ndarray: A numpy array of shape (N, frames, embedding_dim) containing the embeddings of
                    all N input audio clips
        """
        
        # Compute melspectrograms
        melspecs = self._get_melspectrogram_batch(x, batch_size=batch_size, ncpu=ncpu)
        
        # Compute embeddings from melspectrograms
        embeddings = self._get_embeddings_batch(melspecs[:, :, :, None], batch_size=batch_size, ncpu=ncpu)
        
        return embeddings

    def _streaming_melspectrogram(self, x):
        """Note! There seem to be some slight numerical issues depending on the underlying audio data
        such that the streaming method is not exactly the same as when the melspectrogram of the entire
        clip is calculated. It's unclear if this difference is significant and will impact model performance.
        In particular padding with 0 or very small values seems to demonstrate the differences well.
        """
        if len(x) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")
        self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)
        self.melspectrogram_buffer = np.vstack(
            (self.melspectrogram_buffer, self._get_melspectrogram(list(self.raw_data_buffer)[-len(x)-160*3:]))
        )
        
        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]
    
    def _streaming_features(self, x):
        if len(x) != 1280:
            raise ValueError(f"You must provide input samples in frames of 1280 samples @ 1600khz. Received a frame of {len(x)} samples.")
        self._streaming_melspectrogram(x)
        x = self.melspectrogram_buffer[-76:].astype(np.float32)[None,:,:,None]
        if x.shape[1] == 76:
            self.feature_buffer = np.vstack((self.feature_buffer, self.embedding_model.run(None, {'input_1': x})[0].squeeze()))
            
        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

    def get_features(self, n_feature_frames=16):
        return self.feature_buffer[-n_feature_frames:, :][None,].astype(np.float32)
            
    def __call__(self, x):
        self._streaming_features(x)

# Dato I/O utils

def load_wav_clips(files, clip_size=32000):
    """
    Loads the specified audio files and shapes them into an array of N by `clip_size`,
    where N is determined by the length of the audio files and `clip_size` at run time.
    
    Clips longer than `clip size` are truncated and extended into the N+1 row.
    Clips shorter than `clip_size` are combined with the previous or next clip
    (except for the last clip in `files`, which is ignored if it is too short.)
    
    Args:
        files (List[str]): A list of filepaths
        clip_size (int): The number of samples (of 16khz audio) for all of the rows in the array
        
    Returns:
        ndarray: A N by `clip_size` array with the audio data
    """
    
    audio_data = [scipy.io.wavfile.read(i)[1] for i in files]
    N = sum([i.shape[0] for i in audio_data])//clip_size
    
    X = np.empty((N, clip_size))
    previous_row_remainder = None
    cnt = 0
    for row in audio_data:
        row = np.hstack((previous_row_remainder, row))
        while row.shape[0] >= clip_size:
            X[cnt, :] = row[0:clip_size]
            row = row[clip_size:]
            cnt += 1
        
        previous_row_remainder = row if row.size > 0 else None
            
    return X

class mmap_batch_generator:
    """
    A generator class designed to dynamically build batches from mmaped numpy arrays.

    The generator will return tuples of (data, labels) with a batch size determined
    by the `n_per_class` initialization argument. When a mmaped numpy array has been
    fully interated over, it will restart at the zeroth index automatically.
    """
    def __init__(self, data_files, n_per_class):
        """
        Initialize the generator object

        Args:
            data_files (dict): A dictionary of labels (as keys) and on-disk numpy array paths (as values).
                               Keys should be integer strings representing class labels.
            n_per_class (dict): A dictionary with integer string labels (as keys) and number of example per batch
                               (as values).
        """
        # inputs
        self.data_files = data_files
        self.n_per_class = n_per_class
        
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
                X.append(x)
                y.extend([int(label)]*x.shape[0])

            return np.vstack(X), np.array(y)