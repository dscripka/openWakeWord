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
import numpy as np
import pathlib
from collections import deque
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Queue
import time
import logging
from tqdm import tqdm
import openwakeword
from numpy.lib.format import open_memmap
from typing import Union, List, Callable, Deque
import requests


# Base class for computing audio features using Google's speech_embedding
# model (https://tfhub.dev/google/speech_embedding/1)
class AudioFeatures():
    """
    A class for creating audio features from audio data, including melspectograms and Google's
    `speech_embedding` features.
    """
    def __init__(self,
                 melspec_model_path: str = "",
                 embedding_model_path: str = "",
                 sr: int = 16000,
                 ncpu: int = 1,
                 inference_framework: str = "onnx",
                 device: str = 'cpu'
                 ):
        """
        Initialize the AudioFeatures object.

        Args:
            melspec_model_path (str): The path to the model for computing melspectograms from audio data
            embedding_model_path (str): The path to the model for Google's `speech_embedding` model
            sr (int): The sample rate of the audio (default: 16000 khz)
            ncpu (int): The number of CPUs to use when computing melspectrograms and audio features (default: 1)
            inference_framework (str): The inference framework to use when for model prediction. Options are
                                       "tflite" or "onnx". The default is "tflite" as this results in better
                                       efficiency on common platforms (x86, ARM64), but in some deployment
                                       scenarios ONNX models may be preferable.
            device (str): The device to use when running the models, either "cpu" or "gpu" (default is "cpu".)
                          Note that depending on the inference framework selected and system configuration,
                          this setting may not have an effect. For example, to use a GPU with the ONNX
                          framework the appropriate onnxruntime package must be installed.
        """
        # Initialize the models with the appropriate framework
        if inference_framework == "onnx":
            try:
                import onnxruntime as ort
            except ImportError:
                raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

            if melspec_model_path == "":
                melspec_model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "resources", "models", "melspectrogram.onnx")
            if embedding_model_path == "":
                embedding_model_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "resources", "models", "embedding_model.onnx")

            if ".tflite" in melspec_model_path or ".tflite" in embedding_model_path:
                raise ValueError("The onnx inference framework is selected, but tflite models were provided!")

            # Initialize ONNX options
            sessionOptions = ort.SessionOptions()
            sessionOptions.inter_op_num_threads = ncpu
            sessionOptions.intra_op_num_threads = ncpu

            # Melspectrogram model
            self.melspec_model = ort.InferenceSession(melspec_model_path, sess_options=sessionOptions,
                                                      providers=["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"])
            self.onnx_execution_provider = self.melspec_model.get_providers()[0]
            self.melspec_model_predict = lambda x: self.melspec_model.run(None, {'input': x})

            # Audio embedding model
            self.embedding_model = ort.InferenceSession(embedding_model_path, sess_options=sessionOptions,
                                                        providers=["CUDAExecutionProvider"] if device == "gpu"
                                                        else ["CPUExecutionProvider"])
            self.embedding_model_predict = lambda x: self.embedding_model.run(None, {'input_1': x})[0].squeeze()

        elif inference_framework == "tflite":
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                raise ValueError("Tried to import the TFLite runtime, but it was not found."
                                 "Please install it using `pip install tflite-runtime`")

            if melspec_model_path == "":
                melspec_model_path = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                                  "resources", "models", "melspectrogram.tflite")
            if embedding_model_path == "":
                embedding_model_path = os.path.join(pathlib.Path(__file__).parent.resolve(),
                                                    "resources", "models", "embedding_model.tflite")

            if ".onnx" in melspec_model_path or ".onnx" in embedding_model_path:
                raise ValueError("The tflite inference framework is selected, but onnx models were provided!")

            # Melspectrogram model
            self.melspec_model = tflite.Interpreter(model_path=melspec_model_path, num_threads=ncpu)
            self.melspec_model.resize_tensor_input(0, [1, 1280], strict=True)  # initialize with fixed input size
            self.melspec_model.allocate_tensors()

            melspec_input_index = self.melspec_model.get_input_details()[0]['index']
            melspec_output_index = self.melspec_model.get_output_details()[0]['index']

            self._tflite_current_melspec_input_size = 1280

            def tflite_melspec_predict(x):
                if x.shape[1] != 1280:
                    self.melspec_model.resize_tensor_input(0, [1, x.shape[1]], strict=True)  # initialize with fixed input size
                    self.melspec_model.allocate_tensors()
                    self._tflite_current_melspec_input_size = x.shape[1]
                elif self._tflite_current_melspec_input_size != 1280:
                    self.melspec_model.resize_tensor_input(0, [1, 1280], strict=True)  # initialize with fixed input size
                    self.melspec_model.allocate_tensors()
                    self._tflite_current_melspec_input_size = 1280

                self.melspec_model.set_tensor(melspec_input_index, x)
                self.melspec_model.invoke()
                return self.melspec_model.get_tensor(melspec_output_index)

            self.melspec_model_predict = tflite_melspec_predict

            # Audio embedding model
            self.embedding_model = tflite.Interpreter(model_path=embedding_model_path, num_threads=ncpu)
            self.embedding_model.allocate_tensors()

            embedding_input_index = self.embedding_model.get_input_details()[0]['index']
            embedding_output_index = self.embedding_model.get_output_details()[0]['index']

            self._tflite_current_embedding_batch_size = 1

            def tflite_embedding_predict(x):
                if x.shape[0] != 1:
                    self.embedding_model.resize_tensor_input(0, [x.shape[0], 76, 32, 1], strict=True)  # initialize with fixed input size
                    self.embedding_model.allocate_tensors()
                    self._tflite_current_embedding_batch_size = x.shape[0]
                elif self._tflite_current_embedding_batch_size != 1:
                    self.embedding_model.resize_tensor_input(0, [1, 76, 32, 1], strict=True)  # initialize with fixed input size
                    self.embedding_model.allocate_tensors()
                    self._tflite_current_embedding_batch_size = x.shape[0]

                self.embedding_model.set_tensor(embedding_input_index, x)
                self.embedding_model.invoke()
                return self.embedding_model.get_tensor(embedding_output_index).squeeze()

            self.embedding_model_predict = tflite_embedding_predict

        # Create databuffers with empty/random data
        self.raw_data_buffer: Deque = deque(maxlen=sr*10)
        self.melspectrogram_buffer = np.ones((76, 32))  # n_frames x num_features
        self.melspectrogram_max_len = 10*97  # 97 is the number of frames in 1 second of 16hz audio
        self.accumulated_samples = 0  # the samples added to the buffer since the audio preprocessor was last called
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))
        self.feature_buffer_max_len = 120  # ~10 seconds of feature buffer history

    def reset(self):
        """Reset the internal buffers"""
        self.raw_data_buffer.clear()
        self.melspectrogram_buffer = np.ones((76, 32))
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))

    def _get_melspectrogram(self, x: Union[np.ndarray, List], melspec_transform: Callable = lambda x: x/10 + 2):
        """
        Function to compute the mel-spectrogram of the provided audio samples.

        Args:
            x (Union[np.ndarray, List]): The input audio data to compute the melspectrogram from
            melspec_transform (Callable): A function to transform the computed melspectrogram. Defaults to a transform
                                          that makes the ONNX melspectrogram model closer to the native Tensorflow
                                          implementation from Google (https://tfhub.dev/google/speech_embedding/1).

        Return:
            np.ndarray: The computed melspectrogram of the input audio data
        """
        # Get input data and adjust type/shape as needed
        x = np.array(x).astype(np.int16) if isinstance(x, list) else x
        if x.dtype != np.int16:
            raise ValueError("Input data must be 16-bit integers (i.e., 16-bit PCM audio)."
                             f"You provided {x.dtype} data.")
        x = x[None, ] if len(x.shape) < 2 else x
        x = x.astype(np.float32) if x.dtype != np.float32 else x

        # Get melspectrogram
        outputs = self.melspec_model_predict(x)
        spec = np.squeeze(outputs[0])

        # Arbitrary transform of melspectrogram
        spec = melspec_transform(spec)

        return spec

    def _get_embeddings_from_melspec(self, melspec):
        """
        Computes the Google `speech_embedding` features from a melspectrogram input

        Args:
            melspec (np.ndarray): The input melspectrogram

        Returns:
            np.ndarray: The computed audio features/embeddings
        """
        if melspec.shape[0] != 1:
            melspec = melspec[None, ]
        embedding = self.embedding_model_predict(melspec)
        return embedding

    def _get_embeddings(self, x: np.ndarray, window_size: int = 76, step_size: int = 8, **kwargs):
        """Function to compute the embeddings of the provide audio samples."""
        spec = self._get_melspectrogram(x, **kwargs)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size:  # truncate short windows
                windows.append(window)

        batch = np.expand_dims(np.array(windows), axis=-1).astype(np.float32)
        embedding = self.embedding_model_predict(batch)
        return embedding

    def get_embedding_shape(self, audio_length: float, sr: int = 16000):
        """Function that determines the size of the output embedding array for a given audio clip length (in seconds)"""
        x = (np.random.uniform(-1, 1, int(audio_length*sr))*32767).astype(np.int16)
        return self._get_embeddings(x).shape

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
        mel_bins = 32  # fixed by melspectrogram model
        melspecs = np.empty((x.shape[0], n_frames, mel_bins), dtype=np.float32)
        for i in range(0, max(batch_size, x.shape[0]), batch_size):
            batch = x[i:i+batch_size]

            if "CUDA" in self.onnx_execution_provider:
                result = self._get_melspectrogram(batch)

            elif pool:
                chunksize = batch.shape[0]//ncpu if batch.shape[0] >= ncpu else 1
                result = np.array(pool.map(self._get_melspectrogram,
                                           batch, chunksize=chunksize))

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

        # Calculate array sizes and make batches
        n_frames = (x.shape[1] - 76)//8 + 1
        embedding_dim = 96  # fixed by embedding model
        embeddings = np.empty((x.shape[0], n_frames, embedding_dim), dtype=np.float32)

        batch = []
        ndcs = []
        for ndx, melspec in enumerate(x):
            window_size = 76
            for i in range(0, melspec.shape[0], 8):
                window = melspec[i:i+window_size]
                if window.shape[0] == window_size:  # ignore windows that are too short (truncates end of clip)
                    batch.append(window)
            ndcs.append(ndx)

            if len(batch) >= batch_size or ndx+1 == x.shape[0]:
                batch = np.array(batch).astype(np.float32)
                if "CUDA" in self.onnx_execution_provider:
                    result = self.embedding_model_predict(batch)

                elif pool:
                    chunksize = batch.shape[0]//ncpu if batch.shape[0] >= ncpu else 1
                    result = np.array(pool.map(self._get_embeddings_from_melspec,
                                      batch, chunksize=chunksize))

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

    def _streaming_melspectrogram(self, n_samples):
        """Note! There seem to be some slight numerical issues depending on the underlying audio data
        such that the streaming method is not exactly the same as when the melspectrogram of the entire
        clip is calculated. It's unclear if this difference is significant and will impact model performance.
        In particular padding with 0 or very small values seems to demonstrate the differences well.
        """
        if len(self.raw_data_buffer) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")

        self.melspectrogram_buffer = np.vstack(
            (self.melspectrogram_buffer, self._get_melspectrogram(list(self.raw_data_buffer)[-n_samples-160*3:]))
        )

        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def _buffer_raw_data(self, x):
        """
        Adds raw audio data to the input buffer
        """
        self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

    def _streaming_features(self, x):
        # Add raw audio data to buffer, temporarily storing extra frames if not an even number of 80 ms chunks
        processed_samples = 0

        if self.raw_data_remainder.shape[0] != 0:
            x = np.concatenate((self.raw_data_remainder, x))
            self.raw_data_remainder = np.empty(0)

        if self.accumulated_samples + x.shape[0] >= 1280:
            remainder = (self.accumulated_samples + x.shape[0]) % 1280
            if remainder != 0:
                x_even_chunks = x[0:-remainder]
                self._buffer_raw_data(x_even_chunks)
                self.accumulated_samples += len(x_even_chunks)
                self.raw_data_remainder = x[-remainder:]
            elif remainder == 0:
                self._buffer_raw_data(x)
                self.accumulated_samples += x.shape[0]
                self.raw_data_remainder = np.empty(0)
        else:
            self.accumulated_samples += x.shape[0]
            self._buffer_raw_data(x)

        # Only calculate melspectrogram once minimum samples are accumulated
        if self.accumulated_samples >= 1280 and self.accumulated_samples % 1280 == 0:
            self._streaming_melspectrogram(self.accumulated_samples)

            # Calculate new audio embeddings/features based on update melspectrograms
            for i in np.arange(self.accumulated_samples//1280-1, -1, -1):
                ndx = -8*i
                ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
                x = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
                if x.shape[1] == 76:
                    self.feature_buffer = np.vstack((self.feature_buffer,
                                                    self.embedding_model_predict(x)))

            # Reset raw data buffer counter
            processed_samples = self.accumulated_samples
            self.accumulated_samples = 0

        if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
            self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

        return processed_samples if processed_samples != 0 else self.accumulated_samples

    def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1):
        if start_ndx != -1:
            end_ndx = start_ndx + int(n_feature_frames) \
                if start_ndx + n_feature_frames != 0 else len(self.feature_buffer)
            return self.feature_buffer[start_ndx:end_ndx, :][None, ].astype(np.float32)
        else:
            return self.feature_buffer[int(-1*n_feature_frames):, :][None, ].astype(np.float32)

    def __call__(self, x):
        return self._streaming_features(x)


# Bulk prediction function
def bulk_predict(
                 file_paths: List[str],
                 wakeword_models: List[str],
                 prediction_function: str = 'predict_clip',
                 ncpu: int = 1,
                 inference_framework: str = "tflite",
                 **kwargs
                 ):
    """
    Bulk predict on the provided input files in parallel using multiprocessing using the specified model.

    Args:
        input_paths (List[str]): The list of input file to predict
        wakeword_models (List[str])): The paths to the wakeword model files
        prediction_function (str): The name of the method used to predict on the input audio files
                                   (default is the `predict_clip` method)
        ncpu (int): How many processes to create (up to max of available CPUs)
        inference_framework (str): The inference framework to use when for model prediction. Options are
                                    "tflite" or "onnx". The default is "tflite" as this results in better
                                    efficiency on common platforms (x86, ARM64), but in some deployment
                                    scenarios ONNX models may be preferable.
        kwargs (dict): Any other keyword arguments to pass to the model initialization or
                       specified prediction function

    Returns:
        dict: A dictionary containing the predictions for each file, with the filepath as the key
    """

    # Create openWakeWord model objects
    n_batches = max(1, len(file_paths)//ncpu)
    remainder = len(file_paths) % ncpu
    chunks = [file_paths[i:i+n_batches] for i in range(0, max(1, len(file_paths)-remainder), n_batches)]
    for i in range(1, remainder+1):
        chunks[i-1].append(file_paths[-1*i])

    # Create jobs
    ps = []
    mdls = []
    q: Queue = Queue()
    for chunk in chunks:
        filtered_kwargs = {key: value for key, value in kwargs.items()
                           if key in openwakeword.Model.__init__.__code__.co_varnames}
        oww = openwakeword.Model(
            wakeword_models=wakeword_models,
            inference_framework=inference_framework,
            **filtered_kwargs
        )
        mdls.append(oww)

        def f(clips):
            results = []
            for clip in clips:
                func = getattr(mdls[-1], prediction_function)
                filtered_kwargs = {key: value for key, value in kwargs.items()
                                   if key in func.__code__.co_varnames}
                results.append({clip: func(clip, **filtered_kwargs)})
            q.put(results)

        ps.append(Process(target=f, args=(chunk,)))

    # Submit jobs
    for p in ps:
        p.start()

    # Collection results
    results = []
    for p in ps:
        while q.empty():
            time.sleep(0.01)
        results.extend(q.get())

    # Consolidate results and return
    return {list(i.keys())[0]: list(i.values())[0] for i in results}


def compute_features_from_generator(generator, n_total, clip_duration, output_file, device="cpu", ncpu=1):
    """
    Computes audio features from a generator that produces Numpy arrays of shape (batch_size, samples)
    containing 16-bit PCM audio data.

    Args:
        generator (Generator): The generator that process the arrays of audio data
        n_total (int): The total number of rows (audio clips) that the generator will produce.
                       Ideally this is precise, but it can be approximate as well as the output
                       .npy file will be automatically trimmed to remove empty values.
        clip_duration (float): The duration (in samples) of the audio produced by the generator
        output_file (str): The output file (.npy) containing the audio features. Note that this file
                           will be written to using memmap arrays, so it can be substantially larger
                           than the available system memory.
        device (str): The device ("cpu" or "gpu") to use for computing features.
        ncpu (int): The number of cores to use when process the audio features (if computing on CPU)

    Returns:
        None
    """
    # Function specific imports
    from openwakeword.data import trim_mmap

    # Create audio features object
    F = AudioFeatures(device=device)

    # Determine the output shape and create output file
    n_feature_cols = F.get_embedding_shape(clip_duration/16000)
    output_shape = (n_total, n_feature_cols[0], n_feature_cols[1])
    fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_shape)

    # Get batch size by pulling one value from the generator and store features
    row_counter = 0
    audio_data = next(generator)
    batch_size = audio_data.shape[0]

    if batch_size > n_total:
        raise ValueError(f"The value of 'n_total' ({n_total}) is less than the batch size ({batch_size})."
                         " Please increase 'n_total' to be >= batch size.")

    features = F.embed_clips(audio_data, batch_size=batch_size)
    fp[row_counter:row_counter+features.shape[0], :, :] = features
    row_counter += features.shape[0]
    fp.flush()

    # Compute features and add data to output file
    for audio_data in tqdm(generator, total=n_total//batch_size, desc="Computing features"):
        if row_counter >= n_total:
            break

        features = F.embed_clips(audio_data, batch_size=batch_size, ncpu=ncpu)
        if row_counter + features.shape[0] > n_total:
            features = features[0:n_total-row_counter]

        fp[row_counter:row_counter+features.shape[0], :, :] = features
        row_counter += features.shape[0]
        fp.flush()

    # Trip empty rows from the mmapped array
    trim_mmap(output_file)


# Function to download files from a URL with a progress bar
def download_file(url, target_directory, file_size=None):
    """A simple function to download a file from a URL with a progress bar using only the requests library"""
    local_filename = url.split('/')[-1]

    with requests.get(url, stream=True) as r:
        if file_size is not None:
            progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"{local_filename}")
        else:
            total_size = int(r.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{local_filename}")

        with open(os.path.join(target_directory, local_filename), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()


# Function to download models from GitHub release assets
def download_models(
        model_names: List[str] = [],
        target_directory: str = os.path.join(pathlib.Path(__file__).parent.resolve(), "resources", "models")
        ):
    """
    Download the specified models from the release assets in the openWakeWord GitHub repository.
    Uses the official urls in the MODELS dictionary in openwakeword/__init__.py.

    Args:
        model_names (List[str]): The names of the models to download (e.g., hey_jarvis_v0.1). Both ONNX and
                                 tflite models will be downloaded. If not provided (the default),
                                 the latest versions of all models will be downloaded.
        target_directory (str): The directory to save the models to. Defaults to the install location
                                of openWakeWord (i.e., the `resources/models` directory).
    Returns:
        None
    """
    if not isinstance(model_names, list):
        raise ValueError("The model_names argument must be a list of strings")

    # Always download melspectrogram and embedding models, if they don't already exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    for feature_model in openwakeword.FEATURE_MODELS.values():
        if not os.path.exists(os.path.join(target_directory, feature_model["download_url"].split("/")[-1])):
            download_file(feature_model["download_url"], target_directory)
            download_file(feature_model["download_url"].replace(".tflite", ".onnx"), target_directory)

    # Always download VAD models, if they don't already exist
    for vad_model in openwakeword.VAD_MODELS.values():
        if not os.path.exists(os.path.join(target_directory, vad_model["download_url"].split("/")[-1])):
            download_file(vad_model["download_url"], target_directory)

    # Get all model urls
    official_model_urls = [i["download_url"] for i in openwakeword.MODELS.values()]
    official_model_names = [i["download_url"].split("/")[-1] for i in openwakeword.MODELS.values()]

    if model_names != []:
        for model_name in model_names:
            url = [i for i, j in zip(official_model_urls, official_model_names) if model_name in j]
            if url != []:
                if not os.path.exists(os.path.join(target_directory, url[0].split("/")[-1])):
                    download_file(url[0], target_directory)
                    download_file(url[0].replace(".tflite", ".onnx"), target_directory)
    else:
        for official_model_url in official_model_urls:
            if not os.path.exists(os.path.join(target_directory, official_model_url.split("/")[-1])):
                download_file(official_model_url, target_directory)
                download_file(official_model_url.replace(".tflite", ".onnx"), target_directory)


# Handle deprecated arguments and naming (thanks to https://stackoverflow.com/a/74564394)
def re_arg(kwarg_map):
    def decorator(func):
        def wrapped(*args, **kwargs):
            new_kwargs = {}
            for k, v in kwargs.items():
                if k in kwarg_map:
                    logging.warning(f"DEPRECATION: keyword argument '{k}' is no longer valid and "
                                    f"will be removed in future releases. Use '{kwarg_map[k]}' instead.")
                new_kwargs[kwarg_map.get(k, k)] = v
            return func(*args, **new_kwargs)
        return wrapped
    return decorator
