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
import pathlib
from collections import deque

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

        # Create databuffers
        self.raw_data_buffer = deque(maxlen=sr*10)
        self.melspectrogram_buffer = np.zeros((0,32)) #n_frames x num_features
        self.melspectrogram_max_len = 10*97 # 97 is the number of frames in 1 second of 16hz audio
        self.feature_buffer = np.zeros((32,96))
        self.feature_buffer_max_len = 120 # ~10 seconds of feature buffer history
        
    def _get_melspectrogram(self, x):
        """Function to compute the mel-spectrogram of the provided audio samples."""
        x = np.array(x)[None,] if isinstance(x, list) else x[None,]
        x = x.astype(np.float32) if x.dtype!=np.float32 else x 
        outputs = self.melspec_model.run(None, {'input': x})
        spec = np.squeeze(outputs[0])
        spec = spec/10 + 2  # Arbitrary adjustment to make result closer to original Google speech_embedding model
        
        return spec
        
    def _get_embeddings(self, x, window_size=76, step_size = 8):
        """Function to compute the embeddings of the provide audio samples."""
        spec = self._get_melspectrogram(x)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i+window_size]
            if window.shape[0] == window_size: # truncate short windows
                windows.append(window)
        
        batch = np.expand_dims(np.array(windows), axis=-1)
        embedding = self.embedding_model.run(None, {'input_1': batch})[0].squeeze()
        return embedding
        
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