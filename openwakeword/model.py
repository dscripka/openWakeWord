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
import onnxruntime as ort
import numpy as np
import wave
import os
from openwakeword.utils import AudioFeatures
from typing import List
from functools import reduce
import time

class Model():
    def __init__(self, wakeword_model_paths: List[str], input_sizes: List[int], **kwargs):
        # Initialize the ONNX models and store them
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = 1
        sessionOptions.intra_op_num_threads = 1

        self.models = {}
        self.model_inputs = {}
        self.model_input_names = {}
        for size, mdl_path in zip(input_sizes, wakeword_model_paths):
            mdl_name = mdl_path.split(os.path.sep)[-1].strip(".onnx")
            self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions, providers=["CPUExecutionProvider"])
            self.model_inputs[mdl_name] = size
            self.model_input_names[mdl_name] = self.models[mdl_name].get_inputs()[0].name

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(**kwargs)

    def predict_clip(self, clip, padding=True, **kwargs):
        """Predict on an full audio clip, simulating streaming prediction.
        The input clip must bit a 16-bit, 16 khz, single-channel WAV file.

        Args:
            clip (str): The path to a 16-bit PCM, 16 khz, single-channel WAV file
            padding (bool): Whether to pad the clip on either side with 1 second of silence
                            to make sure that short clips can be processed correctly (default: True)
            kwargs: Any keyword arguments to pass to the class `predict` method
        
        Returns:
            list: A list containing the frame-level prediction dictionaries for the audio clip
        """
        # Load audio clip as 16-bit PCM data
        with wave.open(clip, mode='rb') as f:
            # Load WAV clip frames
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            if padding:
                data = np.concatenate((np.zeros(32000), data, np.zeros(32000)))

        # Iterate through clip, getting predictions
        predictions = []
        step_size = 1280
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions.append(self.predict(data[i:i+step_size], **kwargs))

        return predictions

    def predict(self, x, verify_rounds=None, timing=False):
        """Predict with all of the wakeword models on the input audio frames"""
        # Get audio features
        if timing:
            timing_dict = {}
            timing_dict["models"] = {}
            feature_start = time.time()
        
        self.preprocessor(x)

        if timing:
            feature_end = time.time()
            timing_dict["preprocessor"] = feature_end - feature_start

        # Get predictions from model(s)
        predictions = {}
        for mdl in self.models.keys():
            input_name = self.model_input_names[mdl]

            if timing:
                model_start = time.time()

            predictions[mdl] = self.models[mdl].run(
                                    None,
                                    {input_name: self.preprocessor.get_features(self.model_inputs[mdl])}
                                )[0][0][0]

            if verify_rounds is not None and predictions[mdl] >= 0.5: # make threshold user configurable?
                # Use TTA (test time augmentation) to re-check positive predictions
                tta_predictions = []
                offset = 200
                for round in range(1, verify_rounds+1):
                    x = list(self.preprocessor.raw_data_buffer)[-32000-offset*round:-offset*round] # arbitrary chunk size, adjust as needed?
                    tta_predictions.append(self.models[mdl].run(
                                    None,
                                    {input_name: self.preprocessor._get_embeddings(x)[None,]}
                                )[0][0][0])
                    predictions[mdl] = reduce(lambda x, y: x*y, tta_predictions)

            if timing:
                model_end = time.time()
                timing_dict["models"][mdl] = model_end - model_start

        if timing:
            return predictions, timing_dict
        else:
            return predictions

        