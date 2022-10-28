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
import numpy as np
import onnxruntime as ort
from openwakeword.utils import AudioFeatures

import statistics
import wave
import os
from collections import deque, defaultdict
from typing import List
from functools import partial
import time
import pprint
from typing import List, Union

class Model():
    """
    The main model class for openWakeWord. Creates a model object with the shared audio pre-processer
    and for arbitrarily many custom wake word/wake phrase models.
    """
    def __init__(self, wakeword_model_paths: List[str], **kwargs):
        """
        Initialize the openWakeWord model object.

        Args:
            wakeword_model_paths (List[str]): A list of paths of ONNX models to load into the openWakeWord model object
        """

        # Initialize the ONNX models and store them
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = 1
        sessionOptions.intra_op_num_threads = 1

        # Create attributes to store models and metadat
        self.models = {}
        self.model_inputs = {}
        self.model_input_names = {}
        for mdl_path in wakeword_model_paths:
            mdl_name = mdl_path.split(os.path.sep)[-1].strip(".onnx")
            self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions, providers=["CPUExecutionProvider"])
            self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
            self.model_input_names[mdl_name] = self.models[mdl_name].get_inputs()[0].name

        # Create buffer to store frame predictios
        self.prediction_buffer = defaultdict(partial(deque, maxlen=5))

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(**kwargs)


    def predict(self, x: Union[np.ndarray, List], median_smooth: bool=False, timing: bool=False):
        """Predict with all of the wakeword models on the input audio frames
        
        Args:
            x (Union[ndarray, List]): The input audio data to predict on with the models. Must be 1280 samples of 16khz, 16-bit audio data.
            median_smooth (bool): Whether to apply a running median smooth of the last three predictions before returning a score.
                                  Can reduce false-positive productions at the cost of a lower true-positive rate.
            timing (bool): Whether to print timing information of the models. Can be useful to debug and assess how efficiently models
                           are running the current hardware.

        Returns:
            dict: A dictionary of scores between 0 and 1 for each model, where 0 indicates no wake-word/wake-phrase detected
        """
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

            # Run model
            predictions[mdl] = self.models[mdl].run(
                                    None,
                                    {input_name: self.preprocessor.get_features(self.model_inputs[mdl])}
                                )[0][0][0]

            # Update prediction buffer
            self.prediction_buffer[mdl].append(predictions[mdl])

            # (Optionally) Smooth model predictions with simple median calculate of last three predictions
            if median_smooth:
                predictions[mdl] = statistics.median(list(self.prediction_buffer[mdl])[-3:])

            if timing:
                model_end = time.time()
                timing_dict["models"][mdl] = model_end - model_start

        if timing:
            pp = pprint.PrettyPrinter().pprint(timing_dict)
            return predictions
        else:
            return predictions


    def predict_clip(self, clip: str, padding: bool=True, **kwargs):
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