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
import openwakeword
from openwakeword.utils import AudioFeatures

import wave
import os
from collections import deque, defaultdict
from functools import partial
import time
from typing import List, Union, DefaultDict, Dict


# Define main model class
class Model():
    """
    The main model class for openWakeWord. Creates a model object with the shared audio pre-processer
    and for arbitrarily many custom wake word/wake phrase models.
    """
    def __init__(
            self,
            wakeword_model_paths: List[str] = [],
            class_mapping_dicts: List[dict] = [],
            **kwargs
            ):
        """Initialize the openWakeWord model object.

        Args:
            wakeword_model_paths (List[str]): A list of paths of ONNX models to load into the openWakeWord model object.
                                              If not provided, will load all of the pre-trained models.
            class_mapping_dicts (List[dict]): A list of dictionaries with integer to string class mappings for
                                              each model in the `wakeword_model_paths` arguments
                                              (e.g., {"0": "class_1", "1": "class_2"})
        """

        # Initialize the ONNX models and store them
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = 1
        sessionOptions.intra_op_num_threads = 1

        # Get model paths for pre-trained models if user doesn't provide models to load
        if wakeword_model_paths == []:
            wakeword_model_paths = openwakeword.get_pretrained_model_paths()
            wakeword_model_names = list(openwakeword.models.keys())
        else:
            wakeword_model_names = [os.path.basename(i.strip(".onnx")) for i in wakeword_model_paths]

        # Create attributes to store models and metadat
        self.models = {}
        self.model_inputs = {}
        self.model_outputs = {}
        self.class_mapping = {}
        self.model_input_names = {}
        for mdl_path, mdl_name in zip(wakeword_model_paths, wakeword_model_names):
            self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions,
                                                         providers=["CPUExecutionProvider"])
            self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
            self.model_outputs[mdl_name] = self.models[mdl_name].get_outputs()[0].shape[1]
            if class_mapping_dicts and class_mapping_dicts[wakeword_model_paths.index(mdl_path)].get(mdl_name, None):
                self.class_mapping[mdl_name] = class_mapping_dicts[wakeword_model_paths.index(mdl_path)]
            elif openwakeword.model_class_mappings.get(mdl_name, None):
                self.class_mapping[mdl_name] = openwakeword.model_class_mappings[mdl_name]
            else:
                self.class_mapping[mdl_name] = {str(i): str(i) for i in range(0, self.model_outputs[mdl_name])}
            self.model_input_names[mdl_name] = self.models[mdl_name].get_inputs()[0].name

        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(partial(deque, maxlen=30))

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(**kwargs)

    def get_parent_model_from_label(self, label):
        """Gets the parent model associated with a given prediction label"""
        parent_model = ""
        for mdl in self.class_mapping.keys():
            if label in self.class_mapping[mdl].values():
                parent_model = mdl
            elif label in self.class_mapping.keys() and label == mdl:
                parent_model = mdl

        return parent_model

    def reset(self):
        """Reset the prediction buffer"""
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))

    def predict(self, x: Union[np.ndarray, List], patience: dict = {}, threshold: dict = {}, timing: bool = False):
        """Predict with all of the wakeword models on the input audio frames

        Args:
            x (Union[ndarray, List]): The input audio data to predict on with the models. Must be 1280
                                      samples of 16khz, 16-bit audio data.
            patience (dict): How many consecutive frames (of 1280 samples or 80 ms) above the threshold that must
                             be observed before the current frame will be returned as non-zero.
                             Must be provided as an a dictionary where the keys are the
                             model names and the values are the number of frames. Can reduce false-positive
                             detections at the cost of a lower true-positive rate.
                             By default, this behavior is disabled.
            threshold (dict): The threshold values to use when the `patience` behavior is enabled.
                              Must be provided as an a dictionary where the keys are the
                              model names and the values are the thresholds.
            timing (bool): Whether to return timing information of the models. Can be useful to debug and
                           assess how efficiently models are running on the current hardware.

        Returns:
            dict: A dictionary of scores between 0 and 1 for each model, where 0 indicates no
                  wake-word/wake-phrase detected. If the `timing` argument is true, returns a
                  tuple of dicts containing model predictions and timing information, respectively.
        """
        # Get audio features
        if timing:
            timing_dict: Dict[str, Dict] = {}
            timing_dict["models"] = {}
            feature_start = time.time()

        self.preprocessor(x)

        if timing:
            feature_end = time.time()
            timing_dict["models"]["preprocessor"] = feature_end - feature_start

        # Get predictions from model(s)
        predictions = {}
        for mdl in self.models.keys():
            input_name = self.model_input_names[mdl]

            if timing:
                model_start = time.time()

            # Run model
            prediction = self.models[mdl].run(
                                    None,
                                    {input_name: self.preprocessor.get_features(self.model_inputs[mdl])}
                                )
            if self.model_outputs[mdl] == 1:
                predictions[mdl] = prediction[0][0][0]
            else:
                for int_label, cls in self.class_mapping[mdl].items():
                    predictions[cls] = prediction[0][0][int(int_label)]

            # Update prediction buffer, and zero predictions for first 5 frames during model initialization
            for cls in predictions.keys():
                if len(self.prediction_buffer[cls]) < 5:
                    predictions[cls] = 0.0
                self.prediction_buffer[cls].append(predictions[cls])

            # Get timing information
            if timing:
                model_end = time.time()
                timing_dict["models"][mdl] = model_end - model_start

        # Update scores based on thresholds or patience arguments
        if patience != {}:
            if threshold == {}:
                raise ValueError("Error! When using the `patience` argument, threshold "
                                 "values must be provided via the `threshold` argument!")
            for mdl in predictions.keys():
                parent_model = self.get_parent_model_from_label(mdl)
                if parent_model in patience.keys():
                    scores = np.array(self.prediction_buffer[mdl])[-patience[parent_model]:]
                    if (scores >= threshold[parent_model]).sum() < patience[parent_model]:
                        predictions[mdl] = 0.0

        if timing:
            return predictions, timing_dict
        else:
            return predictions

    def predict_clip(self, clip: str, padding: int = 1, **kwargs):
        """Predict on an full audio clip, simulating streaming prediction.
        The input clip must bit a 16-bit, 16 khz, single-channel WAV file.

        Args:
            clip (str): The path to a 16-bit PCM, 16 khz, single-channel WAV file
            padding (int): How many seconds of silence to pad the start/end of the clip with
                            to make sure that short clips can be processed correctly (default: 1)
            kwargs: Any keyword arguments to pass to the class `predict` method

        Returns:
            list: A list containing the frame-level prediction dictionaries for the audio clip
        """
        # Load audio clip as 16-bit PCM data
        with wave.open(clip, mode='rb') as f:
            # Load WAV clip frames
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            if padding:
                data = np.concatenate(
                    (
                        np.zeros(16000*padding).astype(np.int16),
                        data,
                        np.zeros(16000*padding).astype(np.int16)
                    )
                )

        # Iterate through clip, getting predictions
        predictions = []
        step_size = 1280
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions.append(self.predict(data[i:i+step_size], **kwargs))

        return predictions
