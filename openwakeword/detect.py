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
import os
from openwakeword.utils import AudioFeatures
from typing import List

class Model():
    def __init__(self, wakeword_model_paths: List[str], input_sizes: List[int], **kwargs):
        # Initialize the ONNX models and store them
        sessionOptions = ort.SessionOptions()
        sessionOptions.inter_op_num_threads = 1
        sessionOptions.intra_op_num_threads = 1

        self.models = {}
        self.model_inputs = {}
        for size, mdl_path in zip(input_sizes, wakeword_model_paths):
            mdl_name = mdl_path.split(os.path.sep)[-1].strip(".onnx")
            self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions)
            self.model_inputs[mdl_name] = size

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(**kwargs)

    def predict(self, x):
        """Predict with all of the wakeword models on the input audio frames"""
        self.preprocessor(x)
        predictions = {}
        for mdl in self.models.keys():
            input_name = self.models[mdl].get_inputs()[0].name
            predictions[mdl] = self.models[mdl].run(
                                    None,
                                    {input_name: self.preprocessor.get_features(self.model_inputs[mdl])}
                                )[0][0][0]
        return predictions

        