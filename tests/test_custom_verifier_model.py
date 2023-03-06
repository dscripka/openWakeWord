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
import openwakeword
import os
import numpy as np
import scipy.io.wavfile
import tempfile
import pytest


# Tests
class TestModels:
    def test_train_verifier_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Make random negative data for verifier model training
            scipy.io.wavfile.write(os.path.join(tmp_dir, "negative_reference.wav"),
                                   16000, np.random.randint(-1000, 1000, 16000*4).astype(np.int16))

            # Load random clips
            reference_clips = [os.path.join("tests", "data", "hey_mycroft_test.wav")]
            negative_clips = [os.path.join(tmp_dir, "negative_reference.wav")]

            # Check for error message when no positive examples are found
            with pytest.raises(ValueError):
                openwakeword.train_custom_verifier(
                    positive_reference_clips=reference_clips,
                    negative_reference_clips=negative_clips,
                    output_path=os.path.join(tmp_dir, 'verifier_model.pkl'),
                    model_name="alexa"
                )

            # Train verifier model on the reference clips
            openwakeword.train_custom_verifier(
                positive_reference_clips=reference_clips,
                negative_reference_clips=negative_clips,
                output_path=os.path.join(tmp_dir, 'verifier_model.pkl'),
                model_name="hey_mycroft"
            )

            # Train verifier model on the reference clips, using full path of model file
            openwakeword.train_custom_verifier(
                positive_reference_clips=reference_clips,
                negative_reference_clips=negative_clips,
                output_path=os.path.join(tmp_dir, 'verifier_model.pkl'),
                model_name=os.path.join("openwakeword", "resources", "models", "hey_mycroft_v0.1.onnx")
            )

            with pytest.raises(ValueError):
                # Load model with verifier model incorrectly to catch ValueError
                owwModel = openwakeword.Model(
                    wakeword_model_paths=[os.path.join("openwakeword", "resources", "models", "hey_mycroft_v0.1.onnx")],
                    custom_verifier_models={"bad_key": os.path.join(tmp_dir, "verifier_model.pkl")},
                    custom_verifier_threshold=0.3,
                )

            # Load model with verifier model incorrectly to catch ValueError
            owwModel = openwakeword.Model(
                wakeword_model_paths=[os.path.join("openwakeword", "resources", "models", "hey_mycroft_v0.1.onnx")],
                custom_verifier_models={"hey_mycroft_v0.1": os.path.join(tmp_dir, "verifier_model.pkl")},
                custom_verifier_threshold=0.3,
            )

            # Prediction on random data
            owwModel.predict_clip(reference_clips[0])
