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
from pathlib import Path
import collections
import pytest
import platform


# Tests
class TestModels:
    def test_load_models_by_path(self):
        # Load model with defaults
        owwModel = openwakeword.Model(wakeword_model_paths=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                      ])

        # Prediction on random data
        owwModel.predict(np.random.randint(-1000, 1000, 1280).astype(np.int16))

        # Prediction on random data with different chunk size
        owwModel.predict(np.random.randint(-1000, 1000, 1280*2).astype(np.int16))

    def test_custom_model_label_mapping_dict(self):
        # Load model with model path
        owwModel = openwakeword.Model(wakeword_model_paths=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                      ],
                                      class_mapping_dicts=[{"alexa_v0.1": {"0": "positive"}}]
                                      )

        # Prediction on random data
        owwModel.predict(np.random.randint(-1000, 1000, 1280).astype(np.int16))

    def test_models(self):
        # Load model with defaults
        owwModel = openwakeword.Model()

        # Get clips for each model (assumes that test clips will have the model name in the filename)
        test_dict = {}
        for mdl_name in owwModel.models.keys():
            all_clips = [str(i) for i in Path(os.path.join("tests", "data")).glob("*.wav")]
            test_dict[mdl_name] = [i for i in all_clips if mdl_name in i]

        # Predict
        for model, clips in test_dict.items():
            for clip in clips:
                # Get predictions for reach frame in the clip
                predictions = owwModel.predict_clip(clip)
                owwModel.reset()  # reset after each clip to ensure independent results

                # Make predictions dictionary flatter
                predictions_flat = collections.defaultdict(list)
                [predictions_flat[key].append(i[key]) for i in predictions for key in i.keys()]

            # Check scores against default threshold (0.5)
            for key in predictions_flat.keys():
                if key in clip:
                    assert max(predictions_flat[key]) >= 0.5
                else:
                    assert max(predictions_flat[key]) < 0.5

    def test_models_with_speex_noise_cancellation(self):
        # Skip test on Windows for now
        if platform.system() == "Windows":
            assert 1 == 1
        else:
            # Load model with defaults
            owwModel = openwakeword.Model(enable_speex_noise_suppression=True)

            # Get clips for each model (assumes that test clips will have the model name in the filename)
            test_dict = {}
            for mdl_name in owwModel.models.keys():
                all_clips = [str(i) for i in Path(os.path.join("tests", "data")).glob("*.wav")]
                test_dict[mdl_name] = [i for i in all_clips if mdl_name in i]

            # Predict
            for model, clips in test_dict.items():
                for clip in clips:
                    # Get predictions for reach frame in the clip
                    predictions = owwModel.predict_clip(clip)
                    owwModel.reset()  # reset after each clip to ensure independent results

                    # Make predictions dictionary flatter
                    predictions_flat = collections.defaultdict(list)
                    [predictions_flat[key].append(i[key]) for i in predictions for key in i.keys()]

                # Check scores against default threshold (0.5)
                for key in predictions_flat.keys():
                    if key in clip:
                        assert max(predictions_flat[key]) >= 0.5
                    else:
                        assert max(predictions_flat[key]) < 0.5

    def test_models_with_vad(self):
        # Load model with defaults
        owwModel = openwakeword.Model(vad_threshold=0.5)

        # Get clips for each model (assumes that test clips will have the model name in the filename)
        test_dict = {}
        for mdl_name in owwModel.models.keys():
            all_clips = [str(i) for i in Path(os.path.join("tests", "data")).glob("*.wav")]
            test_dict[mdl_name] = [i for i in all_clips if mdl_name in i]

        # Predict
        for model, clips in test_dict.items():
            for clip in clips:
                # Get predictions for reach frame in the clip
                predictions = owwModel.predict_clip(clip)
                owwModel.reset()  # reset after each clip to ensure independent results

                # Make predictions dictionary flatter
                predictions_flat = collections.defaultdict(list)
                [predictions_flat[key].append(i[key]) for i in predictions for key in i.keys()]

            # Check scores against default threshold (0.5)
            for key in predictions_flat.keys():
                if key in clip:
                    assert max(predictions_flat[key]) >= 0.5
                else:
                    assert max(predictions_flat[key]) < 0.5

    def test_predict_clip_with_array(self):
        # Load model with defaults
        owwModel = openwakeword.Model()

        # Make random array and predict
        dat = np.random.random(16000)
        predictions = owwModel.predict_clip(dat)
        assert isinstance(predictions[0], dict)

    def test_models_with_timing(self):
        # Load model with defaults
        owwModel = openwakeword.Model(vad_threshold=0.5)

        owwModel.predict(np.zeros(1280).astype(np.int16), timing=True)

    def test_prediction_with_patience(self):
        owwModel = openwakeword.Model()
        target_model_name = list(owwModel.models.keys())[0]

        with pytest.raises(ValueError):
            owwModel.predict(
                np.zeros(1280),
                patience={target_model_name: 5}
                )

        owwModel.predict(
            np.zeros(1280),
            patience={target_model_name: 5},
            threshold={target_model_name: 0.5}
            )

    def test_get_parent_model_from_prediction_label(self):
        owwModel = openwakeword.Model()
        target_model_name = list(owwModel.models.keys())[0]
        owwModel.get_parent_model_from_label(target_model_name)

    def test_get_positive_prediction_frames(self):
        owwModel = openwakeword.Model()

        # Get a clip to use for the test
        clip = [str(i) for i in Path(os.path.join("tests", "data")).glob("*.wav")][0]
        features = owwModel._get_positive_prediction_frames(clip)
        assert list(features.values())[0].shape[0] > 0
