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
import sys
import logging
import numpy as np
from pathlib import Path
import collections
import pytest
import platform
import pickle
import tempfile
import mock
import wave

# Download models needed for tests
openwakeword.utils.download_models()


# Tests
class TestModels:
    def test_load_models_by_path(self):
        # Load model with defaults
        owwModel = openwakeword.Model(wakeword_models=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                      ], inference_framework="onnx")

        owwModel = openwakeword.Model(wakeword_models=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.tflite")
                                      ], inference_framework="tflite")

        # Prediction on random data
        owwModel.predict(np.random.randint(-1000, 1000, 1280).astype(np.int16))

    def test_predict_with_different_frame_sizes(self):
        # Test with binary model
        owwModel1 = openwakeword.Model(wakeword_models=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                      ], inference_framework="onnx")

        owwModel2 = openwakeword.Model(wakeword_models=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                      ], inference_framework="onnx")

        # Prediction on random data with integer multiples of standard chunk size (1280 samples)
        predictions1 = owwModel1.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1280)
        predictions2 = owwModel2.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1280*2)
        np.testing.assert_approx_equal(max([i['alexa_v0.1'] for i in predictions1]), max([i['alexa_v0.1'] for i in predictions2]), 5)

        # Prediction on data with a chunk size not an integer multiple of 1280
        predictions1 = owwModel1.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1024)
        predictions2 = owwModel2.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1024*2)
        np.testing.assert_approx_equal(max([i['alexa_v0.1'] for i in predictions1]), max([i['alexa_v0.1'] for i in predictions2]), 5)

        # Test with multiclass model
        owwModel1 = openwakeword.Model(wakeword_models=["timer"], inference_framework="onnx")
        owwModel2 = openwakeword.Model(wakeword_models=["timer"], inference_framework="onnx")

        # Prediction on random data with integer multiples of standard chunk size (1280 samples)
        predictions1 = owwModel1.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1280)
        predictions2 = owwModel2.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1280*2)
        assert abs(max([i['1_minute_timer'] for i in predictions1]) - max([i['1_minute_timer'] for i in predictions2])) < 0.00001

        # Prediction on data with a chunk size not an integer multiple of 1280
        predictions1 = owwModel1.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1024)
        predictions2 = owwModel2.predict_clip(os.path.join("tests", "data", "alexa_test.wav"), chunk_size=1024*2)
        assert abs(max([i['1_minute_timer'] for i in predictions1]) - max([i['1_minute_timer'] for i in predictions2])) < 0.00001

    def test_exception_handling_for_inference_framework(self):
        with mock.patch.dict(sys.modules, {'onnxruntime': None}):
            with pytest.raises(ValueError):
                openwakeword.Model(wakeword_models=[
                                                os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                            ], inference_framework="onnx")

        with mock.patch.dict(sys.modules, {'tflite_runtime': None}):
            openwakeword.Model(wakeword_models=[
                                            os.path.join("openwakeword", "resources", "models", "alexa_v0.1.tflite")
                                        ], inference_framework="tflite")

    def test_predict_with_custom_verifier_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Train custom verifier model with random data
            verifier_model = openwakeword.custom_verifier_model.train_verifier_model(np.random.random((2, 1536)), np.array([0, 1]))
            pickle.dump(verifier_model, open(os.path.join(tmp_dir, "test_verifier.pkl"), "wb"))

            # Load model with verifier
            owwModel = openwakeword.Model(
                wakeword_models=[os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")],
                inference_framework="onnx",
                custom_verifier_models={"alexa_v0.1": os.path.join(tmp_dir, "test_verifier.pkl")},
                custom_verifier_threshold=0.0
            )

            owwModel.predict(np.random.randint(-1000, 1000, 1280).astype(np.int16))

    def test_load_pretrained_model_by_name(self):
        # Load model with defaults
        owwModel = openwakeword.Model(wakeword_models=["alexa", "hey mycroft"], inference_framework="onnx")

        owwModel = openwakeword.Model(wakeword_models=["alexa", "hey mycroft"], inference_framework="tflite")

        # Prediction on random data
        owwModel.predict(np.random.randint(-1000, 1000, 1280).astype(np.int16))

    def test_custom_model_label_mapping_dict(self):
        # Load model with model path
        owwModel = openwakeword.Model(wakeword_models=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                      ],
                                      class_mapping_dicts=[{"alexa_v0.1": {"0": "positive"}}],
                                      inference_framework="onnx"
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
            try:
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
            except ImportError:
                logging.warning("Attemped to test Speex noise cancelling functionality, but the 'speexdsp_ns' library was not installed!"
                                " If you want these tests to be run, install this library as shown in the openwakeword documentation."
                                )
                assert 1 == 1

    def test_models_with_debounce(self):
        # Load model with defaults
        owwModel = openwakeword.Model()

        # Predict with chunks of 1280 with and without debounce
        predictions = owwModel.predict_clip(os.path.join("tests", "data", "alexa_test.wav"),
                                            debounce_time=0, threshold={"alexa_v0.1": 0.5})
        scores = np.array([i['alexa'] for i in predictions])

        predictions = owwModel.predict_clip(os.path.join("tests", "data", "alexa_test.wav"),
                                            debounce_time=1.25, threshold={"alexa": 0.5})
        scores_with_debounce = np.array([i['alexa'] for i in predictions])
        print(scores, scores_with_debounce)
        assert (scores >= 0.5).sum() > 1
        assert (scores_with_debounce >= 0.5).sum() == 1

    def test_model_reset(self):
        # Load the model
        owwModel = openwakeword.Model()

        # Get test clip and load it
        clip = os.path.join("tests", "data", "alexa_test.wav")
        with wave.open(clip, mode='rb') as f:
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

        # Predict frame by frame
        for i in range(0, len(data), 1280):
            prediction = owwModel.predict(data[i:i+1280])
            if prediction['alexa'] > 0.5:
                break

        # Assert that next prediction is still > 0.5
        prediction = owwModel.predict(data[i:i+1280])
        assert prediction['alexa'] > 0.5

        # Reset the model
        owwModel.reset()

        # Assert that next prediction is < 0.5
        prediction = owwModel.predict(data[i:i+1280])
        assert prediction['alexa'] < 0.5

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
        owwModel = openwakeword.Model(wakeword_models=[
                                        os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")
                                      ], inference_framework="onnx")

        clip = os.path.join("tests", "data", "alexa_test.wav")
        features = owwModel._get_positive_prediction_frames(clip)
        assert list(features.values())[0].shape[0] > 0
