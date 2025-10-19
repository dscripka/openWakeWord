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
import pytest
import time


# Tests
class TestSelfConfirm:
    def test_self_confirm_basic_functionality(self):
        """Test that self_confirm returns properly formatted predictions_dict"""
        # Initialize model with self_confirm enabled
        owwModel = openwakeword.Model(
            wakeword_models=[os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")],
            inference_framework="onnx",
            self_confirm=True
        )

        # Feed in ~10 seconds of random data to fill the audio buffer (10 seconds * 16000 Hz = 160000 samples)
        # Process in chunks of 1280 samples (80 ms)
        chunk_size = 1280
        n_samples = 160000  # 10 seconds of audio

        for i in range(0, n_samples, chunk_size):
            random_audio = np.random.randint(-1000, 1000, chunk_size).astype(np.int16)
            owwModel.predict(random_audio)

        # Run the self-confirm function
        predictions_dict = owwModel.self_confirm(last_n_seconds=1.5)

        # Verify predictions_dict is properly formed
        assert isinstance(predictions_dict, dict), "predictions_dict should be a dictionary"

        # Check that it has the expected model keys
        expected_models = list(owwModel.models.keys())
        assert len(predictions_dict) == len(expected_models), f"predictions_dict should have {len(expected_models)} key(s)"

        for model_name in expected_models:
            assert model_name in predictions_dict, f"predictions_dict should contain key '{model_name}'"

            # Check that values are between 0 and 1
            score = predictions_dict[model_name]
            assert isinstance(score, (float, np.floating)), f"Score for {model_name} should be a float"
            assert 0 <= score <= 1, f"Score for {model_name} should be between 0 and 1, got {score}"

    def test_self_confirm_with_multiple_models(self):
        """Test self_confirm with multiple models loaded"""
        owwModel = openwakeword.Model(
            wakeword_models=["alexa", "hey mycroft"],
            inference_framework="onnx",
            self_confirm=True
        )

        # Feed in ~10 seconds of random data
        chunk_size = 1280
        n_samples = 160000

        for i in range(0, n_samples, chunk_size):
            random_audio = np.random.randint(-1000, 1000, chunk_size).astype(np.int16)
            owwModel.predict(random_audio)

        # Run self-confirm
        predictions_dict = owwModel.self_confirm(last_n_seconds=1.5)

        # Verify all models have predictions
        assert len(predictions_dict) >= 2, "predictions_dict should have at least 2 models"

        for model_name, score in predictions_dict.items():
            assert 0 <= score <= 1, f"Score for {model_name} should be between 0 and 1"

    def test_self_confirm_without_enable_flag(self):
        """Test that self_confirm raises ValueError when not enabled"""
        # Initialize model WITHOUT self_confirm enabled
        owwModel = openwakeword.Model(
            wakeword_models=[os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")],
            inference_framework="onnx",
            self_confirm=False
        )

        # Feed in some random data
        chunk_size = 1280
        n_samples = 160000

        for i in range(0, n_samples, chunk_size):
            random_audio = np.random.randint(-1000, 1000, chunk_size).astype(np.int16)
            owwModel.predict(random_audio)

        # Attempting to call self_confirm should raise ValueError
        with pytest.raises(ValueError, match="self-confirm functionality is not enabled"):
            owwModel.self_confirm(last_n_seconds=1.5)

    def test_self_confirm_insufficient_audio_data(self):
        """Test that self_confirm raises ValueError when insufficient audio data"""
        owwModel = openwakeword.Model(
            wakeword_models=[os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")],
            inference_framework="onnx",
            self_confirm=True
        )

        # Feed in only a small amount of data (less than required for self_confirm)
        chunk_size = 1280
        random_audio = np.random.randint(-1000, 1000, chunk_size).astype(np.int16)
        owwModel.predict(random_audio)

        # Attempting to call self_confirm should raise ValueError
        with pytest.raises(ValueError, match="Not enough audio data"):
            owwModel.self_confirm(last_n_seconds=1.5)

    def test_self_confirm_with_tflite_models(self):
        """Test self_confirm with tflite inference framework"""
        owwModel = openwakeword.Model(
            wakeword_models=[os.path.join("openwakeword", "resources", "models", "alexa_v0.1.tflite")],
            inference_framework="tflite",
            self_confirm=True
        )

        # Feed in ~10 seconds of random data
        chunk_size = 1280
        n_samples = 160000

        for i in range(0, n_samples, chunk_size):
            random_audio = np.random.randint(-1000, 1000, chunk_size).astype(np.int16)
            owwModel.predict(random_audio)

        # Run self-confirm
        predictions_dict = owwModel.self_confirm(last_n_seconds=1.5)

        # Verify predictions_dict is properly formed
        assert isinstance(predictions_dict, dict)
        for model_name, score in predictions_dict.items():
            assert 0 <= score <= 1, f"Score for {model_name} should be between 0 and 1"

    def test_self_confirm_multiclass_model(self):
        """Test self_confirm with a multiclass model"""
        owwModel = openwakeword.Model(
            wakeword_models=["timer"],
            inference_framework="onnx",
            self_confirm=True
        )

        # Feed in ~10 seconds of random data
        chunk_size = 1280
        n_samples = 160000

        for i in range(0, n_samples, chunk_size):
            random_audio = np.random.randint(-1000, 1000, chunk_size).astype(np.int16)
            owwModel.predict(random_audio)

        # Run self-confirm
        predictions_dict = owwModel.self_confirm(last_n_seconds=1.5)

        # Verify predictions_dict is properly formed
        assert isinstance(predictions_dict, dict)
        assert len(predictions_dict) > 0, "predictions_dict should not be empty"

        for model_name, score in predictions_dict.items():
            assert isinstance(score, (float, np.floating)), f"Score for {model_name} should be a float"
            assert 0 <= score <= 1, f"Score for {model_name} should be between 0 and 1, got {score}"

    def test_self_confirm_background_true(self):
        """Test self_confirm with background=True returns None and populates confirmation_results"""
        owwModel = openwakeword.Model(
            wakeword_models=[os.path.join("openwakeword", "resources", "models", "alexa_v0.1.onnx")],
            inference_framework="onnx",
            self_confirm=True
        )

        # Feed in ~10 seconds of random data to fill the audio buffer
        chunk_size = 1280
        n_samples = 160000

        for i in range(0, n_samples, chunk_size):
            random_audio = np.random.randint(-1000, 1000, chunk_size).astype(np.int16)
            owwModel.predict(random_audio)

        # Run self-confirm in background mode
        result = owwModel.self_confirm(last_n_seconds=1.5, background=True)

        # When background=True, should return None immediately
        assert result is None, "self_confirm with background=True should return None"

        # confirmation_results should eventually be populated
        # Poll for results with a timeout (max 10 seconds)
        max_wait_time = 10
        start_time = time.time()
        while owwModel.confirmation_results is None and (time.time() - start_time) < max_wait_time:
            time.sleep(0.1)

        # Verify that confirmation_results has been populated
        assert owwModel.confirmation_results is not None, "confirmation_results should be populated after background execution"

        # Verify confirmation_results is properly formed
        predictions_dict = owwModel.confirmation_results
        assert isinstance(predictions_dict, dict), "confirmation_results should be a dictionary"

        expected_models = list(owwModel.models.keys())
        assert len(predictions_dict) == len(expected_models), f"confirmation_results should have {len(expected_models)} key(s)"

        for model_name in expected_models:
            assert model_name in predictions_dict, f"confirmation_results should contain key '{model_name}'"
            score = predictions_dict[model_name]
            assert isinstance(score, (float, np.floating)), f"Score for {model_name} should be a float"
            assert 0 <= score <= 1, f"Score for {model_name} should be between 0 and 1, got {score}"
