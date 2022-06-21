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
import scipy.io.wavfile
import pytest

# Tests
class TestModels:
    @pytest.fixture(scope="class")
    def hey_jane_clip(self):
        sr, dat = scipy.io.wavfile.read("tests/data/hey_jane.wav", "rb")
        return dat

    def test_hey_jane(self, hey_jane_clip):
        model = openwakeword.Model(
            wakeword_model_paths=["openwakeword/resources/models/hey_jane.onnx"],
            input_sizes=[16]
        )

        step_size = 1280
        predictions = []
        for i in range(0, hey_jane_clip.shape[0]-step_size, step_size):
            predictions.append(model.predict(hey_jane_clip[i:i+step_size])["hey_jane"])
        
        assert max(predictions) > 0.5
