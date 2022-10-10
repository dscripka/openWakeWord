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
import plotext as plt
import sounddevice
import numpy as np
from openwakeword.model import Model

# Get microphone stream
mic_stream = sounddevice.InputStream(
    samplerate=16000,
    blocksize=1280,
    device = 3,
    dtype = np.int16,
)

# Load openwakeword model(s)
model_name = "hey_mycroft_v1"
model = Model(
    wakeword_model_paths=[os.path.join("../", "openwakeword", "resources", "models", model_name + ".onnx")],
    input_sizes=[16]
)

# Run capture loop, checking for hotwords
if __name__ == "__main__":
    # Start the mic stream
    mic_stream.start()

    # Create a prediction buffer
    prediction_buffer = [0]*30
    while True:
        # Get audio
        audio, overflowed = mic_stream.read(1280)
        audio = audio.squeeze()

        # Feed to openWakeWord model
        prediction = model.predict(audio)
        prediction_buffer = prediction_buffer[1:] + [round(prediction[model_name], 2)]
        
        # Plot predictions in graph
        plt.cld()
        plt.clt()
        plt.plot(prediction_buffer)
        plt.ylim(0,1)
        plt.show()
        plt.sleep(0.005)