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

# Load pre-trained openwakeword models
owwModel = Model()

# Run capture loop, checking for hotwords
if __name__ == "__main__":
    # Start the mic stream
    mic_stream.start()

    # Create a prediction buffer
    while True:
        # Get audio
        audio, overflowed = mic_stream.read(1280)
        audio = audio.squeeze()

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Get predictions from prediction buffers and plot
        plt.cld()
        plt.clt()
        for mdl in owwModel.prediction_buffer.keys():
            # Plot scores in graph
            scores = list(owwModel.prediction_buffer[mdl])
            plt.plot(scores)

            # Plot text showing name of model with scores >= 0.5 (default threshold)
            if max(scores) >= 0.5:
                plt.text(mdl, 15, 0.9, alignment="center", color = "blue", style="bold")

        plt.ylim(0,1)
        plt.show()
        plt.sleep(0.005)
