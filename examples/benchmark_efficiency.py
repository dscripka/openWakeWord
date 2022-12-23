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
import pyaudio
import numpy as np
import os
import argparse
parser=argparse.ArgumentParser()
from openwakeword.model import Model

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)

# Get desired number of CPU cores per calculation
parser.add_argument(
    "--ncores",
    help="How many CPU cores to use for the efficiency estimation",
    type=int,
    default=1
)
args=parser.parse_args()

# Load pre-trained openwakeword models
owwModel = Model()

# Run capture loop, checking for hotwords
if __name__ == "__main__":
    # Continuously predict and estimate CPU usage
    print("\n############################\n\n")
    for i in range(1000000):
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction, timing_dict = owwModel.predict(audio, timing=True)

        # Estimate CPU usage
        total_time = sum([i for i in timing_dict["models"].values()])
        avg_model_time = np.mean([timing_dict["models"][i] for i in timing_dict["models"].keys() if i != "preprocessor"])
        n_possible_models = int((0.08 - total_time)/avg_model_time) + int(0.08/avg_model_time)*(args.ncores-1)

        if i % 10 == 0:
            print(f"Using {round((total_time)/.08*100, 3)}% of {args.ncores} CPU core(s). "
                  f"Could run up to {n_possible_models} additional models.", end='       \r')