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
import sys
import os
import pyaudio
import numpy as np
from openwakeword.model import Model
from openwakeword.resources.webui.server import openWakeWordWebUI
import argparse
from http.server import ThreadingHTTPServer
import threading

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)

parser.add_argument(
    "--vad_threshold",
    help="The minimum threshold for voice activity detection required before an activations",
    type=float,
    default=0.3
)

parser.add_argument(
    "--custom_verifier_model_online_learning",
    help="Whether to enable online learning of a custom verifier model",
    action='store_true',
)

args=parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
owwModel = Model(
    vad_threshold=args.vad_threshold,
    custom_verifier_model_online_learning=args.custom_verifier_model_online_learning
)
# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    print("\n"*13)

    # Start HTTP server
    os.chdir(owwModel.cache_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 9999), lambda *args, **kwargs: openWakeWordWebUI(owwModel, *args, **kwargs))
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    while True:
        try:
            # Get audio
            audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

            # Feed to openWakeWord model
            prediction = owwModel.predict(audio)

            # Column titles
            n_spaces = 16
            output_string_header = """
                Model Name         | Score | Wakeword Status
                --------------------------------------
                """

            for mdl in owwModel.prediction_buffer.keys():
                # Add scores in formatted table
                scores = list(owwModel.prediction_buffer[mdl])
                curr_score = format(scores[-1], '.20f').replace("-", "")

                output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
                """

            # # Print results table
            # print("\033[F"*15)
            # print(output_string_header, "                             ", end='\r')
        
        except KeyboardInterrupt:
            server.shutdown()
            sys.exit(0)
