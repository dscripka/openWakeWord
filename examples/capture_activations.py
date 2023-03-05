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

##################################

# This example scripts runs openWakeWord continuously on a microphone stream,
# and saves 5 seconds of audio immediately before the activation as WAV clips
# in the specified output location.

##################################

# Imports
import os
import platform
import collections
import time
if platform.system() == "Windows":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio
import numpy as np
from openwakeword.model import Model
import openwakeword
import scipy.io.wavfile
import datetime
import argparse
from utils.beep import playBeep

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    help="Where to save the audio that resulted in an activation",
    type=str,
    default="./",
    required=True
)
parser.add_argument(
    "--threshold",
    help="The score threshold for an activation",
    type=float,
    default=0.5,
    required=False
)
parser.add_argument(
    "--vad_threshold",
    help="""The threshold to use for voice activity detection (VAD) in the openWakeWord instance.
            The default (0.0), disables VAD.""",
    type=float,
    default=0.0,
    required=False
)
parser.add_argument(
    "--noise_suppression",
    help="Whether to enable speex noise suppression in the openWakeWord instance.",
    type=bool,
    default=False,
    required=False
)
parser.add_argument(
    "--model",
    help="The model to use for openWakeWord, leave blank to use all available models",
    type=str,
    required=False
)
parser.add_argument(
    "--disable_activation_sound",
    help="Disables the activation sound, clips are silently captured",
    action='store_true',
    required=False
)

args=parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
if args.model:
    model_paths = openwakeword.get_pretrained_model_paths()
    for path in model_paths:
        if args.model in path:
            model_path = path
            
    if model_path:
        owwModel = Model(
            wakeword_model_paths=[model_path],
            enable_speex_noise_suppression=args.noise_suppression,
            vad_threshold = args.vad_threshold
            )
    else: 
        print(f'Could not find model \"{args.model}\"')
        exit()
else:
    owwModel = Model(
        enable_speex_noise_suppression=args.noise_suppression,
        vad_threshold=args.vad_threshold
    )

# Set waiting period after activation before saving clip (to get some audio context after the activation)
save_delay = 1  # seconds

# Set cooldown period before another clip can be saved
cooldown = 4  # seconds

# Create output directory if it does not already exist
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Run capture loop, checking for hotwords
if __name__ == "__main__":
    # Predict continuously on audio stream
    last_save = time.time()
    activation_times = collections.defaultdict(list)

    print("\n\nListening for wakewords...\n")
    while True:
        # Get audio
        mic_audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(mic_audio)

        # Check for model activations (score above threshold), and save clips
        for mdl in prediction.keys():
            if prediction[mdl] >= args.threshold:
                activation_times[mdl].append(time.time())

            if activation_times.get(mdl) and (time.time() - last_save) >= cooldown \
               and (time.time() - activation_times.get(mdl)[0]) >= save_delay:
                last_save = time.time()
                activation_times[mdl] = []
                detect_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                
                print(f'Detected activation from \"{mdl}\" model at time {detect_time}!')

                # Capture total of 5 seconds, with the microphone audio associated with the
                # activation around the ~4 second point
                audio_context = np.array(list(owwModel.preprocessor.raw_data_buffer)[-16000*5:]).astype(np.int16)
                fname = detect_time + f"_{mdl}.wav"
                scipy.io.wavfile.write(os.path.join(os.path.abspath(args.output_dir), fname), 16000, audio_context)
                
                if not args.disable_activation_sound:
                    playBeep(os.path.join(os.path.dirname(__file__), 'audio', 'activation.wav'), audio)
