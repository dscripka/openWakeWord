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
import numpy as np
import os
import scipy.io.wavfile
import tempfile
import openwakeword
import argparse
import time
from speechbrain.dataio.dataio import read_audio
import collections
from tqdm import tqdm

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--input_files",
    help="""A text file where each line is a full path to an audio file to mine for false-positives.""",
    type=str,
    default="./",
    required=True
)
parser.add_argument(
    "--skip_files",
    help="""A text file where each line is a full path to an audio file that should be skipped.""",
    type=str,
    required=False
)
parser.add_argument(
    "--output_dir",
    help="""Where to save the audio features from a false-positive.
          By default, will be saved as <model_name>.npy files of shape N_clips x frames x features""",
    type=str,
    default="./",
    required=True
)
parser.add_argument(
    "--n_threads",
    help="""The number of CPU threads to use when processing.""",
    type=int,
    default=1,
    required=False
)
parser.add_argument(
    "--max_wall_time",
    help="""The total amount of wall-clock time (in hours) to mine for false-positives. When this limit is reached
            the examples found up to this point will be saved.""",
    type=float,
    default=1,
    required=False
)
parser.add_argument(
    "--max_feature_size",
    help="""The maximum size (in MB) for the false-positive features. If the total collected is larger
            is than this, processing will stop.""",
    type=float,
    default=5000,
    required=False
)
args=parser.parse_args()

if __name__ == "__main__":
    # Get audio files to mine from input list
    with open(args.input_files, 'r') as f:
        input_files = [i.strip() for i in f.readlines()]

    # Get audio files to skip and adjust input file list
    if args.skip_files:
        with open(args.skip_files, 'r') as f:
            skip_files = [i.strip() for i in f.readlines()]
        input_files = [i for i in input_files if i not in skip_files]

    # Set starting time
    start_time = time.time()

    # Begin processing files
    bs = int(args.n_threads*2)
    combined_features = collections.defaultdict(list)
    for i in tqdm(range(0, len(input_files), bs)):
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch = input_files[i:i+bs]
            batch_data = []
            tmp_file_paths = []
            for i in batch:
                dat = read_audio(i).numpy()
                if len(dat.shape) > 1:
                    dat = dat[:, 0]
                dat = (dat*32767).astype(np.int16) # convert to 16-khz, 16-bit audio

                # Save audio to temporary .wav files
                tmp_fname = os.path.join(tmp_dir, i.split(os.path.sep)[-1])
                scipy.io.wavfile.write(tmp_fname, 16000, dat)
                tmp_file_paths.append(tmp_fname)
    
            # Predict on temporary files
            predictions = openwakeword.utils.bulk_predict(
                file_paths=tmp_file_paths,
                wakeword_model_paths=[], # loads all default models
                prediction_function="_get_positive_prediction_frames",
                ncpu=args.n_threads
            )
            
            # Combine and store features
            for fl in predictions.keys():
                for lbl in predictions[fl].keys():
                    combined_features[lbl].append(predictions[fl][lbl])

            # Check for maximum processing time       
            if (time.time() - start_time)/3600 > args.max_wall_time:
                print("\nMaximum wall-time reached. Saving mined false-positives and exiting...")
                break

            # Check for maximum features size in memory
            size = 0
            for key in combined_features.keys():
                for i in combined_features[key]:
                    size += i.nbytes/1e6
            if size > args.max_feature_size:
                print("\nMaximum feature size (in MB) reached. Saving mined false-positives and exiting...")
                break

    # Combine mined features into single numpy arrays
    for lbl in combined_features.keys():
        combined_features[lbl] = np.concatenate(combined_features[lbl], axis=0)

    # Save results to .npy files
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for key in combined_features.keys():
        np.save(f"{args.output_dir}{os.path.sep}{key}.npy", combined_features[key])
        