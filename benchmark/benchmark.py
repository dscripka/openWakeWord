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
import numpy as np
from pathlib import Path
from collections import defaultdict

# Define benchmark to assess inference speed of models at different audio chunk sizes
# Smaller chunk sizes may increase model performance, at the cost of inference efficiency
def run_benchmark():
    # Load models
    model_paths = [str(i) for i in Path("openwakeword/resources/models").glob("*.onnx") \
                   if "embedding" not in str(i) and "melspectrogram" not in str(i)]
    M = openwakeword.Model(
        wakeword_model_paths=model_paths,
        input_sizes=[16]
    )

    # Create random data to use for benchmarking
    clip = np.random.random(16000*10).astype(np.float32)

    # Run the benchmark
    step_size = 1280
    preprocessing_times = []
    model_times = defaultdict(list)
    for i in range(0, clip.shape[0]-step_size, step_size):
        pred, timing_dict = M.predict(clip[i:i+step_size], timing=True)
        preprocessing_times.append(timing_dict["preprocessor"])
        for mdl_name in M.models.keys():
            model_times[mdl_name].append(timing_dict["models"][mdl_name])

    print(f"Average of {np.mean(preprocessing_times)} for audio preprocessing with a frame size of {step_size/16000} seconds")
    for mdl_name in M.models.keys():
        print(f"Average of {np.mean(model_times[mdl_name])} for model \"{mdl_name}\"", "\n\n")

if __name__ == "__main__":
   run_benchmark()