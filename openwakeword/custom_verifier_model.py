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
from tqdm import tqdm
import collections
import openwakeword
import numpy as np
import scipy
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# Define functions to prepare data for speaker dependent verifier model
def get_reference_clip_features(
        reference_clip: str,
        oww_model: openwakeword.Model,
        model_name: str,
        threshold: float = 0.5,
        N: int = 3,
        **kwargs
        ):
    """
    Processes input audio files (16-bit, 16-khz single-channel WAV files) and gets the openWakeWord
    audio features that produce a prediction from the specified model greater than the threshold value.


    Args:
        reference_clip (str): The target audio file to get features from
        oww_model (openwakeword.Model): The openWakeWord model object used to get predictions
        model_name (str): The name of the model to get predictions from (should correspond to
                          a python dictionary key in the oww_model.models attribute)
        threshold (float): The minimum score from the model required to capture the associated features
        N (int): How many times to run feature extraction for a given clip, adding some slight variation
                 in the starting position each time to ensure that the features are not identical

    Returns:
        ndarray: A numpy array of shape N x M x L, where N is the number of examples, M is the number
                 of frames in the window, and L is the audio feature/embedding dimension.
    """

    # Create dictionary to store frames
    positive_data = collections.defaultdict(list)

    # Get predictions
    for _ in range(N):
        # Load clip
        if type(reference_clip) == str:
            sr, dat = scipy.io.wavfile.read(reference_clip)
        else:
            dat = reference_clip

        # Set random starting point to get small variations in features
        if N != 1:
            dat = dat[np.random.randint(0, 1280):]

        # Get predictions
        step_size = 1280
        for i in range(0, dat.shape[0]-step_size, step_size):
            predictions = oww_model.predict(dat[i:i+step_size], **kwargs)
            if predictions[model_name] >= threshold:
                features = oww_model.preprocessor.get_features(  # type: ignore[has-type]
                    oww_model.model_inputs[model_name]           # type: ignore[has-type]
                )
                positive_data[model_name].append(features)

    if len(positive_data[model_name]) == 0:
        positive_data[model_name].append(
            np.empty((0, oww_model.model_inputs[model_name], 96)))  # type: ignore[has-type]

    return np.vstack(positive_data[model_name])


def flatten_features(x):
    return [i.flatten() for i in x]


def train_verifier_model(features: np.ndarray, labels: np.ndarray):
    """
    Train a logistic regression binary classifier model on the provided features and labels

    Args:
        features (ndarray): A N x M numpy array, where N is the number of examples and M
                             is the number of features
        labels (ndarray): A 1D numpy array where each value corresponds to the label of the Nth
                           example in the `features` argument

    Returns:
        The trained scikit-learn logistic regression model
    """
    # C value matters alot here, depending on dataset size (larger datasets work better with larger C?)
    clf = LogisticRegression(random_state=0, max_iter=2000, C=0.001)
    pipeline = make_pipeline(FunctionTransformer(flatten_features), StandardScaler(), clf)
    pipeline.fit(features, labels)

    return pipeline


def train_custom_verifier(
        positive_reference_clips: str,
        negative_reference_clips: str,
        output_path: str,
        model_name: str,
        **kwargs
        ):
    """
    Trains a voice-specific custom verifier model on examples of wake word/phrase speech and other speech
    from a single user.

    Args:
        positive_reference_clips (str): The path to a directory containing single-channel 16khz, 16-bit WAV files
                                        of the target wake word/phrase.
        negative_reference_clips (str): The path to a directory containing single-channel 16khz, 16-bit WAV files
                                        of miscellaneous speech not containing the target wake word/phrase.
        output_path (str): The location to save the trained verifier model (as a scikit-learn .joblib file)
        model_name (str): The name or path of the trained openWakeWord model that the verifier model will be
                          based on. If only a name, it must be one of the pre-trained models included in the
                          openWakeWord release.
        kwargs: Any other keyword arguments to pass to the openWakeWord model initialization

    Returns:
        None
    """
    # Load target openWakeWord model
    if os.path.exists(model_name):
        oww = openwakeword.Model(
            wakeword_models=[model_name],
            **kwargs
        )
        model_name = os.path.splitext(model_name)[0].split(os.path.sep)[-1]
    else:
        oww = openwakeword.Model(**kwargs)

    # Get features from positive reference clips
    positive_features = np.vstack(
        [get_reference_clip_features(i, oww, model_name, N=5)
         for i in tqdm(positive_reference_clips, desc="Processing positive reference clips")]
    )
    if positive_features.shape[0] == 0:
        raise ValueError("The positive features were created! Make sure that"
                         " the positive reference clips contain the appropriate audio"
                         " for the desired model")

    # Get features from negative reference clips
    negative_features = np.vstack(
        [get_reference_clip_features(i, oww, model_name, threshold=0.0, N=1)
         for i in tqdm(negative_reference_clips, desc="Processing negative reference clips")]
    )

    # Train logistic regression model on reference clip features
    print("Training and saving verifier model...")
    lr_model = train_verifier_model(
        np.vstack((positive_features, negative_features)),
        np.array([1]*positive_features.shape[0] + [0]*negative_features.shape[0])
    )

    # Save logistic regression model to specified output location
    print("Done!")
    pickle.dump(lr_model, open(output_path, "wb"))
