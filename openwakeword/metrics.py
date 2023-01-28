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
import re
from tqdm import tqdm
import numpy as np
from typing import List


# Define metric utility functions specific to the wakeword detection use-case

def get_false_positives(scores: List, threshold: float, grouping_window: int = 50):
    """
    Counts the number of false-positives based on a list of scores and a specified threshold.

    Args:
        scores (List): A list of predicted scores, between 0 and 1
        threshold (float): The threshold to use to determine false-positive predictions
        grouping_window (int: The size (in number of frames) for grouping scores above
                                 the threshold into a single false positive for counting

    Returns:
        int: The number of false positive predictions in the list of scores
    """
    bin_pred = np.array(scores) >= threshold
    bin_pred_string = ''.join(["1" if i else "0" for i in bin_pred])
    transitions = list(re.finditer("01", bin_pred_string))
    n = grouping_window
    for t in transitions:
        if bin_pred[t.end()] != 0:
            bin_pred[t.end():t.end() + min(len(transitions) - t.end(), n)] = [0]*min(len(transitions) - t.end(), n)

    return sum(bin_pred)


def generate_roc_curve_fprs(
                            scores: list,
                            n_points: int = 25,
                            time_per_prediction: float = .08,
                            **kwargs
                            ):
    """
    Generates the false positive rate (fpr) per hour for the given predictions
    over a range of score thresholds. Assumes that all predictions should be less than the threshold,
    else the prediction is a false positive.

    Args:
        scores (List): A list of predicted scores, between 0 and 1
        n_points (int): The number of points to use when calculating false positive rates
        time_per_prediction (float): The time (in seconds) that each prediction represents
        kwargs (dict): Any other keyword arguments to pass to the `get_false_positives` function

    Returns:
        list: A list of false positive rates per hour at different score threshold levels
    """

    # Determine total time
    total_hours = time_per_prediction*len(scores)/3600  # convert to hours

    # Calculate true positive rate
    fprs = []
    for threshold in tqdm(np.linspace(0.01, 0.99, num=n_points)):
        fpr = get_false_positives(scores, threshold=threshold, **kwargs)
        fprs.append(fpr/total_hours)

    return fprs


def generate_roc_curve_tprs(
                            scores: list,
                            n_points: int = 25
                            ):
    """
    Generates the true positive rate (true accept rate) for the given predictions
    over a range score thresholds. Assumes that all predictions are supposed to be equal to 1.

    Args:
        scores (list): A list of scores for each prediction

    Returns:
        list: A list of true positive rates at different score threshold levels
    """

    tprs = []
    for threshold in tqdm(np.linspace(0.01, 0.99, num=n_points)):
        tprs.append(sum(scores >= threshold)/len(scores))

    return tprs
