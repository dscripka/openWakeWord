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
import openwakeword
from openwakeword.utils import AudioFeatures, re_arg

import wave
import os
import logging
import functools
import pickle
from collections import deque, defaultdict
from functools import partial
import time
from typing import List, Union, DefaultDict, Dict


# Define main model class
class Model():
    """
    The main model class for openWakeWord. Creates a model object with the shared audio pre-processer
    and for arbitrarily many custom wake word/wake phrase models.
    """
    @re_arg({"wakeword_model_paths": "wakeword_models"})  # temporary handling of keyword argument change
    def __init__(
            self,
            wakeword_models: List[str] = [],
            class_mapping_dicts: List[dict] = [],
            enable_speex_noise_suppression: bool = False,
            vad_threshold: float = 0,
            custom_verifier_models: dict = {},
            custom_verifier_threshold: float = 0.1,
            inference_framework: str = "tflite",
            **kwargs
            ):
        """Initialize the openWakeWord model object.

        Args:
            wakeword_models (List[str]): A list of paths of ONNX/tflite models to load into the openWakeWord model object.
                                              If not provided, will load all of the pre-trained models. Alternatively,
                                              just the names of pre-trained models can be provided to select a subset of models.
            class_mapping_dicts (List[dict]): A list of dictionaries with integer to string class mappings for
                                              each model in the `wakeword_models` arguments
                                              (e.g., {"0": "class_1", "1": "class_2"})
            enable_speex_noise_suppression (bool): Whether to use the noise suppresion from the SpeexDSP
                                                   library to pre-process all incoming audio. May increase
                                                   model performance when reasonably stationary background noise
                                                   is present in the environment where openWakeWord will be used.
                                                   It is very lightweight, so enabling it doesn't significantly
                                                   impact efficiency.
            vad_threshold (float): Whether to use a voice activity detection model (VAD) from Silero
                                   (https://github.com/snakers4/silero-vad) to filter predictions.
                                   For every input audio frame, a VAD score is obtained and only those model predictions
                                   with VAD scores above the threshold will be returned. The default value (0),
                                   disables voice activity detection entirely.
            custom_verifier_models (dict): A dictionary of paths to custom verifier models, where
                                           the keys are the model names (corresponding to the openwakeword.MODELS
                                           attribute) and the values are the filepaths of the
                                           custom verifier models.
            custom_verifier_threshold (float): The score threshold to use a custom verifier model. If the score
                                               from a model for a given frame is greater than this value, the
                                               associated custom verifier model will also predict on that frame, and
                                               the verifier score will be returned.
            inference_framework (str): The inference framework to use when for model prediction. Options are
                                       "tflite" or "onnx". The default is "tflite" as this results in better
                                       efficiency on common platforms (x86, ARM64), but in some deployment
                                       scenarios ONNX models may be preferable.
            kwargs (dict): Any other keyword arguments to pass the the preprocessor instance
        """
        # Get model paths for pre-trained models if user doesn't provide models to load
        pretrained_model_paths = openwakeword.get_pretrained_model_paths(inference_framework)
        wakeword_model_names = []
        if wakeword_models == []:
            wakeword_models = pretrained_model_paths
            wakeword_model_names = list(openwakeword.MODELS.keys())
        elif len(wakeword_models) >= 1:
            for ndx, i in enumerate(wakeword_models):
                if os.path.exists(i):
                    wakeword_model_names.append(os.path.splitext(os.path.basename(i))[0])
                else:
                    # Find pre-trained path by modelname
                    matching_model = [j for j in pretrained_model_paths if i.replace(" ", "_") in j.split(os.path.sep)[-1]]
                    if matching_model == []:
                        raise ValueError("Could not find pretrained model for model name '{}'".format(i))
                    else:
                        wakeword_models[ndx] = matching_model[0]
                        wakeword_model_names.append(i)

        # Create attributes to store models and metadata
        self.models = {}
        self.model_inputs = {}
        self.model_outputs = {}
        self.model_prediction_function = {}
        self.class_mapping = {}
        self.custom_verifier_models = {}
        self.custom_verifier_threshold = custom_verifier_threshold

        # Do imports for  inference framework
        if inference_framework == "tflite":
            try:
                import tflite_runtime.interpreter as tflite

                def tflite_predict(tflite_interpreter, input_index, output_index, x):
                    tflite_interpreter.set_tensor(input_index, x)
                    tflite_interpreter.invoke()
                    return tflite_interpreter.get_tensor(output_index)[None, ]

            except ImportError:
                logging.warning("Tried to import the tflite runtime, but it was not found. "
                                "Trying to switching to onnxruntime instead, if appropriate models are available.")
                if wakeword_models != [] and all(['.onnx' in i for i in wakeword_models]):
                    inference_framework = "onnx"
                elif wakeword_models != [] and all([os.path.exists(i.replace('.tflite', '.onnx')) for i in wakeword_models]):
                    inference_framework = "onnx"
                    wakeword_models = [i.replace('.tflite', '.onnx') for i in wakeword_models]
                else:
                    raise ValueError("Tried to import the tflite runtime for provided tflite models, but it was not found. "
                                     "Please install it using `pip install tflite-runtime`")

        if inference_framework == "onnx":
            try:
                import onnxruntime as ort

                def onnx_predict(onnx_model, x):
                    return onnx_model.run(None, {onnx_model.get_inputs()[0].name: x})

            except ImportError:
                raise ValueError("Tried to import onnxruntime, but it was not found. Please install it using `pip install onnxruntime`")

        for mdl_path, mdl_name in zip(wakeword_models, wakeword_model_names):
            # Load openwakeword models
            if inference_framework == "onnx":
                if ".tflite" in mdl_path:
                    raise ValueError("The onnx inference framework is selected, but tflite models were provided!")

                sessionOptions = ort.SessionOptions()
                sessionOptions.inter_op_num_threads = 1
                sessionOptions.intra_op_num_threads = 1

                self.models[mdl_name] = ort.InferenceSession(mdl_path, sess_options=sessionOptions,
                                                             providers=["CPUExecutionProvider"])

                self.model_inputs[mdl_name] = self.models[mdl_name].get_inputs()[0].shape[1]
                self.model_outputs[mdl_name] = self.models[mdl_name].get_outputs()[0].shape[1]
                pred_function = functools.partial(onnx_predict, self.models[mdl_name])
                self.model_prediction_function[mdl_name] = pred_function

            if inference_framework == "tflite":
                if ".onnx" in mdl_path:
                    raise ValueError("The tflite inference framework is selected, but onnx models were provided!")

                self.models[mdl_name] = tflite.Interpreter(model_path=mdl_path, num_threads=1)
                self.models[mdl_name].allocate_tensors()

                self.model_inputs[mdl_name] = self.models[mdl_name].get_input_details()[0]['shape'][1]
                self.model_outputs[mdl_name] = self.models[mdl_name].get_output_details()[0]['shape'][1]

                tflite_input_index = self.models[mdl_name].get_input_details()[0]['index']
                tflite_output_index = self.models[mdl_name].get_output_details()[0]['index']

                pred_function = functools.partial(tflite_predict, self.models[mdl_name], tflite_input_index, tflite_output_index)
                self.model_prediction_function[mdl_name] = pred_function

            if class_mapping_dicts and class_mapping_dicts[wakeword_models.index(mdl_path)].get(mdl_name, None):
                self.class_mapping[mdl_name] = class_mapping_dicts[wakeword_models.index(mdl_path)]
            elif openwakeword.model_class_mappings.get(mdl_name, None):
                self.class_mapping[mdl_name] = openwakeword.model_class_mappings[mdl_name]
            else:
                self.class_mapping[mdl_name] = {str(i): str(i) for i in range(0, self.model_outputs[mdl_name])}

            # Load custom verifier models
            if isinstance(custom_verifier_models, dict):
                if custom_verifier_models.get(mdl_name, False):
                    self.custom_verifier_models[mdl_name] = pickle.load(open(custom_verifier_models[mdl_name], 'rb'))

            if len(self.custom_verifier_models.keys()) < len(custom_verifier_models.keys()):
                raise ValueError(
                    "Custom verifier models were provided, but some were not matched with a base model!"
                    " Make sure that the keys provided in the `custom_verifier_models` dictionary argument"
                    " exactly match that of the `.models` attribute of an instantiated openWakeWord Model object"
                    " that has the same base models but doesn't have custom verifier models."
                )

        # Create buffer to store frame predictions
        self.prediction_buffer: DefaultDict[str, deque] = defaultdict(partial(deque, maxlen=30))

        # Initialize SpeexDSP noise canceller
        if enable_speex_noise_suppression:
            from speexdsp_ns import NoiseSuppression
            self.speex_ns = NoiseSuppression.create(160, 16000)
        else:
            self.speex_ns = None

        # Initialize Silero VAD
        self.vad_threshold = vad_threshold
        if vad_threshold > 0:
            self.vad = openwakeword.VAD()

        # Create AudioFeatures object
        self.preprocessor = AudioFeatures(inference_framework=inference_framework, **kwargs)

    def get_parent_model_from_label(self, label):
        """Gets the parent model associated with a given prediction label"""
        parent_model = ""
        for mdl in self.class_mapping.keys():
            if label in self.class_mapping[mdl].values():
                parent_model = mdl
            elif label in self.class_mapping.keys() and label == mdl:
                parent_model = mdl

        return parent_model

    def reset(self):
        """Reset the prediction and audio feature buffers. Useful for re-initializing the model, though may not be efficient
        when called too frequently."""
        self.prediction_buffer = defaultdict(partial(deque, maxlen=30))
        self.preprocessor.reset()

    def predict(self, x: np.ndarray, patience: dict = {},
                threshold: dict = {}, debounce_time: float = 0.0, timing: bool = False):
        """Predict with all of the wakeword models on the input audio frames

        Args:
            x (ndarray): The input audio data to predict on with the models. Ideally should be multiples of 80 ms
                                (1280 samples), with longer lengths reducing overall CPU usage
                                but decreasing detection latency. Input audio with durations greater than or less
                                than 80 ms is also supported, though this will add a detection delay of up to 80 ms
                                as the appropriate number of samples are accumulated.
            patience (dict): How many consecutive frames (of 1280 samples or 80 ms) above the threshold that must
                             be observed before the current frame will be returned as non-zero.
                             Must be provided as an a dictionary where the keys are the
                             model names and the values are the number of frames. Can reduce false-positive
                             detections at the cost of a lower true-positive rate.
                             By default, this behavior is disabled.
            threshold (dict): The threshold values to use when the `patience` or `debounce_time` behavior is enabled.
                              Must be provided as an a dictionary where the keys are the
                              model names and the values are the thresholds.
            debounce_time (float): The time (in seconds) to wait before returning another non-zero prediction
                                   after a non-zero prediction. Can preven multiple detections of the same wake-word.
            timing (bool): Whether to return timing information of the models. Can be useful to debug and
                           assess how efficiently models are running on the current hardware.

        Returns:
            dict: A dictionary of scores between 0 and 1 for each model, where 0 indicates no
                  wake-word/wake-phrase detected. If the `timing` argument is true, returns a
                  tuple of dicts containing model predictions and timing information, respectively.
        """
        # Check input data type
        if not isinstance(x, np.ndarray):
            raise ValueError(f"The input audio data (x) must by a Numpy array, instead received an object of type {type(x)}.")

        # Setup timing dict
        if timing:
            timing_dict: Dict[str, Dict] = {}
            timing_dict["models"] = {}
            feature_start = time.time()

        # Get audio features (optionally with Speex noise suppression)
        if self.speex_ns:
            n_prepared_samples = self.preprocessor(self._suppress_noise_with_speex(x))
        else:
            n_prepared_samples = self.preprocessor(x)

        if timing:
            timing_dict["models"]["preprocessor"] = time.time() - feature_start

        # Get predictions from model(s)
        predictions = {}
        for mdl in self.models.keys():
            if timing:
                model_start = time.time()

            # Run model to get predictions
            if n_prepared_samples > 1280:
                group_predictions = []
                for i in np.arange(n_prepared_samples//1280-1, -1, -1):
                    group_predictions.extend(
                        self.model_prediction_function[mdl](
                            self.preprocessor.get_features(
                                    self.model_inputs[mdl],
                                    start_ndx=-self.model_inputs[mdl] - i
                            )
                        )
                    )
                prediction = np.array(group_predictions).max(axis=0)[None, ]
            elif n_prepared_samples == 1280:
                prediction = self.model_prediction_function[mdl](
                    self.preprocessor.get_features(self.model_inputs[mdl])
                )
            elif n_prepared_samples < 1280:  # get previous prediction if there aren't enough samples
                if self.model_outputs[mdl] == 1:
                    if len(self.prediction_buffer[mdl]) > 0:
                        prediction = [[[self.prediction_buffer[mdl][-1]]]]
                    else:
                        prediction = [[[0]]]
                elif self.model_outputs[mdl] != 1:
                    n_classes = max([int(i) for i in self.class_mapping[mdl].keys()])
                    prediction = [[[0]*(n_classes+1)]]

            if self.model_outputs[mdl] == 1:
                predictions[mdl] = prediction[0][0][0]
            else:
                for int_label, cls in self.class_mapping[mdl].items():
                    predictions[cls] = prediction[0][0][int(int_label)]

            # Update scores based on custom verifier model
            if self.custom_verifier_models != {}:
                for cls in predictions.keys():
                    if predictions[cls] >= self.custom_verifier_threshold:
                        parent_model = self.get_parent_model_from_label(cls)
                        if self.custom_verifier_models.get(parent_model, False):
                            verifier_prediction = self.custom_verifier_models[parent_model].predict_proba(
                                self.preprocessor.get_features(self.model_inputs[mdl])
                            )[0][-1]
                            predictions[cls] = verifier_prediction

            # Zero predictions for first 5 frames during model initialization
            for cls in predictions.keys():
                if len(self.prediction_buffer[cls]) < 5:
                    predictions[cls] = 0.0

            # Get timing information
            if timing:
                timing_dict["models"][mdl] = time.time() - model_start

        # Update scores based on thresholds or patience arguments
        if patience != {} or debounce_time > 0:
            if threshold == {}:
                raise ValueError("Error! When using the `patience` argument, threshold "
                                 "values must be provided via the `threshold` argument!")
            if patience != {} and debounce_time > 0:
                raise ValueError("Error! The `patience` and `debounce_time` arguments cannot be used together!")
            for mdl in predictions.keys():
                parent_model = self.get_parent_model_from_label(mdl)
                if predictions[mdl] != 0.0:
                    if parent_model in patience.keys():
                        scores = np.array(self.prediction_buffer[mdl])[-patience[parent_model]:]
                        if (scores >= threshold[parent_model]).sum() < patience[parent_model]:
                            predictions[mdl] = 0.0
                    elif debounce_time > 0:
                        if parent_model in threshold.keys():
                            n_frames = int(np.ceil(debounce_time/(n_prepared_samples/16000)))
                            recent_predictions = np.array(self.prediction_buffer[mdl])[-n_frames:]
                            if predictions[mdl] >= threshold[parent_model] and \
                               (recent_predictions >= threshold[parent_model]).sum() > 0:
                                predictions[mdl] = 0.0

        # Update prediction buffer
        for mdl in predictions.keys():
            self.prediction_buffer[mdl].append(predictions[mdl])

        # (optionally) get voice activity detection scores and update model scores
        if self.vad_threshold > 0:
            if timing:
                vad_start = time.time()

            self.vad(x)

            if timing:
                timing_dict["models"]["vad"] = time.time() - vad_start

            # Get frames from last 0.4 to 0.56 seconds (3 frames) before the current
            # frame and get max VAD score
            vad_frames = list(self.vad.prediction_buffer)[-7:-4]
            vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0
            for mdl in predictions.keys():
                if vad_max_score < self.vad_threshold:
                    predictions[mdl] = 0.0

        if timing:
            return predictions, timing_dict
        else:
            return predictions

    def predict_clip(self, clip: Union[str, np.ndarray], padding: int = 1, chunk_size=1280, **kwargs):
        """Predict on an full audio clip, simulating streaming prediction.
        The input clip must bit a 16-bit, 16 khz, single-channel WAV file.

        Args:
            clip (Union[str, np.ndarray]): The path to a 16-bit PCM, 16 khz, single-channel WAV file,
                                           or an 1D array containing the same type of data
            padding (int): How many seconds of silence to pad the start/end of the clip with
                            to make sure that short clips can be processed correctly (default: 1)
            chunk_size (int): The size (in samples) of each chunk of audio to pass to the model
            kwargs: Any keyword arguments to pass to the class `predict` method

        Returns:
            list: A list containing the frame-level prediction dictionaries for the audio clip
        """
        if isinstance(clip, str):
            # Load audio clip as 16-bit PCM data
            with wave.open(clip, mode='rb') as f:
                # Load WAV clip frames
                data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        elif isinstance(clip, np.ndarray):
            data = clip

        if padding:
            data = np.concatenate(
                (
                    np.zeros(16000*padding).astype(np.int16),
                    data,
                    np.zeros(16000*padding).astype(np.int16)
                )
            )

        # Iterate through clip, getting predictions
        predictions = []
        step_size = chunk_size
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions.append(self.predict(data[i:i+step_size], **kwargs))

        return predictions

    def _get_positive_prediction_frames(
            self,
            file: str,
            threshold: float = 0.5,
            return_type: str = "features",
            **kwargs
            ):
        """
        Gets predictions for the input audio data, and returns the audio features (embeddings)
        or audio data for all of the frames with a score above the `threshold` argument.
        Can be a useful way to collect false-positive predictions.

        Args:
            file (str): The path to a 16-bit 16khz WAV audio file to process
            threshold (float): The minimum score required for a frame of audio features
                               to be returned.
            return_type (str): The type of data to return when a positive prediction is
                               detected. Can be either 'features' or 'audio' to return
                               audio embeddings or raw audio data, respectively.
            kwargs: Any keyword arguments to pass to the class `predict` method

        Returns:
            dict: A dictionary with filenames as keys and  N x M arrays as values,
                  where N is the number of examples and M is the number
                  of audio features, depending on the model input shape.
        """
        # Load audio clip as 16-bit PCM data
        with wave.open(file, mode='rb') as f:
            # Load WAV clip frames
            data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

        # Iterate through clip, getting predictions
        positive_data = defaultdict(list)
        step_size = 1280
        for i in range(0, data.shape[0]-step_size, step_size):
            predictions = self.predict(data[i:i+step_size], **kwargs)
            for lbl in predictions.keys():
                if predictions[lbl] >= threshold:
                    mdl = self.get_parent_model_from_label(lbl)
                    features = self.preprocessor.get_features(self.model_inputs[mdl])
                    if return_type == 'features':
                        positive_data[lbl].append(features)
                    if return_type == 'audio':
                        context = data[max(0, i - 16000*3):i + 16000]
                        if len(context) == 16000*4:
                            positive_data[lbl].append(context)

        positive_data_combined = {}
        for lbl in positive_data.keys():
            positive_data_combined[lbl] = np.vstack(positive_data[lbl])

        return positive_data_combined

    def _suppress_noise_with_speex(self, x: np.ndarray, frame_size: int = 160):
        """
        Runs the input audio through the SpeexDSP noise suppression algorithm.
        Note that this function updates the state of the existing Speex noise
        suppression object, and isn't intended to be called externally.

        Args:
            x (ndarray): The 16-bit, 16khz audio to process. Must always be an
                         integer multiple of `frame_size`.
            frame_size (int): The frame size to use for the Speex Noise suppressor.
                              Must match the frame size specified during the
                              initialization of the noise suppressor.

        Returns:
            ndarray: The input audio with noise suppression applied
        """
        cleaned = []
        for i in range(0, x.shape[0], frame_size):
            chunk = x[i:i+frame_size]
            cleaned.append(self.speex_ns.process(chunk.tobytes()))

        cleaned_bytestring = b''.join(cleaned)
        cleaned_array = np.frombuffer(cleaned_bytestring, np.int16)
        return cleaned_array
