![Github CI](https://github.com/dscripka/openWakeWord/actions/workflows/tests.yml/badge.svg)

# openWakeWord

openWakeWord is a fully open-source wakeword library that can be used to create voice-enabled applications and interfaces. It has a (growing) set of pre-trained models for common words & phrases that work well in real-world environments, and does not require user-specific data or training to achieve robust performance.

# Installation & Usage

Installing openWakeWord is simple, and there are minimal dependencies:

```
pip install openwakeword
```

If you want to do a full-install to support all examples and model training (note! will install PyTorch):

```
pip install openwakeword[full]
```

For testing, use the included [example script](examples/detect_from_microphone.py) to try streaming detection from a local microphone.

Or, try the default included models right in your browser at the [HuggingFace Spaces demo]()!

Include openWakeWord in your own code with just a few lines:

```python
from openwakeword.model import Model

# Instantiate the model
model = Model(
    wakeword_model_paths=["path/to/model.onnx"],
)

# Get an 80 ms audio frame from a file, microphone, network stream, etc.
frame = get_audio_frame()

# Get a prediction for the frame
prediction = model.predict(frame)
```

# Project Goals

openWakeWord has four high-level goals, which combine to (hopefully!) produce a framework that is simple to use *and* extend.

1) Be fast *enough* for real-world usage, while maintaining ease of use and development. For example, the models can run in real-time using ~x% of a single core on a Raspberry Pi3 (see the see the [Performance & Evaluation]() section for more details), but are likely too big for less powerful systems or microcontrollers. Commercial options like [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/) or [Fluent Wakeword](https://fluent.ai/products/wakeword/) are likely better suited for constrained hardware environments.

2) Be accurate *enough* for real-world usage. The models should have false-accept and false-reject rates below the annoyance threshold for the average user. This is obviously subjective, by a false-accept rate of <0.5 per hour and a false-reject rate of <5% is often reasonable in practice. See the [Performance & Evaluation]() section for details about how well the included models perform.

2) Have a simple model architecture and inference process. Models process a stream of audio data in 80 ms frames, and return a prediction for each frame indicating whether the wake word/phrase has been detected. All models also have a shared feature extraction backbone, so that each additional model only has a small impact to overall system complexity and resource requirements. See the [Model Architecture]() section for more details.

4) Require **little to no manual data collection** to train new models. The included models (see the [Pre-trained Models]() section for more details) were all trained with *100% synthetic* speech generated from text-to-speech models. Training new models is a simple as generating new clips for the target wake word/phrase and training a small head-model on top of of the frozen common feature extractor. See the [Training New Models]() section for more details.

Future releases of openWakeWord will aim to stay aligned with the goals, even when adding new functionality.

# Pre-Trained Models

openWakeWord comes with pre-trained models for common words & phrases. Currently, only English models are supported, but they should be reasonably robust across different types speaker accents and pronunciation.

Each model aims to have false-reject rates of less than 5% and false-accept rates of less than 0.5/hr on the evaluation data to be considered minimally viable (see [Performance & Evaluation]() for definitions and details). These levels are chosen so that, for example, a user could be expected to have a extended conversation of several hours with maybe a single false activation, and a failed activation only 1/20 attempts (and a failed *second* activation only 1/400 attempts).

The table below lists each model and the false-reject rate @ 0.5 false accepts/hr, along with the recommended threshold value to obtain that performance (though this should be adjusted to your use-case). See the individual [model pages]() in the documentation for the full false-reject/false-accept curves across different thresholds.

| Word/Phrase | False-Reject Rate | False-Accepts per Hour | Recommended Score Threshold |
| ------------- | ------------- | ------------- | ------------- |
| "alexa" | 0.05 | 0.5 | 0.9 |
| "hey mycroft" | 0.05 | 0.5 | 0.75 |
| "what's the weather"* | 0.04| 0.1 | 0.5 |


*Many models are trained on multiple variations of the same phrase. See the individual documentation for each model to see all supported variations

# Model Architecture

openWakeword models are composed of three separate components:

1) A model that computes [melspectrogram](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html) on the input audio data. For openWakeword, an ONNX implementation of Torch's melspectrogram function with fixed parameters is used to enable efficient performance across devices.

2) A shared feature extraction backbone model that converts melspectrogram inputs into general-purpose speech audio embeddings. This [model](https://arxiv.org/abs/2002.01322) is provided by [Google](https://tfhub.dev/google/speech_embedding/1) as a TFHub module under an [Apache-2.0](https://opensource.org/licenses/Apache-2.0) license. For openWakeWord, the model was re-implemented in native Tensorflow/PyTorch to separate out different functionality and allow for more control of architecture modifications compared to a TFHub module. The model itself is a relatively simple convoluational structure, and gains its strong performance from extensive pre-training on large amounts of data. This model is the core component of openWakeWord, and enables the [strong performance](#-Performance-and-Evaluation) that is seen even when training on fully-synthetic data.

3) A classification model that sits on top of the shared (and typically frozen) feature extraction model. This structure of this model is arbitrary, but in practice a simple fully-connected model works well.

# Performance and Evaluation

Evaluting wakeword models is difficult, and it is very difficult to assess how different models presented in papers or other projects will perform in-practice on two critical metrics: false-reject rates and false-accept rates.

A *false-reject* is when the model fails to detect an intended activation from the user.

A *false-accept* is when the model inadvertently activates when the user did not intend for it to do so.

For openWakeWord, evaluation follows two principles:
- The *false-reject* rate should be determined from wakeword/phrases that represent realistic recording environments such as those with background noise and reverberation. This can be accomplished by directly collected data from these environments, or simulating them with data augementation methods.

- The *false-accept* rate should be determined from audio that represents the types of environments that would be expected for the deployed model, not simply on the training/evaluation data. In practice, this means that the model should only rarely activate in error, even in the presence of hours of continuous speech.

While other wakeword evaluation standards [do exist](https://github.com/Picovoice/wake-word-benchmark), for openWakeWord it was decided that a more challenging evaluation would better indicate what users could expect in practice. Specifically:

1) *false-reject* rates are calculated from otherwise clean examples that are mixed with the HOME background noise from the DEMAND dataset at an SNR of 10 dB, and have simulated reverberation applied using the room-impulse-response functions from the [?] dataset. Any manually collected data is done in realistic environments that include background noise, varying microphone distances, and speaker direction.

2) *false-accept* rates are determined by using the [Dinner Party Corpus](https://www.amazon.science/publications/dipco-dinner-party-corpus) dataset. Representing ~5.5 hours of far-field speech, background music, and miscellaneous noise, this sets a realistic goal for how many false activations might occur in a similar situation.

To illustrate how openWakeWord can produce very capable models, the false-accept/false-reject curves for the included `"alexa"` model is shown below along with the performance of a strong commercial competitor, [Picovoice Porcupine](https://picovoice.ai/platform/porcupine/). Other existing open-source wakeword engines (e.g., [Snowboy](https://github.com/Kitt-AI/snowboy), [PocketSphinx](https://github.com/cmusphinx/pocketsphinx), etc.) are not included as they are either no longer maintained or demonstrate performance significantly below that that Porcupine. The positive test examples used were those included in [Picovoice's](https://github.com/Picovoice/wake-word-benchmark) repository, a fantastic resource that they have provided freely to the community.

- roc curve 1

As a second illustration, the false-accept/false-reject rate of the included `"hey-mycroft"` model is shown below along with the performance of Picovoice Porcupine and [Mycroft Precise](https://mycroft-ai.gitbook.io/docs/mycroft-technologies/precise). In this case, the positive test examples were manually collected from a male speaker with a relatively neutral American english accent in realistic home recording scenarios (e.g., a small room with a desk fan running, an unfinished basement with an echo, a kitchen with water running, etc.)

- roc curve 2 (for "hey mycroft")

If you are aware of another open-source wakeword/phrase library that should be added to this evaluation, or have suggestions on how to improve it, please open an issue! We are eager to continue improving openWakeWord by learning how others are approaching this problem.

# Training New Models

There new models is conceptually simple:

1) Generate new training data for the desired wakeword/phrase using open-source STT systems. For openWakeWord, extensive testing was done to determine which models produce synthetic speech that results in models with strong real-world performance. These models are provided in a [separate repository](). The number of generated examples require can vary, a minimum of several thousand is recommended and performance seems to increase smoothly with increasing dataset size.

2) Collect negative data (e.g., audio where the wakeword/phrase is not present) to help the model to have a low false-accept rate. This also benefits from scale, and the [included models](#-Pre-Trained-Models) where all trained with ~30,000 hours of negative data across speech, noice, and music. See the [Documentation]() for more details on the training data curation and preparation.

# Language Support

Currently, openWakeWord only supports English, primarily because the pre-trained text-to-speech models used to generate training data are all english. It's likely that speech-to-text models trained on other languages would also work well, but non-english models & datasets are less commonly available.

Future release roadmaps may have non-english support. In particular, [Mycroft.AIs Mimic 3](https://github.com/MycroftAI/mimic3-voices) TTS engine could work well to help extend support to other languages.