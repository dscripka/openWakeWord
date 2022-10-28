# openWakeWord

openWakeWord is a fully open-source wakeword library that can be used to create voice-enabled applications and interfaces. It comes with a (growing!) set of pre-trained models, and new models can be easily trained as well for different wake words & phrases. The overall goal of the library is to provide a simple framework for wakeword/phrase detection, while also providing pre-built models (and the ability to train new models) that perform well enough to be useable in the real-word.

More specifically, openWakeWord aims to:

1) Be fast & accurate *enough* for real-world usage. The models can easily run in real-time using ~x% of a single core on a Raspberry Pi3 (see the see the [Performance & Evaluation]() section for more details), but are likely too big for less powerful systems or microcontrollers. The models should have false-accept and false-reject rates of that are below the annoyance threshold for the average user. This is obviously subjective, by a false-accept rate of <0.5 per hour and a false-reject rate of <5% seems reasonable in practice.

2) Have a simple interface for model inference. Models process a stream of audio data in 80 ms frames, and return a prediction for each frame indicated whether the wake word/phrase has been detected.

3) Have a shared feature extraction backbone for all models so that many separate models can be run with minimal additional resource requirements. See the [Model Architecture]() section for more details.

4) Require *little to no manual data collection* to train new models. The included models (see the [Pre-trained Models]() section for more details) were all trained with *100% synthetic* speech generated from text-to-speech models. Training new models is a simple as generating new clips for the target wake word/phrase and training a small head-model on top of of the frozen common backbone. See the [Training New Models]() section for more details.

# Pre-Trained Models

# Model Architecture

# Performance and Evaluation

- Mention 0.5/hour false accept rate for near continuous speech (e.g., dinner party corpus)
- False-reject rate of 5% means that the chanced of missing two activations is only 0.25%. E.g., if a user on average intentially speaks a wake word/phrase 20 times per day, they would expect to have to try two times once per day, and try three times only once every 20 days (assuming the failed activations aren't correlated and the environmental conditions are such that an activation would otherwise be expected).


# Training New Models

# Language Support

Currently, openWakeWord only supports English, primarily because the pre-trained text-to-speech models used to generate training data are all english. It's likely that speech-to-text models trained on other languages would also work well, but non-english models & datasets are less commonly available.

Future release roadmaps may have non-english support. In particular, [Mycroft.AIs Mimic 3](https://github.com/MycroftAI/mimic3-voices) TTS engine could work well.