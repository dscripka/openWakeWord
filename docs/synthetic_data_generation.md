# Synthetic Data Generation

The use of synthetic data for training STT or wakeword/phrase detection models is not a new concept, and in particular the inspiration for openWakeWord was motivated by two specific papers:

1) [Speech Model Pre-training for End-to-End Spoken Language Understanding](https://arxiv.org/abs/1904.03670)
2) [Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models](https://arxiv.org/abs/1910.09463)

In general, the concept of pre-training a model on large speech datasets and then fine-tuning another smaller model on top of this (typically frozen) backbone with use-case specific data is a well-documented approach more broadly that seems to work well for many different applications.

# Choosing TTS Models

During the development of openWakeWord, much effort went into identifying STT models that could produce high-quality speech to use as training data. In particular, two features are assumed to be important to produce robust wakeword models:

1) Random variability in the generated speech (in practice, models based on sampling work well)
2) Multi-speaker models

According to these criteria, the two models chosen as the foundation for openWakeWord model training are [NVIDIA WAVEGLOW](https://github.com/NVIDIA/waveglow) and [VITS](https://github.com/jaywalnut310/vits). The authors and publishers of these models deserve credit for releasing these high quality models to the community.

# Increasing Diversity in Generated Speech

Beyond the inherent ability of Waveglow and VITS to produce variable speech, they both also have hyper-parameters that can be adjusted to control this effect to some extent. A forthcoming repository dedicated to dataset generation will provide more details on this, but in brief:

1) Relatively high values are used for sampling parameters (which results in more variation in the generated speech) even if this causes low quality or incorrect generations some small percentage of the time.

2) To go beyond the original number of speakers used in multi-speaker datasets, [spherical interpolation](https://en.wikipedia.org/wiki/Slerp) of speaker embeddings is used to produce mixtures of different voices to extend beyond the original training set. While this occasionally results in lower quality generations (in particular a gravely texture to the speech), again the benefits of increased generation diversity seem to be more important for the trained openWakeWord models.