# Model Description

A model trained to detect the presence of several different phrases all related to creating a timer or alarm for six common durations: 1 minute, 5 minutes, 10 minutes, 20 minutes, 30 minutes, and 1 hour. It is a multi-class model (e.g., each class will have a score between 0 and 1), indicating how likely a given segment of speech is to contain a phrase setting a timer/alarm for the given duration.

As with other models, similar phrases beyond those included in the training data may also work, but likely with higher false-reject rates. Similarly, a short pause after the speaking the wake phrase is recommended, but the model may also detect the presence of the wake phrase is a continuous stream of speech in certain cases.

# Training Data

## Positive Data

The model was trained on approximately ~100,000 synthetically generated clips of the timer/alarm wake phrases using two text-to-speech (TTS) models:

1) [NVIDIA WAVEGLOW](https://github.com/NVIDIA/waveglow) with the LibriTTS multi-speaker model
2) [VITS](https://github.com/jaywalnut310/vits) with the VCTK multi-speaker model

Clips were generated both with the trained speaker embeddings, and also mixtures of individual speaker embeddings to produce novel voices. See the [Synthetic Data Generation](../synthetic_data_generation.md) documentation page for more details.

The following phrases were included in the training data (where x represents the duration, and words in brackets represent possible slot insertions):

- "[create/set/start] [a/NONE] x [minutes/hour] [alarm/timer]"
- "[create/set/start] [an/a/NONE] [alarm/timer] for x [minutes/hour]"

As an example, here are several of the permutations from the structure above that were included in the training data:

- "set an alarm for 10 minutes"
- "start a 1 hour timer"
- "create timer for 5 minutes"

After generating the synthetic positive wake phrases, they are augmented in two ways:

1) Mixing with clips from the ACAV100M dataset referenced below at ratios of 0 to 30 dB
2) Reverberated with simulated room impulse response functions from the [BIRD Impulse Response Dataset](https://github.com/FrancoisGrondin/BIRD)

## Negative Data

The model was trained on approximately ~31,000 hours of negative data, with the approximate composition shown below:

1) ~10,000 hours of noise, music, and speech from the [ACAV100M dataset](https://acav100m.github.io/)
2) ~10,000 hours from the [Common Voice 11 dataset](https://commonvoice.mozilla.org/en/datasets), representing multiple languages
3) ~10,000 hours of podcasts downloaded from the [Podcastindex database](https://podcastindex.org/)
4) ~1,000 hours of music from the [Free Music Archive dataset](https://github.com/mdeff/fma)

In addition to the above, the total negative dataset also includes reverberated versions of the ACAV100M dataset (also using the simulated room impulse responses from the [BIRD Impulse Response Dataset](https://github.com/FrancoisGrondin/BIRD) dataset). Currently, adversarial STT generations were not added to the training data for this model.

# Test Data

Currently, there is not a test set available to evaluate this model.

# Performance

Due to similar training datasets and methods it is assumed to have similar performance compared to other pretrained models (e.g., <5% false-reject rates and <0.5 false-accepts per hour).

# Other Considerations

While the model was trained to be robust to background noise and reverberation, it will still perform the best when the audio is relativey clean and free of overly loud background noise. In particular, the presence of audio playback of music/speech from the same device that is capturing the microphone stream may result in significantly higher false-reject rates unless acoustic echo cancellation (AEC) is performed via hardware or software.