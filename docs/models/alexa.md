# Model Description

A model trained to detect the presence of the spoken word "Alexa" in an audio recording

Other similar phrases such has "Hey Alexa" or "Alexa stop" may also work, but likely with higher false-reject rates. Similarly, a short pause after the speaking the wakeword is recommended, but the model may also detect the presence of the wakeword is a continuous stream of speech in certain cases.

# Training Data

## Positive Data

The model was trained on approximately ~100,000 synthetically generated clips of the "Alexa" wakeword using two text-to-speech (TTS) models:

1) [NVIDIA WAVEGLOW](https://github.com/NVIDIA/waveglow) with the LibriTTS multi-speaker model
2) [VITS](https://github.com/jaywalnut310/vits) with the VCTK multi-speaker model

Clips were generated both with the trained speakers, and also mixtures of individual speaker embeddings to produce novel voices. See the [Synthetic Data Generation]() documentation page for more details.

The following phrases were included in the training data, all representing cases where the system is trained to detect the presence of the wakeword:

1) "Alexa"
2) "Alexa `<random words>`"

After generating the synthetic positive wakewords, they are augmented in two ways:

1) Mixing with clips from the ACAV100M dataset referenced below at ratios of 0 to 30 dB
2) Reverberated with simulated room impulse response functions from the [BIRD Impulse Response Dataset](https://github.com/FrancoisGrondin/BIRD)

## Negative Data

The model was trained on approximately ~31,000 hours of negative data, with the approximate composition shown below:

1) ~10,000 hours of noise, music, and speech from the [ACAV100M dataset](https://acav100m.github.io/)
2) ~10,000 hours from the [Common Voice 11 dataset](https://commonvoice.mozilla.org/en/datasets), representing multiple languages
3) ~10,000 hours of podcasts downloaded from the [Podcastindex database](https://podcastindex.org/)
4) ~1,000 hours of music from the [Free Music Archive dataset](https://github.com/mdeff/fma)

In addition to the above, the total negative dataset also includes reverberated versions of the ACAV100M dataset (also using the simulated room impulse responses from the [BIRD Impulse Response Dataset](https://github.com/FrancoisGrondin/BIRD) dataset), and adversarial synthetic generations designed to be phonetically similar to the wakeword (e.g., "annex uh").

# Test Data

The positive test examples of the "Alexa" wakeword are those included in [Picovoice's](https://github.com/Picovoice/wake-word-benchmark) repository. This examples are mixed with the HOME background noise from the [DEMAND](https://zenodo.org/record/1227121#.Y3OSG77MJhE) dataset at an SNR of 10 dB, and have simulated reverberation applied using the real room-impulse-response functions from the [Room Impulse Response and Noise](https://www.openslr.org/28/) dataset.

# Performance

The false-accept/false-reject curve for the model on the test data is shown below. Decreasing the `threshold` parameter when using the model will increase the false-accept rate and decrease the false-reject rate.

# Other Considerations

While the model was trained to be robust to background noise and reverberation, it will still perform the best when the audio is relativey clean and free of overly loud background noise. In particular, the presence of audio playback of music/speech from the same device that is capturing the microphone stream may result in significantly higher false-reject rates unless acoustic echo cancellation (AEC) is performed via hardware or software.