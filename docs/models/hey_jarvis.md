# Model Description

A model trained to detect the presence of the phrase "hey jarvis" in an audio recording of speech.

Other similar phrases such as just "jarvis" or may also work, but likely with higher false-reject rates. Similarly, a short pause after the speaking the wakeword is recommended, but the model may also detect the presence of the wakeword is a continuous stream of speech in certain cases.

# Model Architecture

The model is composed of two parts two 3-layer fully connected networks that takes the flattened input features from the frozen audio embedding mode. ReLU activations and layer norms are inserted between the layers. A representative (but not exact) example of this structure is shown below.

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [1, 1]                    --
├─Flatten: 1-1                           [1, 1536]                 --
├─Linear: 1-2                            [1, 64]                   98,368
├─LayerNorm: 1-3                         [1, 64]                   128
├─ReLU: 1-4                              [1, 64]                   --
├─Linear: 1-5                            [1, 64]                   4,160
├─LayerNorm: 1-6                         [1, 64]                   128
├─ReLU: 1-7                              [1, 64]                   --
├─Linear: 1-8                            [1, 1]                    65
├─Sigmoid: 1-9                           [1, 1]                    --
==========================================================================================
Total params: 102,849
Trainable params: 102,849
Non-trainable params: 0
Total mult-adds (M): 0.10
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.00
Params size (MB): 0.41
Estimated Total Size (MB): 0.42
==========================================================================================
```

The first of the two networks is trained on all of the available data and is intended to predict on all audio frames. The second network is trained on a subset of the data (see details below) and is intended to act as a verifier model, only predicting on audio frames that have a score > 0.5 from the first model. These models are combined together prior to exporting to the final ONNX format, so this conditional routing is integrated directly into the model. From testing, this conditional verifier approach can reduce false accept rates without significantly impacting false rejection rates.

# Training Data

## Positive Data

The model was trained on approximately ~200,000 synthetically generated clips of the "hey jarvis" wake phrase using two text-to-speech (TTS) models:

1) [NVIDIA WAVEGLOW](https://github.com/NVIDIA/waveglow) with the LibriTTS multi-speaker model
2) [VITS](https://github.com/jaywalnut310/vits) with the VCTK multi-speaker model

Clips were generated both with the trained speaker embeddings, and also mixtures of individual speaker embeddings to produce novel voices. See the [Synthetic Data Generation](../synthetic_data_generation.md) documentation page for more details.

The following phrases were included in the training data:

1) "hey mycroft"

After generating the synthetic positive wakewords, they are augmented in two ways:

1) Mixing with clips from the ACAV100M, FSD50k, and Free Music Archive datasets referenced below at ratios of 0 to 20 dB
2) Reverberated with simulated room impulse response functions from the [BIRD Impulse Response Dataset](https://github.com/FrancoisGrondin/BIRD)

## Negative Data

The model was trained on approximately ~31,000 hours of negative data, with the approximate composition shown below:

1) ~10,000 hours of noise, music, and speech from the [ACAV100M dataset](https://acav100m.github.io/)
2) ~10,000 hours from the [Common Voice 11 dataset](https://commonvoice.mozilla.org/en/datasets), representing multiple languages
3) ~10,000 hours of podcasts downloaded from the [Podcastindex database](https://podcastindex.org/)
4) ~1,000 hours of music from the [Free Music Archive dataset](https://github.com/mdeff/fma)

In addition to the above, the total negative dataset also includes reverberated versions of the ACAV100M dataset (also using the simulated room impulse responses from the [BIRD Impulse Response Dataset](https://github.com/FrancoisGrondin/BIRD) dataset). Adversarial synthetic generations designed to be phonetically similar to the wakeword (e.g., "hey jealous") were also added to the dataset.

For the second stage verifier model, a subset of negative data was selected from the ACAV100M dataset, adversarial synthetic generations, and manually collected false-activations from additional sources not included in the training dataset (e.g., GigaSpeech, additional podcast downloads, etc.). This was done to (hopefully) focus this model on the challenging examples more likely to result in false activations.

# Test Data

The positive test examples of the "hey jarvis" wakeword were collected manually in a realistic home environment from both near and far-field microphones, at distances ranging from ~3 to ~30 feet. The (male) speaker has a relatively neutral American english accent, and the recordings were captured with normal background noise included fans/air conditioning and a running dishwasher in a kitchen.

# Performance

Currently, there is not a test set available to evaluate this model.

# Other Considerations

While the model was trained to be robust to background noise and reverberation, it will still perform the best when the audio is relatively clean and free of overly loud background noise. In particular, the presence of audio playback of music/speech from the same device that is capturing the microphone stream may result in significantly higher false-reject rates unless acoustic echo cancellation (AEC) is performed via hardware or software.