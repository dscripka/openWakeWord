# Custom Verifier Models

If the performance of a trained openWakeWord model is not sufficient in a production application, training a custom verifier model on a particular speaker or set of speakers can help significantly the performance of the system. A custom verifier model acts as a filter on top of the base openWakeWord model, determining whether a given activation was likely from a known target speaker. In particular, this can be a very effective way at reducing false activations, as the model will be more focused on a the target speaker instead of attempting to activate for any speaker.

There are trade-offs to this approach, however. In general, training a custom verifier model can be beneficial with two assumptions:

1) It is feasible to collect the training data required to build a custom model for all of the desired users of the system. The training requirements are minimal (likely <5 minutes of effort), but needs to be repeated for every user.

2) The range of acoustic environments seen in production are similar enough to that observed during collection of the user-specific data. If there are singicant differences across deployment acoustic environments, custom models will need to be trained for each one.

# Verifier Model Design

The custom verifier models are designed to be very lightweight and easy to train. For the current version of openWakeWord, the verifier models are simple logistic regression binary classifiers the take in the shared audio features from the openWakeWord preprocessing stage and returns a score between 0 and 1 indicating whether the audio contains a wakeword or phrase spoken by the target speaker. Because this task in inherently much more narrow compared to the detecting the wakeword or phrase from any speaker, the combination of the verifier model and base model can be quite effective.

Note that while the verifier model is focused on a target speaker, it is not intended to perform the task of speaker verification directly. Performance on this task may be adequate for certain use-case cases, but caution is recommended.

# Verifier Model Training

Training a custom verifier model is conceptually simple, and only requires a very small amount of training data. Recommendations for training data collection are listed below.

- Positive data (examples of wakeword or phrase)
    - Collect a minimum of 3 examples for each target speaker
    - Positive examples should be as close as possible to the expected deployment scenario, including some level of background noise if that is appropriate
    - The capacity of the verifier model is small, it's not advised to train on a large number of positive examples or for more than a few speakers

- Negative data collection
    - Collect a minimum of ~10 seconds of speech from each target speaker that does not contain the wakeword, trying to include as much variation as possible in the speech
    - Optionally, collect ~5 seconds clips of typical background audio in the deployment environment or use previously collected examples of false activations (this is one of the most effective ways to reduce false activations)
    - The capacity of the verifier model is small, it's not advised to train on a very large number of negative examples as the verifier model should be focused just on the deployment environment and user(s)

After collected the positive and negative examples, a custom verifier model can be trained with the `openwakeword.train_custom_verifier` function:

```python
openwakeword.train_custom_verifier(
    positive_reference_clips = ["positive_clip1.wav", "positive_clip2.wav", "positive_clip3.wav"]
    negative_reference_clips = ["negative_clip1.wav", "negative_clip2.wav"]
    output_path = "path/to/directory/model.pkl"
    model_name = "hey_jarvis.onnx" # the target model path which matches the wake word/phrase of the collected positive examples
)
```

After training a model and saving it, an openWakeWord instance can be created with the verifier model which will be called whenever the base openWakeWord model makes a prediction with a score above the specified threshold, and the returned score will be the one from the verifier model.

```python
oww = openwakeword.Model(
    custom_verifier_models={"hey_jarvis": "path_to_verifier_model.pkl"},
    custom_verifier_threshold=0.3, # the threshold score required to invoke the verifier model
)
```