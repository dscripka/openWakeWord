// Registry of the pre-trained openWakeWord models and their class mappings.
// Mirrors `openwakeword/__init__.py` from the Python package.

export const FEATURE_MODELS = {
  melspectrogram: "melspectrogram.onnx",
  embedding: "embedding_model.onnx",
};

export const VAD_MODELS = {
  silero_vad: "silero_vad.onnx",
};

// name -> onnx filename (relative to the models directory / base URL)
export const PRETRAINED_MODELS = {
  alexa: "alexa_v0.1.onnx",
  hey_mycroft: "hey_mycroft_v0.1.onnx",
  hey_jarvis: "hey_jarvis_v0.1.onnx",
  hey_rhasspy: "hey_rhasspy_v0.1.onnx",
  timer: "timer_v0.1.onnx",
  weather: "weather_v0.1.onnx",
};

// Integer-class -> label mappings for multi-class models.
export const MODEL_CLASS_MAPPINGS = {
  timer: {
    1: "1_minute_timer",
    2: "5_minute_timer",
    3: "10_minute_timer",
    4: "20_minute_timer",
    5: "30_minute_timer",
    6: "1_hour_timer",
  },
};
