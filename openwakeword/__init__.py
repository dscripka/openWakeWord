import os
from openwakeword.model import Model

__all__ = ['Model', ]

models = {
    "alexa": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/alexa_v5.onnx")
    },
    "hey_mycroft": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_mycroft_v1.onnx")
    },
    "timer": {
        "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/timer_v1.onnx")
    }
}


def get_pretrained_model_paths():
    return [models[i]["model_path"] for i in models.keys()]
