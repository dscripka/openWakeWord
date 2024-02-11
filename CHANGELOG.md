# Change Log

## v0.6.0 - 2023/06/15

### Added

* Various bug fixes, and some new functionality in `model.py` to control repeated detections

### Changed

* Models are no longer included in the PyPi package, and must be downloaded separately

### Removed

## v0.5.0 - 2023/06/15

### Added

* A new wakeword model, "hey rhasspy"
* Added support for tflite versions of the melspectrogram model, embedding model, and pre-trained wakeword models
* Added an inference framework argument to allow users to select either ONNX or tflite as the inference framework
* The `detect_from_microphone.py` example now supports additional arguments and has improved console formatting

### Changed

* Made tflite the default inference framework for linux platforms due to improved efficiency, with windows still using ONNX as the default given the lack of pre-built Windows WHLs for the tflite runtime (https://pypi.org/project/tflite/)
* Adjusted the default provider arguments for onnx models to avoid warnings (https://github.com/dscripka/openWakeWord/issues/27)

### Removed