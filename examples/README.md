# Examples

Included are several example scripts demonstrating the usage of openWakeWord. Some of these examples have specific requirements, which are detailed below.

## Detect From Microphone

This is a simple example which allows you to test openWakeWord by using a locally connected microphone. To run the script, follow these steps:

1) Install the example-specific requirements: `pip install plotext pyaudio`

2) Run the script: `python detect_from_microphone.py`.

Note that if you have more than one microphone connected to your system, you may need to adjust the PyAudio configuration in the script to select the appropriate input device.

## Benchmark Efficiency

This is a script that estimates how many openWakeWord models could be run on on the specified number of cores for the current system. Can be useful to determine if a given system has the resources required for a particular use-case.

To run the script: `python benchmark_efficiency.py --ncores <desired integer number of cores>`