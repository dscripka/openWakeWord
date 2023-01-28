import platform
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Build extras_requires based on platform
def build_additional_requires():
    py_version = platform.python_version()[0:3].replace('.', "")
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        additional_requires=[
            f"speexdsp_ns @ https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp{py_version}-cp{py_version}-linux_x86_64.whl",
        ]
    elif platform.system() == "Linux" and platform.machine() == "aarch64":
        additional_requires=[
            f"speexdsp_ns @ https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp{py_version}-cp{py_version}-linux_aarch64.whl",
        ],
    elif platform.system() == "Windows" and platform.machine() == "x86_64":
        additional_requires=[
            'PyAudioWPatch'
        ]
    else:
        additional_requires = []

    return additional_requires

setuptools.setup(
    name="openwakeword",
    version="0.2.0",
    install_requires=['onnxruntime>=1.10.0,<2'],
    extras_require={
        'test': [
                    'pytest>=7.2.0,<8',
                    'pytest-cov>=2.10.1,<3',
                    'pytest-flake8>=1.1.1,<2',
                    'flake8>=4.0,<4.1',
                    'pytest-mypy>=0.10.0,<1'
                ],
        'full': [
                    'mutagen>=1.46.0,<2',
                    'speechbrain>=0.5.13,<1',
                    'pytest>=7.2.0,<8',
                    'pytest-cov>=2.10.1,<3',
                    'pytest-flake8>=1.1.1,<2',
                    'pytest-mypy>=0.10.0,<1',
                    'plotext>=5.2.7,<6',
                    'sounddevice>=0.4.1,<1'
                ]
    },
    author="David Scripka",
    author_email="david.scripka@gmail.com",
    description="An open-source audio wake word (or phrase) detection framework with a focus on performance and simplicity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/openwakeword",
    project_urls={
        "Bug Tracker": "https://pypi.org/project/openwakeword/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
)