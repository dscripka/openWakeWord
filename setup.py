import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openwakeword",
    version="0.0.1",
    install_requires=[
        'onnxruntime>=1.10.0,<2'
    ],
    author="David Scripka",
    author_email="david.scripka@gmail.com",
    description="An open-source audio wake word (or phrase) detection framework with a focus on performance and simplicity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)