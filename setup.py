import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="openwakeword",
    version="0.1.0",
    install_requires=[
        'onnxruntime>=1.10.0,<2'
    ],
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
    python_requires=">=3.6",
)