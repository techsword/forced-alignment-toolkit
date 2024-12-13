from setuptools import setup, find_packages

setup(
    name="falt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.3",
        "praat-textgrids>=1.4.0",
        "torch>=2.5.1",
        "torchaudio>=2.5.1",
        "tqdm>=4.67.1",
        "transformers>=4.47.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Forced Alignment Toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/falt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)