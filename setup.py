from setuptools import setup, find_packages

setup(
    name="falt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "praat-textgrids",
        "torch",
        "torchaudio",
        "tqdm",
        "transformers",
    ],
    python_requires=">=3.8",
    author="Gaofei Shen",
    author_email="g.shen@tilburguniversity.edu",
    description="Forced Alignment Toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/techsword/forced-alignment-toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)