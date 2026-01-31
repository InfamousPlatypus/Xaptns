from setuptools import setup, find_packages

setup(
    name="xaptns",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "arxiv",
        "requests",
        "click",
        "transformers",
        "torch",
        "openvino",
        "usearch",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "xaptns=xaptns.cli:cli",
        ],
    },
)
