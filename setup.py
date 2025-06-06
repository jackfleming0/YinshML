from setuptools import setup, find_packages

setup(
    name="yinsh_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "torch>=2.1.0",
        "coremltools>=6.3.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
        "pandas>=2.0.3",
        "pytest>=7.4.0",
        "click>=8.0.0",
    ],
    entry_points={
        'console_scripts': [
            'yinsh-track=yinsh_ml.cli.main:cli',
        ],
    },
    python_requires=">=3.9",
)