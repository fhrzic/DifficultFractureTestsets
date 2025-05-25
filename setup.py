# setup.py
from setuptools import setup, find_packages

setup(
    name="effnet_eval",
    version="0.1.0",
    description="Code for the paper: Artificial Intelligence Test Set Performance in Difficult Cases Matched with Routine Cases of Pediatric Appendicular Skeleton Fractures",
    author="Your Name",
    author_email="franko.hrzic@uniri.hr",
    packages=find_packages(),  # Finds effnet_eval automatically
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tqdm",
        "xlsxwriter",
        "ultralytics",
        "opencv-python",
        "fastai",
        "efficientnet-pytorch",
        "torch",               
        "timm"                 
    ],
    entry_points={
        'console_scripts': [
            'effnet-eval = effnet_eval.main:main',
        ],
    },
    python_requires=">=3.8",
)