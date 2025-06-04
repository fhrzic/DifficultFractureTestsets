# setup.py
from setuptools import setup, find_packages

setup(
    name="DifficultFractureTestsets",
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
        "timm",
        "rapidfuzz",
        "scikit-image"        
    ],
    entry_points={
        'console_scripts': [
            'DFT-trainYOLO = Scripts.YOLO.train_YOLO:main',
            'DFT-testYOLO = Scripts.YOLO.test_YOLO:main',
            'DFT-CVATtoYOLO = Scripts.YOLO.CVAT_to_YOLO:main',
            'DFT-CaseMatching = Scripts.DataFiltering.CaseMatching:main',
            'DFT-evalEffNet = Scripts.EfficientNet.efficientnet_eval_report:main',
            'DFT-testEffNet = Scripts.EfficientNet.efficientnet_test_eval:main',
            'DFT-trainEffNet = Scripts.EfficientNet.efficientnet_train_fromcsv:main',
            'DFT-gradcamEffNet = Scripts.EfficientNet.gradcam_efficientnet:main',
            'DFT-evalYOLOMetrics = Scripts.StatisticsAndResults.yolov8_pr_curve_eval:main',
            'DFT-evalEffNetMetrics = Scripts.StatisticsAndResults.effnet_eval.main:main',
        ],
    },
    python_requires=">=3.8",
)