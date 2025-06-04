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
            'HCE-trainYOLO = Scripts.YOLO.train_YOLO:main',
            'HCE-testYOLO = Scripts.YOLO.test_YOLO:main',
            'HCE-CVATtoYOLO = Scripts.YOLO.CVAT_to_YOLO:main',
            'HCE-CaseMatching = Scripts.DataFiltering.CaseMatching:main',
            'HCE-evalEffNet = Scripts.EfficientNet.efficientnet_eval_report:main',
            'HCE-testEffNet = Scripts.EfficientNet.efficientnet_test_eval:main',
            'HCE-trainEffNet = Scripts.EfficientNet.efficientnet_train_fromcsv:main',
            'HCE-gradcamEffNet = Scripts.EfficientNet.gradcam_efficientnet:main',
            'HCE-evalYOLOMetrics = Scripts.StatisticsAndResults.yolov8_pr_curve_eval:main',
            'HCE-evalEffNetMetrics = Scripts.StatisticsAndResults.effnet_eval.main:main',
        ],
    },
    python_requires=">=3.8",
)