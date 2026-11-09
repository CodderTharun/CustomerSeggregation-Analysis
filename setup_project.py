from pathlib import Path

directories = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",

    "src/data",
    "src/preprocessing",
    "src/analysis",
    "src/models",
    "src/segmentation",
    "src/visualization",
    "src/business_insights",
    "src/utils",

    "models",
    "reports/figures",
    "tests",

    ".streamlit"
]

files = [
    "app.py",
    "config.py",
    "README.md",
    ".gitignore",

    "src/data/__init__.py",
    "src/preprocessing/__init__.py",
    "src/analysis/__init__.py",
    "src/models/__init__.py",
    "src/segmentation/__init__.py",
    "src/visualization/__init__.py",
    "src/business_insights/__init__.py",
    "src/utils/__init__.py",

    "src/data/data_loader.py",
    "src/data/data_cleaner.py",

    "src/preprocessing/feature_engineering.py",
    "src/preprocessing/feature_scaling.py",

    "src/analysis/eda.py",
    "src/analysis/rfm_analysis.py",

    "src/models/kmeans.py",
    "src/models/model_evaluation.py",

    "src/segmentation/segment_classifier.py",

    "src/visualization/charts.py",

    "src/business_insights/recommendations.py",

    "src/utils/helpers.py",
    "src/utils/logger.py",

    "tests/test_data_cleaning.py",
    "tests/test_segmentation.py",

    ".streamlit/config.toml"
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch(exist_ok=True)

print("Project structure created successfully!")