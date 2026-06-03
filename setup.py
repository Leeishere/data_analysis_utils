from setuptools import find_packages, setup

setup(
    name="data-analysis-utils",
    version="0.1.0",
    description="Reusable statistical data analysis helpers and quickstart materials.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "seaborn",
        "scikit-learn",
        "hdbscan",
        "streamlit",
    ],
    package_data={"data_analysis_utils": ["data/*.csv"]},
    extras_require={
        "app": ["streamlit"],
        "dev": ["ipykernel", "jupyter"],
    },
)
