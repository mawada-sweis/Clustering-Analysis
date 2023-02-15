from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

analysis_packages = ["jupyterlab==3.4.6"]

# Define our package
setup(
    name="Citywide Mobility Clustering Analysis",
    version=1.0,
    description="Assess the travel behavior, preferences, and attitudes of residents of NYC",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": analysis_packages
    },
)
