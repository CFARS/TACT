from setuptools import setup, find_packages

setup(
    name="tact",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
    ],
    author="Cameron Planck",
    description="Turbulence Adjustment Computation Tool",
    python_requires=">=3.7",
) 