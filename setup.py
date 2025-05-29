# setup.py

from setuptools import setup, find_packages

setup(
    name="coverage_planning",
    version="0.1.0",
    author="Kushagra Bharti",
    description="Exact and greedy algorithms for 1D drone-coverage planning",
    packages=find_packages(exclude=["tests*", "benchmarks*", "examples*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
    ],
)
