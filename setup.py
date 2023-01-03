# +
from __future__ import annotations

from setuptools import find_packages, setup

__version__ = "0.1.0"


setup(
    name="wurtzite",
    version=__version__,
    author="Amir Hajibabaei",
    author_email="a.hajibabaei.86@gmail.com",
    description="atomic surfaces",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy", "scipy", "ase"],
    url="https://github.com/amirhajibabaei/wurtzite",
    license="MIT",
)
