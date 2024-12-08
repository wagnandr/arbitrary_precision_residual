import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

setup(
    name="hpresidual",
    version="0.0.1",
    description="Calculates the residual in high precision",
    author="Andreas Wagner",
    license="boost software license 1.0",
    packages=find_packages(where="python/src"),
    package_dir={"": "python/src"},
    cmake_install_dir="python/src",
    cmake_args=[], # here we have '-D...=...'
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    python_requires=">=3.6",
)    