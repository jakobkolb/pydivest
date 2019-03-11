"""
This is the setup.py for the pydivest Model

for developers: recommended way of installing is to run in this directory
pip install -e .
This creates a link insteaed of copying the files, so modifications in this
directory are modifications in the installed package.
"""

from setuptools import setup

setup(name="pydivest",
      version="0.0.1",
      description="to be added",
      url="to be added",
      author="Jakob. J. Kolb",
      author_email="kolb@pik-potsdam.de",
      license="MIT",
      packages=["pydivest"],
      include_package_data=True,
      install_requires=[
            "numpy>=1.11.0",
            "pylama>=7.0.0",
            "pytest-cov>=2.5.1",
            "scipy",
            "pandas",
            "scipy",
            "networkx",
            'matplotlib',
            'Assimulo',
            'sympy',
            "pymofa @ git+https://github.com/jakobkolb/pymofa",
            'pydstool @ git+https://github.com/robclewley/pydstool'
      ],
      # see http://stackoverflow.com/questions/15869473/what-is-the-advantage-
      # of-setting-zip-safe-to-true-when-packaging-a-python-projec
      zip_safe=False)
