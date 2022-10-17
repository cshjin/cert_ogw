from setuptools import find_packages, setup

setup(name="cert_ogw",
      version="0.0.1",
      author="Hongwei Jin",
      summary="TBD",
      license="MIT",
      author_email="hjin25@uic.edu",
      packages=find_packages(exclude=["tests", "results", "log", "cert_ogw.egg-info"]))
