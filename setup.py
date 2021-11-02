#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NQS Project (https://github.com/DisenioSoftware2021/Neural-Quantum-State).
# Copyright (c) 2021, Facundo Serrano
# License: MIT
#   Full Text:          <----------- OJO CON ESTO, AGREGAR PATH DE LA LICENCIA

# =====================================================================
# DOCS
# =====================================================================

"""This file is for distribute and install NQS"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import pathlib


from setuptools import setup  # noqa

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


REQUIREMENTS = ["numpy", "attrs"]# <----------------- CAMBIOS

with open(PATH / "NQS" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break


with open("README.md") as fp:
    LONG_DESCRIPTION = fp.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="NQS",
    version=VERSION,
    description="Represents a quantum state using the restricted Boltzmann machine",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Facundo Serrano",
    author_email="facundo.serrano@mi.unc.edu.ar",
    url="https://github.com/DisenioSoftware2021/Neural-Quantum-State",
    packages=[
        "NQS_1",
    ],
    license="The MIT License",# <-------------------- CONSULTAR
    install_requires=REQUIREMENTS,
    keywords=["Quantum State", "Machine Learning", "Restricted Boltzmann Machine"],
    classifiers=[
        "Development Status :: 4 - Beta",#         <----------- OJO CON ESTO, CAMBIAR
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
)
