#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   NQS Project (https://github.com/DisenioSoftware2021/Neural-Quantum-State).
# Copyright (c) 2021, Facundo Serrano
# License: MIT
#   Full Text: https://github.com/milicolazo/Pyedra/blob/master/LICENSE         <----------- OJO CON ESTO, CAMBIAR

# =============================================================================
# DOCS
# =============================================================================

"""
Neural-Quantum-State.

Represents a quantum state using the restricted Boltzmann machine.
"""

# =============================================================================
# META
# =============================================================================

__version__ = "0.1.0"


# =============================================================================
# IMPORTS
# =============================================================================

from .main import *  # noqa

from .condprobability1 import *  # noqa

from .gradient import *  # noqa

from .hamiltonian import *  # noqa

from .mcmethod import *  # noqa

from .nqs import *  # noqa

from .quantumodel1 import *  # noqa

from .trainer import *  # noqa
