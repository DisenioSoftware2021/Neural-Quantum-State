#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# In[10]:


class Hamiltonian:
    def __init__(self, omega, include_interaction):

        self.include_interaction = include_interaction
        self.omega = omega

    def potential_oscillator(self, x):
        # The function calculates the potential energy of the harmonic
        # oscillator of the system  for a given configuration of the
        # particles of the system

        self.harmonic_oscillator = np.dot(self.omega * self.omega * x, x)
        return self.harmonic_oscillator

    def local_energy(self, nqs):
        # The function computes the energy of the system described by
        # the wave function of the NQS object, and according to the
        # type of Hamiltonian you choose

        harmonic_oscillator = self.potential_oscillator(nqs.x)
        # self.harmonicosc
        kinetic = nqs.laplacian(nqs.x)

        local_energy = 0.5 * (kinetic + harmonic_oscillator)

        if self.include_interaction == "harmonic_oscillator":
            local_energy = local_energy
        elif self.include_interaction == "coulomb":
            local_energy += np.sum(nqs.inverse_distance(nqs.x))
        elif self.include_interaction == "calogero":
            local_energy += nqs.calogero(nqs.x)
        else:
            print("error")
            exit
        return local_energy



