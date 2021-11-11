#!/usr/bin/env python
# coding: utf-8

# In[1]:


import attr
import numpy as np
from attr import validators as vldts

# In[10]:


@attr.s
class Hamiltonian:
    omega = attr.ib(validator=vldts.instance_of(float))
    include_interaction = attr.ib(validator=vldts.instance_of(str))

    def local_energy(self, nqs):
        # The function computes the energy of the system described by
        # the wave function of the NQS object, and according to the
        # type of Hamiltonian you choose

        harmonic_oscillator = np.dot(self.omega * self.omega * nqs.visible_values_, nqs.visible_values_)
        # harmonic_oscillator = self.potential_oscillator(nqs.visible_values_)
        # self.harmonicosc
        kinetic = nqs.laplacian(nqs.visible_values_)

        local_energy = 0.5 * (kinetic + harmonic_oscillator)

        if self.include_interaction == "harmonic_oscillator":
            local_energy = local_energy
        elif self.include_interaction == "coulomb":
            local_energy += np.sum(nqs.inverse_distance(nqs.visible_values_))
        elif self.include_interaction == "calogero":
            local_energy += nqs.calogero(nqs.visible_values_)
        # else:
        #     print("error")
        #     exit
        else:
            raise AssertionError("error")
        return local_energy
