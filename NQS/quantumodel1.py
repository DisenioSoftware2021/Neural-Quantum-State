#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import NQS
import hamiltonian as hm
import Probcondicional1 as cond


# In[3]:


class Quantummodel:
    def __init__(self, nqs, ham, condi):

        self.nqs = nqs
        self.ham = ham
        self.condi = condi

        self.n_dim = nqs.n_dim
        self.n_hidden = nqs.n_hidden
        self.n_visible = nqs.n_visible
        self.a = nqs.a
        self.w = nqs.w
        self.b = nqs.b
        self.x = condi.x

        self.parameters = (
            self.n_visible + self.n_hidden + self.n_visible * self.n_hidden
        )
        self.sig = nqs.sig

        self.energy_gradient = np.zeros(self.parameters)

    def zzero(self):
        # The function initializes variables that I am going to add from zero.

        self.local_energy = 0
        self.local_energy_squared = 0
        self.accept_count = 0

        self.dPsi1 = np.zeros(self.parameters)
        self.energy_dPsi = np.zeros(self.parameters)

    def setupsampling(self, x):
        # The function sets up the model for a Monte Carlo simulation.

        self.Qsigm = self.nqs.sigmoid(x)

        self.psi = self.nqs.psi(x)
        self.loc_energy = self.ham.LocalEnergy(self.nqs)

        self.lap = self.nqs.laplacianalfa(x)

        return self.psi, self.loc_energy, self.lap

    def acumulador(self, x):

        self.lap = self.nqs.laplacianalfa(x)
        self.loc_energy = self.ham.LocalEnergy(self.nqs)

        self.local_energy += self.locenergy
        self.local_energy_squared += self.locenergy * self.locenergy

        self.dPsi1 += self.lap
        self.energy_dPsi += self.lap * self.loc_energy

        return (
            self.local_energy,
            self.local_energy_squared,
            self.dPsi1,
            self.energy_dPsi,
        )

    def valormedio(self, nro_samples):
        self.local_energy = self.local_energy / nro_samples
        self.local_energy_squared = self.local_energy_squared / nro_samples
        # self.acceptcount =self.acceptcount/nrosamples
        self.dPsi1 = self.dPsi1 / nro_samples
        self.energy_dPsi = self.energy_dPsi / nro_samples

        self.var = (
            self.local_energy_squared - self.local_energy * self.local_energy
        ) / nro_samples

        self.loc_en_gradient = 2 * (self.energy_dPsi - self.local_energy * self.dPsi1)
        self.loc_en_gradient_norm = np.sqrt(
            np.dot(self.loc_en_gradient, self.loc_en_gradient)
        )

    def one_bd(self, x, r_min, r_max, bin_width, one_bd, ratio):
        for p in range(0, self.n_visible, self.n_dim):
            r = 0
            for d in range(0, self.n_dim):
                r += x[p + d] * x[p + d]
            r = np.sqrt(r)
            if r_min <= r and r < r_max:
                bin_index = int(np.floor((r - r_min) / bin_width))
                one_bd[bin_index] += 1
                ratio[bin_index] = r

    def shiftparameters(self, shift):
        # The function updates the network parameters by adding a given shift.
        # It is used by the gradient descent method.
        for i in range(0, self.n_visible):
            self.a[i] = self.a[i] + shift[i]
        for j in range(0, self.n_hidden):
            self.b[j] = self.b[j] + shift[self.n_visible + j]
        k = self.n_visible + self.n_hidden
        for i in range(0, self.n_visible):
            for j in range(0, self.n_hidden):
                self.w[i, j] = self.w[i, j] + shift[k]
                k = k + 1

    def newparameters(self, best):
        for i in range(0, self.n_visible):
            self.a[i] = best[i]
        for j in range(0, self.n_hidden):
            self.b[j] = best[j + self.n_visible]
        k = self.n_visible + self.n_hidden
        for i in range(0, self.n_visible):
            for j in range(0, self.n_hidden):
                self.w[i, j] = best[k]
                k = k + 1

    def changes(self, best):
        return best

    def get_gradient(self):
        return self.loc_en_gradient

    def get_gradient_norm(self):
        return self.loc_en_gradient_norm


# In[ ]:
