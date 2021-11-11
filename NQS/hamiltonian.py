import attr
from attr import validators as vldts

import numpy as np


@attr.s
class Hamiltonian:
    omega = attr.ib(validator=vldts.instance_of(float))
    include_interaction = attr.ib(validator=vldts.instance_of(str))

    def local_energy(self, nqs):
        # The function computes the energy of the system described by
        # the wave function of the NQS object, and according to the
        # type of Hamiltonian you choose

        harmonic_oscillator = np.dot(
            self.omega * self.omega * nqs.visible_values_, nqs.visible_values_
        )

        kinetic = nqs.laplacian(nqs.visible_values_)

        local_energy = 0.5 * (kinetic + harmonic_oscillator)

        if self.include_interaction == "harmonic_oscillator":
            local_energy = local_energy
        elif self.include_interaction == "coulomb":
            local_energy += np.sum(nqs.inverse_distance(nqs.visible_values_))
        elif self.include_interaction == "calogero":
            local_energy += nqs.calogero(nqs.visible_values_)
            print(nqs.visible_values_, nqs.calogero(nqs.visible_values_))
        else:
            raise AssertionError("Método no válido")
        return local_energy
