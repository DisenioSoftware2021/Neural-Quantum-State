import attr
from attr import validators as vldts

import numpy as np


INTERACTIONS = {
    "harmonic_oscillator": lambda nqs: 0,
    "coulomb": lambda nqs: np.sum(nqs.inverse_distance(nqs.visible_values_)),
    "calogero": lambda nqs: nqs.calogero(nqs.visible_values_),
}
@attr.s
class Hamiltonian:
    omega = attr.ib(validator=vldts.instance_of(float))
    include_interaction = attr.ib(validator=vldts.in_(INTERACTIONS))

    def local_energy(self, nqs):
        # The function computes the energy of the system described by
        # the wave function of the NQS object, and according to the
        # type of Hamiltonian you choose

        harmonic_oscillator = np.dot(
            self.omega * self.omega * nqs.visible_values_, nqs.visible_values_
        )

        kinetic = nqs.laplacian(nqs.visible_values_)

        local_energy = 0.5 * (kinetic + harmonic_oscillator)

        iinteraction_function = INTERACTIONS[self.include_interaction]
        local_energy += iinteraction_function(nqs)

        return local_energy
        