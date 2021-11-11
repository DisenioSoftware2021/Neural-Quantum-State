import attr
from attr import validators as vldts

import numpy as np


@attr.s
class GaussianNQS:
    """Represents a quantum state using
    the restricted Boltzmann machine."""

    n_hidden = attr.ib(validator=vldts.instance_of(int))
    n_dim = attr.ib(validator=vldts.instance_of(int))
    n_particles = attr.ib(validator=vldts.instance_of(int))
    sigma = attr.ib(validator=vldts.instance_of(float))
    d = attr.ib(validator=vldts.instance_of(float))
    seed = attr.ib(
        default=None, validator=vldts.optional(vldts.instance_of(int))
    )

    init_mean = attr.ib(default=0.0, validator=vldts.instance_of(float))
    init_sigma = attr.ib(default=0.1, validator=vldts.instance_of(float))
    positive = attr.ib(default=0.5)

    n_visible_ = attr.ib(init=False, repr=False)
    random_ = attr.ib(init=False, repr=False)

    visible_values_ = attr.ib(init=False, repr=False)
    hidden_values_ = attr.ib(init=False, repr=False)

    bias_a_ = attr.ib(init=False, repr=False)
    bias_b_ = attr.ib(init=False, repr=False)
    weights_ = attr.ib(init=False, repr=False)

    sigma_2_ = attr.ib(init=False, repr=False)
    sigmoid_q = attr.ib(init=False, repr=False)
    derivative_sigmoid_q = attr.ib(init=False, repr=False)

    @n_visible_.default
    def _nvisible__default(self):
        return self.n_dim * self.n_particles

    @random_.default
    def _random__default(self):
        return np.random.default_rng(self.seed)

    @bias_a_.default
    def _bias_a__default(self):
        return self.random_.normal(
            self.init_mean, self.init_sigma, size=self.n_visible_
        )

    @bias_b_.default
    def _bias_b__default(self):
        return self.random_.normal(
            self.init_mean, self.init_sigma, size=self.n_hidden
        )

    @weights_.default
    def _weights_default(self):
        return self.random_.normal(
            self.init_mean,
            self.init_sigma,
            size=(self.n_visible_, self.n_hidden),
        )

    @visible_values_.default
    def _visible_values__default(self):
        return self.random_.random(self.n_visible_)

    @sigma_2_.default
    def _sigma_2__default(self):
        return self.sigma * self.sigma

    @sigmoid_q.default
    def _sigmoid_q_default(self):
        return np.zeros(self.n_hidden)

    @derivative_sigmoid_q.default
    def _derivative_sigmoid_q_default(self):
        return np.zeros(self.n_hidden)

    def exponential_argument(self, visible_values_):
        q = (
            self.bias_b_
            + (np.matmul(visible_values_, self.weights_)) / self.sigma_2_
        )
        return q

    def psi(self, visible_values_, q):
        """wave function"""
        factor_1 = np.dot(
            visible_values_ - self.bias_a_, visible_values_ - self.bias_a_
        )
        factor_1 = np.exp(-factor_1 / (2.0 * self.sigma_2_))
        factor_2 = 1.0
        for j in range(self.n_hidden):
            factor_2 *= 1 + np.exp(q[j])
        return np.sqrt(factor_1 * factor_2)

    def sigmoid(self, q):
        self.sigmoid_q = 1 / (1 + np.exp(-np.array(q)))
        return self.sigmoid_q

    def derivative_sigmoid(self, q):
        self.derivative_sigmoid_q = np.exp(q) / (
            (1 + np.exp(q)) * (1 + np.exp(q))
        )
        return self.derivative_sigmoid_q

    def laplacian(self, visible_values_):
        """The functions compute the laplacian of the wave function"""

        laplacian = 0.0

        for i in range(self.n_visible_):
            derivative1_ln_psi = (
                (-visible_values_[i] + self.bias_a_[i]) / self.sigma_2_
            ) + np.dot(self.weights_[i, :], self.sigmoid_q) / self.sigma_2_
            sum_term = 0.0

            for j in range(self.n_hidden):
                sum_term += (
                    self.weights_[i, j]
                    * self.weights_[i, j]
                    * self.derivative_sigmoid_q[j]
                )
            derivative2_ln_psi = -1.0 / self.sigma_2_ + sum_term / (
                self.sigma_2_ * self.sigma_2_
            )

            derivative1_ln_psi *= self.positive
            derivative2_ln_psi *= self.positive

            laplacian += (
                -derivative1_ln_psi * derivative1_ln_psi - derivative2_ln_psi
            )
        return laplacian

    def laplacian_alfa(self, visible_values_):
        # The function calculates 1 / psi * derivative_psi / dalpha_i,
        #  this is the derived wave function with respect
        # to the network parameters a_i, b_j and w_ij
        derivative_psi = np.zeros(
            self.n_visible_ + self.n_hidden + (self.n_visible_ * self.n_hidden)
        )
        for k in range(self.n_visible_):
            derivative_psi[k] = (
                visible_values_[k] - self.bias_a_[k]
            ) / self.sigma_2_
            # print(k)
        for k in range(self.n_visible_, self.n_visible_ + self.n_hidden):
            derivative_psi[k] = self.sigmoid_q[k - self.n_visible_]
            # print(k,self.sigmoid_q[k-self.n_visible],k-self.n_visible)
        k = self.n_visible_ + self.n_hidden
        for i in range(self.n_visible_):
            for j in range(self.n_hidden):
                derivative_psi[k] = (
                    self.visible_values_[i] * self.sigmoid_q[j] / self.sigma_2_
                )
                k = k + 1
        return derivative_psi * self.positive

    def inverse_distance(self, visible_values_):
        # Computes the coulombian interaction term
        p1 = 0
        inverse_distances = np.zeros([self.n_particles, self.n_particles])
        # Loop over each particle
        print(self.n_visible_ - self.n_dim, self.n_dim)
        for i1 in range(0, self.n_visible_ - self.n_dim, self.n_dim):
            p2 = p1 + 1
            # Loop over each particles that particle r hasn't been paired with
            for i2 in range(i1 + self.n_dim, self.n_visible_, self.n_dim):
                # if i2>self.n_visible:
                #   break
                distance_particle = 0  # particle distance squared
                # Loop over dimensions
<<<<<<< HEAD
                for d in range(self.n_dim):
<<<<<<< HEAD
                    distance_particle += (x[i1 + d] - x[i2 + d]) * (x[i1 + d] - x[i2 + d])
                self.inverse_distances[p1, p2] = 1.0 / np.sqrt(distance_particle)
        return self.inverse_distances

    def calogero(self,x):
        #calculates the term of calogero model
        distance_particle=((x[0]-x[1])*(x[0]-x[1]))+self.d #particle distance squared
        g=2+2*self.d
        self.calogeno=g/(distance_particle)
        
        return self.calogeno
        
=======
=======
                for d in range(0, self.n_dim):
>>>>>>> a7b4781cb4ae1a2598a06cb7ae9c173f06cbebe9
                    distance_particle += (
                        visible_values_[i1 + d] - visible_values_[i2 + d]
                    ) * (visible_values_[i1 + d] - visible_values_[i2 + d])
                inverse_distances[p1, p2] = 1.0 / np.sqrt(distance_particle)
        return inverse_distances

    def calogero(self, x):
        # calculates the term of calogero model
        distance_particle = (
            (x[0] - x[1]) * (x[0] - x[1])
        ) + self.d  # particle distance squared
        g = 2 + 2 * self.d
        calogeno = g / (distance_particle)

        return calogeno
>>>>>>> 4debeb4c1aeaf76ba09c2f40c20719eeec821d76
