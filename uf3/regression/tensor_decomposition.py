from .least_squares import *
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from time import time_ns
import functools


class Model(WeightedLinearModel):
    def __init__(self,
                 bspline_config,
                 regularization={},
                 data_coverage=None,
                 nterms=None,
                 seed=None,
                 **params):
        ridge_map, curvature_map = bspline_config.get_regularization_maps(regularization)
        regularizer = dict()
        regularizer['ridge'] = ridge_map
        regularizer['curvature'] = curvature_map
        super().__init__(bspline_config, regularizer, data_coverage, **params)
        if seed is None:
            seed = time_ns()
        self.prng_key = jax.random.PRNGKey(seed)

        self.nterms = {2: 2, 3: 5}  # interaction order: number of expansion terms
        if nterms:
            self.nterms.update({order: nterms[order]
                                for order in nterms.keys() & self.nterms.keys()})

        self.component_sizes, self.component_offsets = \
            self.bspline_config.get_interaction_partitions()
        self.interactions_map = self.bspline_config.interactions_map

        self.singular_vectors = None
        self.body_offsets = dict()
        self.initialize_singular_vectors()


    def initialize_singular_vectors(self, sigma=10.0):
        elements = self.interactions_map[1]
        nelements = len(elements)
        pairs = self.interactions_map[2]
        leading_trim = self.bspline_config.leading_trim
        trailing_trim = self.bspline_config.trailing_trim
        if self.singular_vectors is not None:
            del self.singular_vectors
        self.singular_vectors = dict()

        ## 1 body coefficients
        if self.bspline_config.offset_1b:
            length = nelements
        else:
            length = 0
        self.prng_key, subkey = jax.random.split(self.prng_key)
        element_vector = jax.random.normal(subkey, shape=(length,))
        self.singular_vectors[1] = element_vector
        self.body_offsets[1] = 0
        self.body_offsets[2] = self.body_offsets[1] + length

        ## 2 body
        try:  # length of the coefficient vector must be the same
            ncomponents = self.component_sizes[pairs[0]]
            assert all(self.component_sizes[pair] == ncomponents for pair in pairs[1:])
        except AssertionError:
            raise AssertionError(
                f"Tensor decomposition requires that all 2 body interactions have the same resolution."
                )
        self.prng_key, subkey = jax.random.split(self.prng_key)
        element_vector = jax.random.normal(subkey, shape=(self.nterms[2], nelements))
        self.prng_key, subkey = jax.random.split(self.prng_key)
        length = ncomponents - leading_trim - trailing_trim
        coeff_vector = jax.random.normal(subkey, shape=(self.nterms[2], length))
        self.singular_vectors[2] = [element_vector, coeff_vector]
        self.body_offsets[3] = self.body_offsets[2] + length * len(pairs)

        ## Higher order
        if self.bspline_config.degree > 2:
            for degree in range(3, self.bspline_config.degree+1):
                if self.bspline_config.degree > 3:
                    raise NotImplementedError(
                        "UF3 currently only supports up to 3 body interactions."
                        )
                interactions = self.interactions_map[degree]
                try:  # dimensions of the coefficient tensor must be the same
                    ref = self.bspline_config.resolution_map[interactions[0]]
                    all(self.bspline_config.resolution_map[interaction] == ref
                        for interaction in interactions[1:])
                except AssertionError:
                    raise AssertionError(
                        f"Tensor decomposition requires that all {degree} body interactions have the same resolution."
                        )

                element_vectors = []
                for _ in range(degree - 1):
                    self.prng_key, subkey = jax.random.split(self.prng_key)
                    element_vectors.append( jax.random.normal(subkey, shape=(self.nterms[degree], nelements)) )
                coeff_vectors = []
                for i in range( round(degree * (degree-1) / 2) ):
                    self.prng_key, subkey = jax.random.split(self.prng_key)
                    length = ref[i] + 3
                    coeff_vectors.append( jax.random.normal(subkey, shape=(self.nterms[degree], length)) )
                self.singular_vectors[degree] = [element_vectors, coeff_vectors]
                self.body_offsets[degree+1] = self.body_offsets[degree] + \
                    self.component_offsets[interactions[-1]] - self.component_offsets[interactions[0]] \
                    + self.component_sizes[interactions[-1]]


    def flatten_coeff_2b(self, singular_vectors_2b):
        # TODO: constrain first element of the element_vector to be 1
        '''
        coeff_tensor_2b = jnp.einsum('pi,pj,pk->ijk',
                                     singular_vectors_2b[0],
                                     singular_vectors_2b[0],
                                     singular_vectors_2b[1])
        mask = jnp.triu_indices(coeff_tensor_2b.shape[0])
        flattened_coeff_2b = coeff_tensor_2b[mask[0], mask[1]].flatten()  # TODO: test efficiency compared to below
        '''
        coeff_by_pair = []  # TODO: compare efficiency and memory usage if flattened_coeff_2b is preallocated and filled
        for i in range(singular_vectors_2b[0].shape[1]):
            for j in range(i, singular_vectors_2b[0].shape[1]):
                vec = jnp.sum((singular_vectors_2b[0][:, i] *  # TODO: test if it is faster if singular_vectors[2][0] is stored column-wise
                               singular_vectors_2b[0][:, j]).reshape(-1, 1) *
                               singular_vectors_2b[1],
                               axis=0)
                coeff_by_pair.append(vec)
        flattened_coeff_2b = jnp.concatenate(coeff_by_pair)

        return flattened_coeff_2b


    def flatten_coeff_3b(self, singular_vectors_3b):
        '''
        coeff_tensor_3b = jnp.einsum('pi,pj,pk,pl,pm,pn->ijklmn',
                                     singular_vectors_3b[0][0],  # v
                                     singular_vectors_3b[0][1],  # w
                                     singular_vectors_3b[0][1],  # w
                                     singular_vectors_3b[1][0],  # a
                                     singular_vectors_3b[1][1],  # b
                                     singular_vectors_3b[1][2])  # c
        '''
        reduced_abc = jnp.einsum('pi,pj,pk->ijk',
                                  singular_vectors_3b[1][0],  # a
                                  singular_vectors_3b[1][1],  # b
                                  singular_vectors_3b[1][2]   # c
                                  ).flatten()
        coeff_by_triplet = []
        trio_counter = 0
        for i in range(singular_vectors_3b[0][0].shape[1]):
            for j in range(singular_vectors_3b[0][1].shape[1]):
                for k in range(j, singular_vectors_3b[0][1].shape[1]):
                    vec = jnp.sum((singular_vectors_3b[0][0][:, i] *
                                   singular_vectors_3b[0][1][:, j] *
                                   singular_vectors_3b[0][1][:, k]).reshape(-1, 1) *
                                   reduced_abc,
                                   axis=0)
                    trio = self.bspline_config.interactions_map[3][trio_counter]
                    template_mask = self.bspline_config.template_mask[trio]
                    coeff_by_triplet.append( vec[template_mask] )
                    trio_counter += 1
        flattened_coeff_3b = jnp.concatenate(coeff_by_triplet)

        return flattened_coeff_3b

    
    def singular_vectors_to_flattened_coeffs(self, singular_vectors, d):
        flattened_coeffs = dict()
        flattened_coeffs[1] = singular_vectors[1]
        flattened_coeffs[2] = self.flatten_coeff_2b(singular_vectors[2])
        if d > 2:
            flattened_coeffs[3] = self.flatten_coeff_3b(singular_vectors[3])
        return flattened_coeffs


    def train_predict(self, x: jnp.array, coefficients: jnp.array):
        """
        Predict using fit coefficients.

        Args:
            x (jnp.array): input matrix of shape (n_samples, n_features).
            coefficients (jnp.array): 1-dimensional vector of coefficients.

        Returns:
            predictions (jnp.array): vector of predictions.
        """
        predictions = jnp.dot(x, coefficients)
        return predictions


    def loss_function(self,
                      singular_vectors: dict,
                      x_e: jnp.array,
                      y_e: jnp.array,
                      x_f: jnp.array = None,
                      y_f: jnp.array = None,
                      e_weight: float = 1.0,
                      f_weight: float = 1.0,
                      # TODO: add regularization terms
                      ):
        loss = 0.0
        flattened_coeffs = self.singular_vectors_to_flattened_coeffs(singular_vectors,
                                                                     self.bspline_config.degree)

        # ridge regularization
        #loss += self.regularizer['ridge'][1] * jnp.sum(flattened_coeffs[1]**2)  # 1 body ridge
        #loss += self.regularizer['ridge'][2] * jnp.sum(flattened_coeffs[2]**2)  # 2 body ridge
        # TODO: add higher order ridge

        # curvature regularization
        # TODO: figure out curvature regularization for 3 body

        # error
        coefficients = jnp.concatenate( tuple(flattened_coeffs.values()) )
        p_e = self.train_predict(x_e, coefficients)
        loss += e_weight * jnp.sum((p_e-y_e)**2)
        if x_f is not None:
            p_f = self.train_predict(x_f, coefficients)
            loss += f_weight * jnp.sum((p_f-y_f)**2)

        return loss


    def fit(self,
            x_e: np.ndarray,
            y_e: np.ndarray,
            x_f: np.ndarray = None,
            y_f: np.ndarray = None,
            weight: float = 0.5,
            max_epochs: int = 1000,
            tol: float = 1e-6,
            batch_size=2500,
            reinitialize=False,
            ):

        if reinitialize:
            self.initialize_singular_vectors()

        x_e, y_e = freeze_columns(x_e,
                                  y_e,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        x_e = jnp.array(x_e)
        y_e = jnp.array(y_e)
        energy_weight = 1.0

        if x_f is not None:
            x_f, y_f = freeze_columns(x_f,
                                      y_f,
                                      self.mask,
                                      self.frozen_c,
                                      self.col_idx)
            x_f = jnp.array(x_f)
            y_f = jnp.array(y_f)
            warnings.filterwarnings("error", append=True)  # to catch divide by zero warnings
            try:
                energy_weight = weight / len(y_e) / jnp.var(y_e)
                force_weight = (1-weight) / len(y_f) / jnp.var(y_f)
            except (ZeroDivisionError, FloatingPointError, RuntimeWarning):
                energy_weight = 1.0
                force_weight = 1 / len(y_f)
            warnings.filters.pop()  # undo the filter

        # train
        self.loss = functools.partial(self.loss_function, x_e=x_e, y_e=y_e, x_f=x_f, y_f=y_f,
                                 e_weight=energy_weight, f_weight=force_weight)
        self.loss_grad = jax.grad(self.loss, argnums=0)
        self.loss_and_grad = jax.jit(lambda params: (self.loss(params), self.loss_grad(params)))
        optimizer = jaxopt.LBFGS
        solver = optimizer(fun=self.loss_and_grad,
                           value_and_grad=True,
                           maxiter=max_epochs,
                           tol=tol,
                           #verbose=True,
                           #jit=True,
                           )
        self.singular_vectors, state = solver.run(self.singular_vectors)
        loss_value = self.loss(self.singular_vectors)
        print(f"Trained singular vectors: {self.singular_vectors}")
        print(f"Final loss: {loss_value}")
        #print(f"Final state: {state}")

        flattened_coeff = \
            self.singular_vectors_to_flattened_coeffs(self.singular_vectors,
                                                      self.bspline_config.degree)
        solved_coefficients = np.array( jnp.concatenate(
            tuple(flattened_coeff.values())
            ) )
        solved_coefficients = revert_frozen_coefficients(solved_coefficients,
                                                         self.n_feats,
                                                         self.mask,
                                                         self.frozen_c,
                                                         self.col_idx)
        self.coefficients = solved_coefficients