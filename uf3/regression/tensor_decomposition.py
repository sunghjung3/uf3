from .least_squares import WeightedLinearModel
import jax
import jax.numpy as jnp
import numpy as np
from time import time_ns


class Model(WeightedLinearModel):
    def __init__(self,
                 bspline_config,
                 regularizer=None,
                 data_coverage=None,
                 rank_2b=2,
                 rank_3b=5,
                 seed=None,
                 **params):
        if bspline_config.degree > 2:
            raise NotImplementedError("Tensor decomposition currently only supports up to 2 body interactions.")
        if bspline_config.leading_trim != 0 or bspline_config.trailing_trim != 3:
            raise NotImplementedError("Tensor decomposition currently only supports leading_trim=0 and trailing_trim=3.")
        
        super().__init__(bspline_config, regularizer, data_coverage, **params)
        if seed is None:
            seed = time_ns()
        self.prng_key = jax.random.PRNGKey(seed)
        self.rank_2b = rank_2b
        self.rank_3b = rank_3b
        self.singular_vectors = None
        self.idx_map = None  # will map indices of input x to indices of singular vectors
        self.construct_singular_vectors()


    def construct_singular_vectors(self):
        elements = self.bspline_config.interactions_map[1]
        pairs = self.bspline_config.interactions_map[2]
        component_sizes, component_offsets = self.bspline_config.get_interaction_partitions()
        print("component_sizes", component_sizes)
        print("component_offsets", component_offsets)
        print("interactions:", self.bspline_config.interactions_map)
        self.singular_vectors = list()

        ## 1 body coefficients
        if self.bspline_config.offset_1b:
            self.singular_vectors.append( jnp.zeros(len(elements)) )

        ## 2 body
        try:
            assert all(component_sizes[pair] == component_sizes[pairs[0]] for pair in pairs[1:])
        except AssertionError:
            raise AssertionError("Tensor decomposition requires that all 2 body interactions have the same number of basis functions.")
        element_vector = jnp.zeros(len(elements))


        import sys
        sys.exit()

        if self.bspline_config.degree > 2:
            for degree in range(3, self.bspline_config.degree + 1):
                interactions = self.bspline_config.interactions_map[degree]
                raise NotImplementedError("Tensor decomposition currently only supports up to 2 body interactions.")

    
    def fit(self,
            x_e: np.ndarray,
            y_e: np.ndarray,
            x_f: np.ndarray = None,
            y_f: np.ndarray = None,
            weight: float = 0.5,
            batch_size=2500,
            ):
        # convert to jax arrays
        x_e = jnp.array(x_e)
        y_e = jnp.array(y_e)
        x_f = jnp.array(x_f)
        y_f = jnp.array(y_f)

        self.initialize_coefficient_vectors()