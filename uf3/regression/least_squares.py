"""
This module provides the WeightedLinearModel class for fitting UF potentials
from featurized DataFrames using regularized least squares.
"""

from typing import List, Dict, Collection, Tuple
import os
import copy
import time
import warnings
import numpy as np
import pandas as pd
import torch
import scipy
import ndsplines
from uf3.representation import bspline, process
from uf3.data import io
from uf3.data import composition
from uf3.util import json_io
from uf3.util import parallel
from uf3.util import torch_util


class VarianceRecorder:
    """Convenience class for computing online variance and mean"""
    def __init__(self, mean=0, std=0, n=0):
        self.mean = mean
        self.std = std
        self.n = int(n)

    def update(self, batch: Collection) -> Tuple:
        """
        Args:
            batch (list or np.ndarray): n-dimensional data. For speed purposes,
                dimensions are not checked for compatibility so caution
                is advised when working with multidimensional data.
                Statistics are computed along the first axis.

        Returns:
            (current mean, current standard deviation, current entry count)
        """
        if self.n == 0:
            self.mean = np.mean(batch, axis=0)
            self.std = np.std(batch, axis=0)
            self.n = len(batch)
            return self.mean, self.std, self.n
        else:
            batch_std = np.std(batch, axis=0)
            batch_mean = np.mean(batch, axis=0)
            m = float(self.n)
            n = len(batch)
            std = (m / (m + n) * self.std**2
                   + n / (m + n) * batch_std**2
                   + m * n / (m + n)**2 * (self.mean - batch_mean)**2)
            self.std = np.sqrt(std)
            self.mean = m / (m + n) * self.mean + n / (m + n) * batch_mean
            self.n += n
            return self.mean, self.std, self.n

    def update_with_components(self, df, keys=None):
        """Wrapper for dataframe with multiple columns of interest"""
        if keys is None:
            keys = ["fx", "fy", "fz"]
        batch = []
        for j, *components in df[keys].itertuples():
            if any([component is np.nan for component in components]):
                continue
            if np.ndim(components) > 1:  # if components are not scalars
                components = list(np.concatenate(components))
            batch.extend(components)
        self.update(batch)
        return self.mean, self.std, self.n


class BasicLinearModel:
    """
    Base class for linear regression.
    """
    def __init__(self,
                 regularizer: np.ndarray = None):
        """
        Args:
            regularizer (np.ndarray): regularization matrix.
        """
        self.coefficients = None
        self.regularizer = regularizer

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            ridge_penalty: float = 1e-8,
            ):
        """
        Direct solution to linear least squares with LU decomposition.

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            ridge_penalty (float): magnitude of ridge penalty. Ignored
                if self.regularizer is set at initialization.
        """
        gram, ordinate = moore_penrose_components(x, y)
        if self.regularizer is None:
            regularizer = np.eye(len(gram)) * ridge_penalty
        else:
            regularizer = self.regularizer
        regularizer = np.dot(regularizer.T, regularizer)
        coefficients = lu_factorization(gram + regularizer, ordinate)
        self.coefficients = coefficients

    def predict(self, x: np.ndarray):
        """
        Predict using fit coefficients.

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).

        Returns:
            predictions (np.ndarray): vector of predictions.
        """
        predictions = np.dot(x, self.coefficients)
        return predictions

    def score(self, x, y, weights=None, normalize=True):
        """
        Evaluate score (negative error metric).

        Args:
            x (np.ndarray): input matrix of shape (n_samples, n_features).
            y (np.ndarray): output vector of length n_samples.
            weights (np.ndarray): sample weights (optional).
            normalize (bool): whether to normalize by the std of y.

        Returns:
            score (float): negative weighted root-mean-square-error.
        """
        n_features = len(x[0])
        if weights is not None:
            w_matrix = np.eye(n_features) * np.sqrt(weights)
            x = np.dot(w_matrix, x)
            y = np.dot(w_matrix, y)
        predictions = self.predict(x)
        score = -rmse_metric(y, predictions)
        if normalize:
            score /= np.std(y)
        return score


class WeightedLinearModel(BasicLinearModel):
    """
    Handler class for regularized linear least squares using energies and
    forces and basis set provided by bspline.BsplineBasis.
    """
    def __init__(self,
                 bspline_config,
                 regularizer=None,
                 data_coverage=None,
                 **params):
        super().__init__(regularizer)
        self.bspline_config = bspline_config
        n_basis = np.sum(self.bspline_config.get_feature_partition_sizes())
        if data_coverage is not None:
            if len(data_coverage) == n_basis:
                self.data_coverage = data_coverage
            else:
                raise ValueError(
                    f"Incorrect data_coverage shape: "
                    f"{len(data_coverage)} != {n_basis}"
                )
        else:
            self.data_coverage = np.zeros(n_basis, dtype=bool)

        if self.regularizer is None:
            # initialize regularizer matrix if unspecified.
            self.set_params(**params)

    def set_params(self, **params):
        """Set parameters from keyword arguments. Initializes
            regularizer with default parameters if unspecified."""
        if "bspline_config" in params:
            self.bspline_config = params["bspline_config"]
        if "regularizer" in params:
            self.regularizer = params["regularizer"]
        elif self.regularizer is None:
            reg_params = {k: v for k, v in params.items()
                          if isinstance(v, (int, float, np.floating))}
            reg = self.bspline_config.get_regularization_matrix(**reg_params)
            self.regularizer = reg

    @staticmethod
    def from_config(config):
        return WeightedLinearModel.from_dict(config)

    @staticmethod
    def from_dict(config):
        bspline_config = bspline.BSplineBasis.from_dict(config)
        regularizer = config.get("regularizer", None)
        data_coverage = config.get("data_coverage", None)
        model = WeightedLinearModel(bspline_config,
                                    regularizer=regularizer,
                                    data_coverage=data_coverage)
        model.load(solution=config)
        return model

    @staticmethod
    def from_json(filename):
        """Load model (coefficients and knots map) from json file."""
        dump = json_io.load_interaction_map(filename)
        return WeightedLinearModel.from_dict(dump)

    def as_dict(self):
        solution = arrange_coefficients(self.coefficients, self.bspline_config)
        for trio in self.bspline_config.interactions_map.get(3, []):
            solution[trio] = self.bspline_config.decompress_3B(solution[trio],
                                                               trio)
        knots_map = self.bspline_config.knots_map
        dump = dict(coefficients=solution,
                    knots=knots_map,
                    data_coverage=self.data_coverage,
                    **self.bspline_config.as_dict())
        return dump

    @property
    def n_feats(self):
        return self.bspline_config.n_feats

    @property
    def frozen_c(self):
        return self.bspline_config.frozen_c

    @property
    def col_idx(self):
        return self.bspline_config.col_idx

    @property
    def mask(self):
        return get_freezing_mask(self.n_feats, self.col_idx)

    def __repr__(self):
        if self.coefficients is None:
            fit = "False"
        else:
            fit = "True"
        summary = ["WeightedLinearModel:",
                   f"    Fit: {fit}",
                   self.bspline_config.__repr__()
                   ]
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()

    def fit_with_gram(self, gram: np.ndarray, ordinate: np.ndarray):
        """
        Intermediate function for direct solution using gram matrix
        and ordinate (Moore-penrose inverse).

        Args:
            gram (np.ndarray): gram matrix (x^T x)
            ordinate (np.ndarray: ordinate (x^T y)
        """
        data_coverage = (np.sum(gram, axis=0) != 0)
        data_coverage = revert_frozen_coefficients(data_coverage,
                                                   self.n_feats,
                                                   self.mask,
                                                   self.frozen_c,
                                                   self.col_idx)
        self.data_coverage = np.logical_or(self.data_coverage, data_coverage)
        regularizer = freeze_regularizer(self.regularizer, self.mask)
        regularizer = np.dot(regularizer.T, regularizer)
        coefficients = lu_factorization(gram + regularizer, ordinate)
        coefficients = revert_frozen_coefficients(coefficients,
                                                  self.n_feats,
                                                  self.mask,
                                                  self.frozen_c,
                                                  self.col_idx)
        self.coefficients = coefficients

    def fit(self,
            x_e: np.ndarray,
            y_e: np.ndarray,
            x_f: np.ndarray = None,
            y_f: np.ndarray = None,
            weight: float = 0.5,
            batch_size=2500,
            ):
        """
        Direct solution from input-output pairs corresponding to
        energies and forces, with option to weigh their respective
        contributions.

        Args:
            x_e (np.ndarray): input matrix of shape (n_samples, n_features).
            y_e (np.ndarray): output vector of length n_samples.
            x_f (np.ndarray): input matrix corresponding to forces.
            y_f (np.ndarray): output vector corresponding to forces.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            batch_size: maximum batch size for gram matrix construction.
        """
        x_e, y_e = freeze_columns(x_e,
                                  y_e,
                                  self.mask,
                                  self.frozen_c,
                                  self.col_idx)
        gram_e, ord_e = batched_moore_penrose(x_e, y_e, batch_size=batch_size)
        if x_f is not None:
            energy_weight, force_weight = calc_E_F_weights(len(y_e),
                                                           len(y_f),
                                                           np.std(y_e),
                                                           np.std(y_f))
            x_f, y_f = freeze_columns(x_f,
                                      y_f,
                                      self.mask,
                                      self.frozen_c,
                                      self.col_idx)
            gram_f, ord_f = batched_moore_penrose(x_f,
                                                  y_f,
                                                  batch_size=batch_size)
            gram, ordinate = self.combine_weighted_gram(gram_e, gram_f, ord_e,
                                                        ord_f, energy_weight,
                                                        force_weight, weight)
        else:
            gram = gram_e
            ordinate = ord_e
        self.fit_with_gram(gram, ordinate)

    def combine_weighted_gram(self,
                              gram_e: np.ndarray,
                              gram_f: np.ndarray,
                              ord_e: np.ndarray,
                              ord_f: np.ndarray,
                              energy_weight: float,
                              force_weight: float,
                              weight: float):
        """
        Apply weighting to gram matrices and ordinates for energy and
        force contributions to the fit.

        Args:
            gram_e (np.ndarray): gram matrix (x^T x) for energies.
            gram_f (np.ndarray): gram matrix (x^T x) for forces.
            ord_e (np.ndarray): ordinate (x^T y) for energies.
            ord_f (np.ndarray): ordinate (x^T y) for forces.
            energy_weight: 1 / (# energies * sqrt(Var(energies)))
            force_weight: 1 / (# forces * sqrt(Var(forces)))
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.

        Returns:
            gram (np.ndarray): gram matrix (x^T x) for fitting.
            ordinate (np.ndarray): ordinate (x^T y) for fitting.
        """
        gram = ((weight * energy_weight**2 * gram_e)
                + ((1 - weight) * force_weight**2 * gram_f))
        ordinate = ((weight * energy_weight**2 * ord_e)
                    + ((1 - weight) * force_weight**2 * ord_f))
        return gram, ordinate

    def fit_from_file(self,
                      filename: str,
                      subset: Collection,
                      weight: float = 0.5,
                      batch_size=2500,
                      sample_weights: Dict = None,
                      energy_key="energy",
                      progress: str = "bar",
                      drop_columns: List[str] = None,
                      sparse_hdf5: bool = False,
                      ):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and compute direct solution via LU decomposition.

        Args:
            filename (str): path to HDF5 file.
            subset (list): list of keys for training.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            progress (str): style for progress indicators.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
            sparse_hdf5 (bool): whether the HDF5 features file is in sparse format.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        n_tables, _, table_names, _ = io.analyze_hdf_tables(filename)
        gram_e, gram_f, ord_e, ord_f = self.initialize_gram_ordinate()
        e_variance = VarianceRecorder()
        f_variance = VarianceRecorder()
        table_iterator = parallel.progress_iter(np.arange(n_tables),
                                                style=progress)
        for j in table_iterator:
            table_name = table_names[j]
            df = process.load_feature_db(filename, table_name, sparse_hdf5=sparse_hdf5)
            keys = df.index.unique(level=0).intersection(subset)
            if len(keys) == 0:
                continue

            if drop_columns != None:
                df.drop(columns=drop_columns,inplace=True)

            intermediates = self.gram_from_df(df,
                                              keys,
                                              e_variance=e_variance,
                                              f_variance=f_variance,
                                              sample_weights=sample_weights,
                                              energy_key=energy_key,
                                              batch_size=batch_size,
                                              )
            g_e, g_f, o_e, o_f = intermediates
            gram_e += g_e
            gram_f += g_f
            ord_e += o_e
            ord_f += o_f
        energy_weight, force_weight = calc_E_F_weights(e_variance.n,
                                                       f_variance.n,
                                                       e_variance.std,
                                                       f_variance.std)
        gram, ordinate = self.combine_weighted_gram(gram_e,
                                                    gram_f,
                                                    ord_e,
                                                    ord_f,
                                                    energy_weight,
                                                    force_weight,
                                                    weight)
        self.fit_with_gram(gram, ordinate)

    def initialize_gram_ordinate(self):
        """Initialize empty matrices for gram matrices and ordinates."""
        n_columns = self.n_feats - len(self.col_idx)
        gram_e = np.zeros((n_columns, n_columns))
        ord_e = np.zeros(n_columns)
        gram_f = np.zeros((n_columns, n_columns))
        ord_f = np.zeros(n_columns)
        return gram_e, gram_f, ord_e, ord_f

    def gram_from_df(self,
                     df: pd.DataFrame,
                     keys: Collection,
                     e_variance: VarianceRecorder = None,
                     f_variance: VarianceRecorder = None,
                     sample_weights: Dict = None,
                     energy_key: str = "energy",
                     batch_size: int = 2500,
                     ):
        """
        Extract inputs and outputs from dataframe and compute
        moore-penrose components (gram matrices and ordinates).

        Args:
            df (pd.DataFrame): DataFrame of energy/force features.
            keys (list): keys to query from df (e.g. training subset).
            e_variance (VarianceRecorder): handler for accumulating
                statistics for energies (mean and variance).
            f_variance (VarianceRecorder): handler for accumulating
                statistics for forces (mean and variance).
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
        """
        n_elements = len(self.bspline_config.element_list)
        x_e, y_e, x_f, y_f = freeze_columns_from_df(df,
                                                    keys,
                                                    n_elements,
                                                    self.mask,
                                                    self.frozen_c,
                                                    self.col_idx,
                                                    energy_key=energy_key,
                                                    sample_weights=sample_weights,
                                                    )
        if e_variance is not None and f_variance is not None:
            e_variance.update(y_e)
            f_variance.update(y_f)
        gram_e, ordinate_e = batched_moore_penrose(x_e,
                                                   y_e,
                                                   batch_size=batch_size)
        gram_f, ordinate_f = batched_moore_penrose(x_f,
                                                   y_f,
                                                   batch_size=batch_size)
        return gram_e, gram_f, ordinate_e, ordinate_f

    def batched_predict(self,
                        filename: str,
                        keys: List[str] = None,
                        table_names: List[str] = None,
                        score: bool = True,
                        drop_columns: List[str] = None,
                        sparse_hdf5: bool = False,
                        client = None,
                        shuffle: bool = False,
                        progress: str = "bar",
                        ):
        """
        Extract inputs and outputs from HDF5 file and predict energies/forces.

        Args:
            filename: path to HDF5 file.
            keys (list): keys to query from df (e.g. training subset).
            table_names (list): list of table names in HDF5 to read.
            score (bool): whether to return root mean square error metrics.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
            sparse_hdf5 (bool): whether the HDF5 features file is in sparse format.
            client (concurrent.futures.Executor, dask.distributed.Client)
            shuffle (bool): whether to shuffle the order of keys.
            progress (str): style for progress indicators.

        Returns:
            y_e (np.ndarray): target values for energies.
            p_e (np.ndarray): prediction values for energies.
            y_f (np.ndarray): target values for forces.
            p_f (np.ndarray): prediction values for forces.
            rmse_e (np.ndarray): RMSE across energy predictions.
            rmse_e (np.ndarray): RMSE across force predictions.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
            client (concurrent.futures.Executor, dask.distributed.Client)
            shuffle (bool): whether to shuffle the order of keys.
            progress (str): style for progress indicators.
        """
        n_elements = len(self.bspline_config.element_list)
        y_e, p_e, y_f, p_f = batched_prediction_parallel(self,
                                                         filename,
                                                         table_names=table_names,
                                                         subset_keys=keys,
                                                         n_elements=n_elements,
                                                         drop_columns=drop_columns,
                                                         sparse_hdf5=sparse_hdf5,
                                                         client=client,
                                                         shuffle=shuffle,
                                                         progress=progress,
                                                         )
        if score:
            rmse_e = rmse_metric(y_e, p_e)
            rmse_f = rmse_metric(y_f, p_f)
            print(f"RMSE (energy): {rmse_e:.3F}")
            print(f"RMSE (forces): {rmse_f:.3F}")
            return y_e, p_e, y_f, p_f, rmse_e, rmse_f
        else:
            return y_e, p_e, y_f, p_f

    def to_json(self, filename: str):
        """Save model (coefficients and knots map) to json file."""
        json_io.dump_interaction_map(self.as_dict(),
                                     filename=filename,
                                     write=True)

    def dump(self):
        """Legacy alias"""
        return self.as_dict()

    def load(self,
             solution: Dict = None,
             filename: str = None,
             ):
        """
        Reflatten coefficients (e.g. obtained through arrange_coefficients)
        and load into model for prediction.

        Args:
            solution (dict): dictionary of 1B, 2B, ... terms
                organized as interaction: vector entries.
            filename (str): filename of json dump containing solution.
        """
        if filename is not None:
            if solution is not None:
                warnings.warn("Provided solutions ignored; loading file.")
            solution = json_io.load_interaction_map(filename)
        elif solution is None:
            raise ValueError("Neither solution nor filename were provided.")
        if "coefficients" in solution:
            solution = solution["coefficients"]
        elif "solution" in solution:
            # TODO: proper deprecation
            warnings.warn("'solution' should be renamed to 'coefficients'")
            solution = solution["solution"]
        for key in solution:
            if isinstance(key, tuple):
                sorted_key = composition.sort_interaction_symbols(key)
                if sorted_key != key:
                    solution[sorted_key] = solution[key]
        # consistency check with bspline_config
        component_len = self.bspline_config.get_interaction_partitions()[0]
        for pair in self.bspline_config.interactions_map[2]:
            n_target = component_len[pair]
            if pair not in solution:
                warnings.warn(f"{pair} not provided.")
                solution[pair] = np.zeros(n_target)
            n_provided = len(solution[pair])
            if n_provided != n_target:
                raise ValueError(
                    f"Incorrect shape: {pair}, {n_provided} != {n_target}"
                )
        for trio in self.bspline_config.interactions_map.get(3, []):
            n_target = component_len[trio]
            if trio not in solution:
                warnings.warn(f"{trio} not provided.")
            if trio in solution:
                # decompress if necessary
                component = np.array(solution[trio])
                if len(np.shape(component)) > 1:
                    vector = self.bspline_config.compress_3B(component,
                                                             trio,
                                                             fitting = False)
                    solution[trio] = vector
            n_provided = len(solution[trio])
            if n_provided != n_target:
                raise ValueError(
                    f"Incorrect shape: {trio}, {n_provided} != {n_target}"
                )
        flattened_coefficients = []
        for element in self.bspline_config.element_list:
            value = solution[element]
            flattened_coefficients.append([value])
        for degree in range(2, self.bspline_config.degree + 1):
            interactions = self.bspline_config.interactions_map[degree]
            for interaction in interactions:
                values = solution[interaction]
                flattened_coefficients.append(values)
        # self-energies, pair interactions & trio interactions
        n_interactions = len(self.bspline_config.partition_sizes)
        # add self-energy as separate interactions
        n_coefficients = sum(self.bspline_config.partition_sizes)
        if len(flattened_coefficients) != n_interactions:
            error_line = "Incorrect interactions: {} provided, {} expected."
            error_line = error_line.format(len(flattened_coefficients),
                                           n_interactions)
            raise ValueError(error_line)
        flattened_coefficients = np.concatenate(flattened_coefficients)
        if len(flattened_coefficients) != n_coefficients:
            error_line = "Incorrect coefficients: {} provided, {} expected."
            error_line = error_line.format(len(flattened_coefficients),
                                           n_coefficients)
            raise ValueError(error_line)
        self.coefficients = np.array(flattened_coefficients)

    def fix_repulsion_2b(self, pair, r_target=None, min_curvature=2.0):
        components = self.bspline_config.get_interaction_partitions()
        component_sizes, component_offsets = components
        offset = component_offsets[pair]
        n_basis = component_sizes[pair]
        idx_subset = np.arange(offset, offset + n_basis)
        c_subset = self.coefficients[idx_subset]
        coverage = self.data_coverage[idx_subset]
        min_coverage = np.argmax(coverage == True)
        if min_coverage == 0:
            print(f"Coverage is sufficient; no fix applied to {pair}.")
        idx_fix = np.arange(self.bspline_config.leading_trim, min_coverage)

        knot_sequence = self.bspline_config.knots_map[pair]
        r_centers = knot_sequence[2: n_basis + 2]
        if r_target is None:
            r_target = r_centers[min_coverage]
        r_centers = r_centers[idx_fix]
        c_new = get_spline_taylor_expansion(r_target,
                                            r_centers,
                                            c_subset,
                                            knot_sequence,
                                            min_curvature=min_curvature)
        print(f"{pair} Correction: adjusted {len(idx_fix)} coefficients.")
        self.coefficients[idx_subset[idx_fix]] = c_new


class AlchemicalModel(WeightedLinearModel):
    """
    Alchemical learning ("pseudo-interaction") model for fitting energies and
    forces.

    XXX: currently only 2-body interactions and all pseudo-interactions
    must have the same spline construction and offsets are fit.

    XXX: self.data_coverage is not implemented yet.
    
    XXX: the regularizer matrix should already have frozen coefficients removed.

    """
    def __init__(self,
                 bspline_config,
                 n_pseudo,
                 regularizer=None,
                 data_coverage=None,
                 init_params=None,
                 **params):
        super().__init__(bspline_config, regularizer, data_coverage, **params)
        if self.bspline_config.degree != 2:
            raise ValueError("Only 2-body interactions supported.")
        self.n_pseudo = n_pseudo
        component_sizes = self.bspline_config.get_interaction_partitions()[0]

        # temporary sanity checks
        for pair in self.bspline_config.interactions_map[2]:
            if not component_sizes[pair] == self.n_basis + self.bspline_config.leading_trim + self.bspline_config.trailing_trim:
                raise ValueError("Inconsistent component sizes.")
        assert self.bspline_config.offset_1b  # fit 1-body

        # Parameter arrays for training.
        # After training, they will be stored to self.coefficients.
        self.initialize_parameters(init_params)

    @property
    def n_basis(self):
        component_sizes = self.bspline_config.get_interaction_partitions()[0]
        n_basis = component_sizes[self.bspline_config.interactions_map[2][0]]
        return n_basis - self.bspline_config.leading_trim - \
                self.bspline_config.trailing_trim

    @property
    def n_elements(self):
        return len(self.bspline_config.element_list)
    
    @property
    def n_pairtypes(self):
        return len(self.bspline_config.interactions_map[2])

    def __repr__(self):
        if self.coefficients is None:
            fit = "False"
        else:
            fit = "True"
        summary = ["AlchemicalModel:",
                   f"    Fit: {fit}",
                   f"    n_pseudo: {self.n_pseudo}",
                   f"    n_basis: {self.n_basis}",
                   self.bspline_config.__repr__()
                   ]
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()
    
    def initialize_parameters(self, init_params):
        """Initialize parameters for training."""
        # TODO: need to figure out best way to initialize these
        if init_params is None:
            self.coeff_1b = np.zeros(self.n_elements)
            self.coeff_2b = np.zeros((self.n_basis, self.n_pseudo))
            self.pseudo_weights = np.random.rand(self.n_pairtypes, self.n_pseudo)*2-1
        else:
            self.coeff_1b = init_params['coeff_1b']
            if self.coeff_1b.shape != (self.n_elements,):
                raise ValueError("Incorrect shape for 1-body coefficients.\n"
                                 f"\tExpected: ({self.n_elements},)\n"
                                 f"\tProvided: {self.coeff_1b.shape}")
            self.coeff_2b = init_params['coeff_2b']
            if self.coeff_2b.shape != (self.n_basis, self.n_pseudo):
                raise ValueError("Incorrect shape for 2-body coefficients.\n"
                                  f"\tExpected: ({self.n_basis}, {self.n_pseudo})\n"
                                  f"\tProvided: {self.coeff_2b.shape}")
            self.pseudo_weights = init_params['pseudo_weights']
            if self.pseudo_weights.shape != (self.n_pairtypes, self.n_pseudo):
                raise ValueError("Incorrect shape for pseudo weights.\n"
                                  f"\tExpected: ({self.n_pairtypes}, {self.n_pseudo})\n"
                                  f"\tProvided: {self.pseudo_weights.shape}")

    def initialize_training(self,
                            filename: str,
                            subset: Collection,
                            sparse_hdf5: bool,
                            max_iter: int,
                            checkpoint: int,
                            checkpoint_dir: str,
                            params_filename: str,
                            tracker_filename: str,
                            train_iter_filename: str,
                            client,
                            resume: bool,
                            ):
        """
        Initialization for training.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)  # no error if exists
        get_params_dir = lambda i: os.path.join(checkpoint_dir, str(i))
        get_params_path = lambda i: os.path.join(get_params_dir(i), params_filename)
        tracker_filename = os.path.join(checkpoint_dir, tracker_filename)
        train_iter_filename = os.path.join(checkpoint_dir, train_iter_filename)

        if resume:
            try:
                train_tracker = np.load(tracker_filename)
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find {tracker_filename}.\n"
                                        "Set `resume=False` to start from scratch.\n"
                                        "Exiting...")
            self.frozen_2b_data_coverage = train_tracker["frozen_2b_data_coverage"]
            change_pseudo_tracker = train_tracker["change_pseudo"]
            assert len(change_pseudo_tracker) == max_iter
            change_1b_tracker = train_tracker["change_1b"]
            assert len(change_1b_tracker) == max_iter
            change_2b_tracker = train_tracker["change_2b"]
            assert len(change_2b_tracker) == max_iter
            rmse_e_tracker = train_tracker["rmse_e"]
            assert len(rmse_e_tracker) == max_iter + 1
            rmse_f_tracker = train_tracker["rmse_f"]
            assert len(rmse_f_tracker) == max_iter + 1
            time_tracker = train_tracker["time"]
            assert len(time_tracker) == max_iter + 1

            # find iteration to resume from
            init_iter = np.where(np.isnan(time_tracker))[0][0] - 1
            init_iter = (init_iter // checkpoint) * checkpoint
            if init_iter < checkpoint:
                raise ValueError(f"Tried to resume from iteration {init_iter+1}"
                                 f" but this is before the first checkpoint.\n"
                                 f"Please set `resume=False` to start from scratch.\n"
                                 "Exiting...")

            try:
                alchemical_params = np.load(get_params_path(init_iter))
            except:
                raise FileNotFoundError(f"Tried to resume from iteration{init_iter+1}"
                                        f" but could not find {get_params_path(init_iter)}.\n"
                                        "Exiting...")
            self.initialize_parameters(alchemical_params)
            
        else:  # initialize from scratch
            init_iter = 0
            if os.path.exists( get_params_path(1) ):
                warnings.warn(f"Warning: {get_params_path(1)} already exists. It will be overwritten")
            if os.path.exists(tracker_filename):
                warnings.warn(f"Warning: {tracker_filename} already exists. It will be overwritten")

            self.frozen_2b_data_coverage = np.zeros(self.n_basis, dtype=bool)
            change_pseudo_tracker = np.full(max_iter, np.nan)
            change_1b_tracker = np.full(max_iter, np.nan)
            change_2b_tracker = np.full(max_iter, np.nan)
            rmse_e_tracker = np.full((max_iter+1,), np.nan)  # +1 for initial RMSE
            rmse_f_tracker = np.full((max_iter+1,), np.nan)  # +1 for initial RMSE
            time_tracker = np.full((max_iter+1,), np.nan)
            time_tracker[0] = 0

        # initial RMSE check
        print(f"Initial RMSE check before iteration {init_iter+1}:")
        self.decompress_alchemical_parameters()
        _, _, _, _, rmse_e, rmse_f = self.batched_predict(filename,
                                                keys=subset,
                                                sparse_hdf5=sparse_hdf5,
                                                client=client,
                                                )
        print()

        return init_iter, get_params_dir, get_params_path, tracker_filename, \
               train_iter_filename, change_pseudo_tracker, change_1b_tracker, \
               change_2b_tracker, rmse_e_tracker, rmse_f_tracker, time_tracker, \
               rmse_e, rmse_f

    def fit_from_file(self,
                      filename: str,
                      subset: Collection,
                      weight: float = 0.5,
                      sparsity_reg: float = 0.0,
                      sparsity_epsilon: float = 1e-12,
                      batch_size=2500,
                      sample_weights: Dict = None,
                      energy_key="energy",
                      progress: str = "bar",
                      drop_columns: List[str] = None,
                      sparse_hdf5: bool = False,
                      max_iter: int = 1,
                      checkpoint: int = 10,
                      checkpoint_dir: str = ".",
                      params_filename: str = "alchemical_model_params.npz",
                      tracker_filename: str = "train_tracker.npz",
                      train_iter_filename: str = ".train_iter",
                      client = None,
                      resume: bool = False,
                      ):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and train the model parameters using alternating least-squaures
        of the alchemical spline coefficients and weighting factors.

        Args:
            filename (str): path to HDF5 file.
            subset (list): list of keys for training.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            sparse_reg (float): regularization strength for sparsity of
                pseudo-weights (L1 penalty). Defaults to 0.0.
            sparsity_epsilon (float): small value for L1 penalty to avoid
                division by zero. Defaults to 1e-12.
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            progress (str): style for progress indicators.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
            sparse_hdf5 (bool): whether the HDF5 features file is in sparse format.
            max_iter (int): maximum number of iterations for alternating
                least-squares optimization.
            checkpoint (int): frequency of checkpointing the training RMSE and
                saving parameters to disk.
            checkpoint_dir (str): directory for saving checkpoints.
            params_filename (str): filename for saving parameters during
                checkpoints. Will be saved to `checkpoint_dir/<iteration>/`.
            train_tracker (str): filename for saving training tracker during
                checkpoints. Will be saved to `checkpoint_dir/`.
            train_iter_filename (str): filename for writing current iteration
                number during checkpoints. Will be saved to `checkpoint_dir/`.
            client (concurrent.futures.Executor, dask.distributed.Client)
            resume (bool): whether to resume training from a previous checkpoint.
        """
        init_iter, get_params_dir, get_params_path, tracker_filename, \
        train_iter_filename, change_pseudo_tracker, change_1b_tracker, \
        change_2b_tracker, rmse_e_tracker, rmse_f_tracker, time_tracker, \
        rmse_e, rmse_f = \
            self.initialize_training(filename=filename,
                                     subset=subset,
                                     sparse_hdf5=sparse_hdf5,
                                     max_iter=max_iter,
                                     checkpoint=checkpoint,
                                     checkpoint_dir=checkpoint_dir,
                                     params_filename=params_filename,
                                     tracker_filename=tracker_filename,
                                     train_iter_filename=train_iter_filename,
                                     client=client,
                                     resume=resume,
                                     )

        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        n_tables, _, table_names, _ = io.analyze_hdf_tables(filename)

        # Outer-most ALS loop
        print(f"Beginning alternating least-squares optimization:")
        print(f"\tResume: {resume}")
        print(f"\tMax iterations: {max_iter}")
        print(f"\tCheckpoint frequency: {checkpoint}")
        print(f"\tParameters filename: {get_params_path('<iteration>')}")
        print(f"\tTracker filename: {tracker_filename}")
        print(f"\tTrain iteration filename: {train_iter_filename}")
        print()
        for i in range(init_iter, max_iter):
            print(f"Iteration {i+1}/{max_iter}")
            starttime = time.time()

            for param_to_fit in ("coeff", "pseudo_weights"):
                e_variance = VarianceRecorder()
                f_variance = VarianceRecorder()

                if param_to_fit == "coeff":
                    print(f"\tFitting alchemical spline coefficients.")
                    gram_e, gram_f, ord_e, ord_f = self.initialize_gramC_ordinateC()
                elif param_to_fit == "pseudo_weights":
                    print(f"\tFitting pseudo weights.")
                    gram_e, gram_f, ord_e, ord_f = self.initialize_gramW_ordinateW()
                else:
                    raise ValueError("Something went wrong.")

                table_iterator = parallel.progress_iter(np.arange(n_tables),
                                                        style=progress,
                                                        total=n_tables,
                                                        leave=False)
                for j in table_iterator:
                    table_name = table_names[j]
                    df = process.load_feature_db(filename, table_name, sparse_hdf5=sparse_hdf5)
                    keys = df.index.unique(level=0).intersection(subset)
                    if len(keys) == 0:
                        continue

                    if drop_columns != None:
                        df.drop(columns=drop_columns,inplace=True)

                    if param_to_fit == "coeff":
                        intermediates = self.gramC_from_df(df,
                                                         keys,
                                                         e_variance=e_variance,
                                                         f_variance=f_variance,
                                                         sample_weights=sample_weights,
                                                         energy_key=energy_key,
                                                         batch_size=batch_size)
                    elif param_to_fit == "pseudo_weights":
                        intermediates = self.gramW_from_df(df,
                                                         keys,
                                                         e_variance=e_variance,
                                                         f_variance=f_variance,
                                                         sample_weights=sample_weights,
                                                         energy_key=energy_key,
                                                         batch_size=batch_size)
                    else:
                        raise ValueError("Something went wrong.")
                    g_e, g_f, o_e, o_f = intermediates
                    gram_e += g_e
                    gram_f += g_f
                    ord_e += o_e
                    ord_f += o_f
                energy_weight, force_weight = calc_E_F_weights(e_variance.n,
                                                            f_variance.n,
                                                            e_variance.std,
                                                            f_variance.std)
                gram, ordinate = self.combine_weighted_gram(gram_e,
                                                            gram_f,
                                                            ord_e,
                                                            ord_f,
                                                            energy_weight,
                                                            force_weight,
                                                            weight)

                # Add regularization
                if param_to_fit == "coeff":
                    regularizer = np.dot(self.regularizer.T, self.regularizer)
                elif param_to_fit == "pseudo_weights":
                    regularizer = sparsity_reg_matrix(self.pseudo_weights.flatten(),
                                                      strength=sparsity_reg,
                                                      epsilon=sparsity_epsilon)
                else:
                    raise ValueError("Something went wrong.")
                gram += regularizer
                fitted_params = lu_factorization(gram, ordinate)

                if param_to_fit == "coeff":
                    old_coeff_1b = self.coeff_1b
                    old_coeff_2b = self.coeff_2b
                    self.coeff_1b = fitted_params[:self.n_elements]
                    self.coeff_2b = fitted_params[self.n_elements:].\
                        reshape(self.n_pseudo, self.n_basis).T
                elif param_to_fit == "pseudo_weights":
                    pseudo_weights = fitted_params.reshape(self.n_pairtypes, self.n_pseudo)
                    # normalize weights s.t. each column is between -1 and 1
                    normalization_factor = np.max(np.abs(pseudo_weights), axis=0)
                    pseudo_weights /= normalization_factor  # broadcasted over columns
                    self.coeff_2b *= normalization_factor  # not necessary if coeffs are trained again
                    max_change_pseudo = np.max(np.abs(pseudo_weights - self.pseudo_weights))
                    max_change_1b = np.max(np.abs(old_coeff_1b - self.coeff_1b))
                    max_change_2b = np.max(np.abs(
                        old_coeff_2b[self.frozen_2b_data_coverage] - 
                        self.coeff_2b[self.frozen_2b_data_coverage]
                        ))
                    change_1b_tracker[i] = max_change_1b
                    change_2b_tracker[i] = max_change_2b
                    change_pseudo_tracker[i] = max_change_pseudo
                    print(f"\tMax change in 1-body coefficients: {max_change_1b:.3E}")
                    print(f"\tMax change in 2-body coefficients: {max_change_2b:.3E}")
                    print(f"\tMax change in pseudo weights: {max_change_pseudo:.3E}")
                    self.pseudo_weights = pseudo_weights
                print()
            
            rmse_e_tracker[i] = rmse_e
            rmse_f_tracker[i] = rmse_f

            endtime = time.time()
            elapsed_time = endtime - starttime
            time_tracker[i+1] = elapsed_time + time_tracker[i]
            print(f"\tTime elapsed: {elapsed_time:.3F} seconds\n")
        
            # Checkpoint
            if ((i+1) % checkpoint == 0) or (i+1 == max_iter):
                print("Checkpointing.")

                # Decompress and store to self.coefficients
                self.decompress_alchemical_parameters()

                # Check RMSE
                _, _, _, _, rmse_e, rmse_f = self.batched_predict(filename,
                                                        keys=subset,
                                                        sparse_hdf5=sparse_hdf5,
                                                        client=client,
                                                        )
                if i+1 == max_iter:
                    rmse_e_tracker[-1] = rmse_e
                    rmse_f_tracker[-1] = rmse_f

                os.makedirs(get_params_dir(i+1), exist_ok=True)
                np.savez(get_params_path(i+1),
                         coeff_1b=self.coeff_1b,
                         coeff_2b=self.coeff_2b,
                         pseudo_weights=self.pseudo_weights)

                np.savez(tracker_filename,
                         change_pseudo=change_pseudo_tracker,
                         change_1b=change_1b_tracker,
                         change_2b=change_2b_tracker,
                         rmse_e=rmse_e_tracker,
                         rmse_f=rmse_f_tracker,
                         frozen_2b_data_coverage=self.frozen_2b_data_coverage,
                         time=time_tracker)

                with open(train_iter_filename, "w") as f:
                    f.write(f"{i+1}\n")
                print()

    def decompress_alchemical_parameters(self):
        """Decompress the alchemical spline coefficients and store to self.coefficients."""
        coefficients = (self.coeff_2b @ self.pseudo_weights.T).flatten(order="F")
        coefficients = np.concatenate([self.coeff_1b, coefficients])
        coefficients = revert_frozen_coefficients(coefficients,
                                                self.n_feats,
                                                self.mask,
                                                self.frozen_c,
                                                self.col_idx)
        self.coefficients = coefficients

    def initialize_gramC_ordinateC(self):
        """Initialize empty matrices for gram matrices and ordinates for fitting
        the alchemical spline coefficients."""
        n_columns = self.n_elements + self.n_basis * self.n_pseudo
        gram_e = np.zeros((n_columns, n_columns))
        ord_e = np.zeros(n_columns)
        gram_f = np.zeros((n_columns, n_columns))
        ord_f = np.zeros(n_columns)
        return gram_e, gram_f, ord_e, ord_f

    def gramC_from_df(self,
                      df: pd.DataFrame,
                      keys: Collection,
                      e_variance: VarianceRecorder = None,
                      f_variance: VarianceRecorder = None,
                      sample_weights: Dict = None,
                      energy_key: str = "energy",
                      batch_size: int = 2500):
        """
        Extract inputs and outputs from dataframe and compute
        moore-penrose components (gram matrices and ordinates) for
        training the alchemical spline coefficients.

        Args:
            df (pd.DataFrame): DataFrame of energy/force features.
            keys (list): keys to query from df (e.g. training subset).
            e_variance (VarianceRecorder): handler for accumulating
                statistics for energies (mean and variance).
            f_variance (VarianceRecorder): handler for accumulating
                statistics for forces (mean and variance).
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
        """
        x_e, y_e, x_f, y_f = freeze_columns_from_df(df,
                                                    keys,
                                                    self.n_elements,
                                                    self.mask,
                                                    self.frozen_c,
                                                    self.col_idx,
                                                    energy_key=energy_key,
                                                    sample_weights=sample_weights,
                                                    )
        if e_variance is not None and f_variance is not None:
            e_variance.update(y_e)
            f_variance.update(y_f)
        data_coverage_2b_e = (np.sum(x_e[:, self.n_elements:], axis=0) != 0).reshape(self.n_pairtypes, self.n_basis)
        data_coverage_2b_e = np.any(data_coverage_2b_e, axis=0)
        data_coverage_2b_f = (np.sum(x_f[:, self.n_elements:], axis=0) != 0).reshape(self.n_pairtypes, self.n_basis)
        data_coverage_2b_f = np.any(data_coverage_2b_f, axis=0)
        self.frozen_2b_data_coverage = np.logical_or(self.frozen_2b_data_coverage,
                                                    data_coverage_2b_e)
        self.frozen_2b_data_coverage = np.logical_or(self.frozen_2b_data_coverage,
                                                    data_coverage_2b_f)
        
        WX_e = self.feature_matrixC(x_e, self.pseudo_weights)
        WX_f = self.feature_matrixC(x_f, self.pseudo_weights)

        gram_e, ordinate_e = batched_moore_penrose(WX_e,
                                                   y_e,
                                                   batch_size=batch_size)
        gram_f, ordinate_f = batched_moore_penrose(WX_f,
                                                   y_f,
                                                   batch_size=batch_size)
        return gram_e, gram_f, ordinate_e, ordinate_f
    
    def feature_matrixC(self, X, W):
        """
        Given the frozen UF3 feature matrix X of shape
        (n_data, n_e + n_basis * n_pairtypes) and the pseudo_weights W of shape
        (n_pairs, n_pseudo), compute the feature matrix WX for training the
        alchemical spline coefficients C.

        Args:
            X (np.ndarray): frozen UF3 feature matrix of shape 
                (n_data, n_e + n_basis * n_pairtypes)
            W (np.ndarray): pseudo_weights of shape (n_pairs, n_pseudo)

        Returns:
            WX (np.ndarray): feature matrix for training the alchemical spline
                coefficients C
        """
        X1 = X[:, :self.n_elements]  # 1-body features
        X2 = X[:, self.n_elements:]  # 2-body features
        n_data, _ = np.shape(X2)
        X2_tensor = X2.reshape(n_data, self.n_pairtypes, self.n_basis)
        WX = broad_row_krp_sum(W, X2_tensor)
        WX = np.hstack((X1, WX))  # append 1-body features
        return WX

    def initialize_gramW_ordinateW(self):
        """Initialize empty matrices for gram matrices and ordinates for fitting
        the pseudo_weights."""
        n_columns = self.n_pairtypes * self.n_pseudo
        gram_e = np.zeros((n_columns, n_columns))
        ord_e = np.zeros(n_columns)
        gram_f = np.zeros((n_columns, n_columns))
        ord_f = np.zeros(n_columns)
        return gram_e, gram_f, ord_e, ord_f

    def gramW_from_df(self,
                      df: pd.DataFrame,
                      keys: Collection,
                      e_variance: VarianceRecorder = None,
                      f_variance: VarianceRecorder = None,
                      sample_weights: Dict = None,
                      energy_key: str = "energy",
                      batch_size: int = 2500):
        """
        Extract inputs and outputs from dataframe and compute
        moore-penrose components (gram matrices and ordinates) for
        training the pseudo_weights.

        Args:
            df (pd.DataFrame): DataFrame of energy/force features.
            keys (list): keys to query from df (e.g. training subset).
            e_variance (VarianceRecorder): handler for accumulating
                statistics for energies (mean and variance).
            f_variance (VarianceRecorder): handler for accumulating
                statistics for forces (mean and variance).
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            batch_size (int): batch size, in rows, for matrix multiplication
                operations in constructing gram matrices.
        """
        x_e, y_e, x_f, y_f = freeze_columns_from_df(df,
                                                    keys,
                                                    self.n_elements,
                                                    self.mask,
                                                    self.frozen_c,
                                                    self.col_idx,
                                                    energy_key=energy_key,
                                                    sample_weights=sample_weights,
                                                    )
        if e_variance is not None and f_variance is not None:
            e_variance.update(y_e)
            f_variance.update(y_f)
        
        XC_e, yhat_e_1b = self.feature_matrixW(x_e, self.coeff_2b, self.coeff_1b)
        XC_f = self.feature_matrixW(x_f, self.coeff_2b)

        gram_e, ordinate_e = batched_moore_penrose(XC_e,
                                                   y_e - yhat_e_1b,
                                                   batch_size=batch_size)
        gram_f, ordinate_f = batched_moore_penrose(XC_f,
                                                   y_f,
                                                   batch_size=batch_size)
        return gram_e, gram_f, ordinate_e, ordinate_f

    def feature_matrixW(self, X, C2, C1=None):
        """
        Given the frozen UF3 feature matrix X of shape
        (n_data, n_e + n_basis * n_pairtypes) and the 2-body alchemical spline
        coefficients C2 of shape (n_basis, n_pseudo), compute the feature matrix
        XC for training the pseudo_weights W.

        If the 1-body alchemical spline coefficients C1 are provided, their
        contribution is also returned to be subtracted from the target values.
        Necessary for energies if the 1-body offset is being fit. Not necessary
        for forces.

        Args:
            X (np.ndarray): frozen UF3 feature matrix of shape 
                (n_data, n_e + n_basis * n_pairtypes)
            C2 (np.ndarray): 2-body alchemical spline coefficients of shape
                (n_basis, n_pseudo)
            C1 (np.ndarray): 1-body alchemical spline coefficients of shape
                (n_elements,)

        Returns:
            XC (np.ndarray): feature matrix for training the pseudo_weights W
        """
        X1 = X[:, :self.n_elements]
        X2 = X[:, self.n_elements:]
        n_data, _ = np.shape(X2)
        X2_tensor = X2.reshape(n_data, self.n_pairtypes, self.n_basis)
        XC = X2_tensor @ C2  # (n_data, n_pairtypes, n_pseudo)
        XC = XC.reshape(n_data, self.n_pairtypes * self.n_pseudo)
        if C1 is None:
            return XC
        else:
            return XC, X1 @ C1


class AlchemicalModelTorch(AlchemicalModel, torch.nn.Module):
    """
    Alchemical learning ("pseudo-interaction") model for fitting energies and
    forces using PyTorch.

    XXX: currently only 2-body interactions and all pseudo-interactions
    must have the same spline construction and offsets are fit.

    XXX: self.data_coverage is not implemented yet.

    XXX: the regularizer matrix should already have frozen coefficients removed.
    """
    def __init__(self,
                 bspline_config,
                 n_pseudo,
                 regularizer=None,
                 data_coverage=None,
                 init_params=None,
                 dtype=torch.float64,
                 **params):
        AlchemicalModel.__init__(self,
                                 bspline_config,
                                 n_pseudo,
                                 regularizer,
                                 data_coverage,
                                 init_params,
                                 **params)
        torch.nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.dtype = dtype
        self.convert2torch()

    def __repr__(self):
        if self.coefficients is None:
            fit = "False"
        else:
            fit = "True"
        summary = ["AlchemicalModelTorch:",
                   f"    Fit: {fit}",
                   f"    n_pseudo: {self.n_pseudo}",
                   f"    n_basis: {self.n_basis}",
                   self.bspline_config.__repr__()
                   ]
        return "\n".join(summary)

    def convert2torch(self):
        """Convert numpy array attributes to torch tensors."""
        self.coeff_1b = torch.tensor(self.coeff_1b, device=self.device,
                                     dtype=self.dtype, requires_grad=True)
        self.coeff_2b = torch.tensor(self.coeff_2b, device=self.device,
                                     dtype=self.dtype, requires_grad=True)
        self.pseudo_weights = torch.tensor(self.pseudo_weights, device=self.device,
                                           dtype=self.dtype, requires_grad=True)
        self.regularizer = torch.tensor(self.regularizer, device=self.device,
                                        dtype=self.dtype, requires_grad=False)

    def parameters(self):
        return [self.coeff_1b, self.coeff_2b, self.pseudo_weights]

    def decompress_alchemical_parameters(self, write2self=False):
        """Redefining parent method but with torch tensors."""
        coefficients = (self.pseudo_weights @ self.coeff_2b.T).view(-1)  # row-wise version
        coefficients = torch.cat([self.coeff_1b, coefficients])
        if write2self:
            coefficients = coefficients.detach().cpu().numpy()
            coefficients = revert_frozen_coefficients(coefficients,
                                                    self.n_feats,
                                                    self.mask,
                                                    self.frozen_c,
                                                    self.col_idx)
            self.coefficients = coefficients
        else:
            return coefficients

    def forward(self, x):
        """
        Forward pass for the alchemical model.

        Args:
            x (torch.Tensor): input tensor of shape (n_data, n_elements + n_basis * n_pairtypes)

        Returns:
            y (torch.Tensor): output tensor of shape (n_data,)
        """
        decompressed_params = self.decompress_alchemical_parameters(write2self=False)
        y = torch.matmul(x, decompressed_params)
        return y
    
    def normalize_parameters(self):
        """Normalize the alchemical model parameters."""
        normalization_factor = torch.max(torch.abs(self.pseudo_weights))
        self.pseudo_weights /= normalization_factor
        self.coeff_2b *= normalization_factor

    def regularization_loss(self,
                            sparse_reg: float = 0.0):
        """
        Compute the regularization loss for the alchemical model.

        Args:
            sparse_reg (float): regularization strength for sparsity of
                pseudo-weights (L1 penalty). Defaults to 0.0.

        Returns:
            loss (torch.Tensor): regularization loss
        """
        alchemical_coeffs = self.coeff_2b.T.contiguous().view(-1)  # TODO: redefine coeff_2b to be row-wise for each pseudo-interaction
        alchemical_coeffs = torch.cat([self.coeff_1b, alchemical_coeffs])
        loss = torch.matmul(self.regularizer, alchemical_coeffs)
        loss = torch.sum(loss**2)
        loss += sparse_reg * torch.sum(torch.abs(self.pseudo_weights))
        return loss

    def train_from_file(self,
                        filename: str,
                        subset: Collection,
                        weight: float = 0.5,
                        sparsity_reg: float = 0.0,
                        batch_size=1,
                        sample_weights: Dict = None,
                        energy_key="energy",
                        progress: str = "bar",
                        drop_columns: List[str] = None,
                        sparse_hdf5: bool = False,
                        max_epochs: int = 1,
                        optimizer: torch.optim.Optimizer = None,
                        checkpoint: int = 10,
                        checkpoint_dir: str = ".",
                        shuffle: bool = True,
                        drop_last: bool = False,
                        dataloader_n_workers: int = 0,
                        params_filename: str = "alchemical_model_params.npz",
                        tracker_filename: str = "train_tracker.npz",
                        train_iter_filename: str = ".train_iter",
                        ):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and train the Alchemical model parameters using a gradient-based
        optimizer.

        Args:
            filename (str): path to HDF5 file.
            subset (list): list of keys for training.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            sparse_reg (float): regularization strength for sparsity of
                pseudo-weights (L1 penalty). Defaults to 0.0.
            batch_size (int): batch size, in number of tables from HDF5 file,
                for PyTorch DataLoader.
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            progress (str): style for progress indicators.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
            sparse_hdf5 (bool): whether the HDF5 features file is in sparse format.
            max_epochs (int): maximum number of iterations for alternating
                least-squares optimization.
            optimizer (torch.optim.Optimizer): optimizer for training. If None,
                defaults to Adam.
            checkpoint (int): frequency of checkpointing the training RMSE and
                saving parameters to disk.
            checkpoint_dir (str): directory for saving checkpoints.
            shuffle (bool): shuffle the dataset during training.
            drop_last (bool): drop the last incomplete batch during training.
            dataloader_n_workers (int): number of workers for PyTorch DataLoader.
            params_filename (str): filename for saving parameters during
                checkpoints.
            train_tracker (str): filename for saving training tracker during
                checkpoints.
            train_iter_filename (str): filename for writing current iteration
                number during checkpoints.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)  # no error if exists
        get_params_dir = lambda i: os.path.join(checkpoint_dir, str(i))
        get_params_path = lambda i: os.path.join(get_params_dir(i), params_filename)
        tracker_filename = os.path.join(checkpoint_dir, tracker_filename)
        train_iter_filename = os.path.join(checkpoint_dir, train_iter_filename)
        if os.path.exists( get_params_path(1) ):
            warnings.warn(f"Warning: {get_params_path(1)} already exists. It will be overwritten")
        if os.path.exists(tracker_filename):
            warnings.warn(f"Warning: {tracker_filename} already exists. It will be overwritten")

        self.frozen_2b_data_coverage = np.zeros(self.n_basis, dtype=bool)
        change_pseudo_tracker = np.full((max_epochs,), np.nan)
        change_1b_tracker = np.full((max_epochs,), np.nan)
        change_2b_tracker = np.full((max_epochs,), np.nan)
        # RMSE at the beginning of the epoch is recorded
        rmse_e_tracker = np.full((max_epochs+1,), np.nan)  # +1 for initial RMSE
        rmse_f_tracker = np.full((max_epochs+1,), np.nan)  # +1 for initial RMSE
        time_tracker = np.full((max_epochs+1,), np.nan)
        time_tracker[0] = 0

        # Optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        else:
            optimizer = optimizer

        # Create PyTorch DataLoader
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        _, _, table_names, _ = io.analyze_hdf_tables(filename)
        assert batch_size == 1  # XXX: for now
        dataloader = torch_util.hdf5_dataloader(filename,
                                                table_names,
                                                subset,
                                                batch_size=batch_size,
                                                sparse_hdf5=sparse_hdf5,
                                                shuffle=shuffle,
                                                drop_last=drop_last,
                                                num_workers=dataloader_n_workers,
                                                )
        n_batches = len(dataloader)

        # Outer-most training loop
        self.train()  # set model to training mode
        print(f"Beginning training:")
        print(f"\tDevice: {self.device}")
        print(f"\tMax epochs: {max_epochs}")
        print(f"\tDataLoader batch size: {batch_size}")
        print(f"\tDataLoader workers: {dataloader_n_workers}")
        print(f"\tShuffle: {shuffle}")
        print(f"\tDrop last: {drop_last}")
        print(f"\tOptimizer: {optimizer}")
        print(f"\tCheckpoint frequency: {checkpoint}")
        print(f"\tParameters filename: {get_params_path('<iteration>')}")
        print(f"\tTracker filename: {tracker_filename}")
        print(f"\tTrain iteration filename: {train_iter_filename}")
        print()
        for i in range(max_epochs):
            print(f"Iteration {i+1}/{max_epochs}")
            starttime = time.time()

            # if doing stochastic optimization, we need to do these each time
            # the parameters are updated
            e_variance = VarianceRecorder()
            f_variance = VarianceRecorder()
            optimizer.zero_grad()
            loss_e = 0.0
            loss_f = 0.0
            #self.normalize_parameters()
            old_pseudo_weights = self.pseudo_weights.detach().clone()
            old_coeff_1b = self.coeff_1b.detach().clone()
            old_coeff_2b = self.coeff_2b.detach().clone()

            table_iterator = parallel.progress_iter(dataloader,
                                                    style=progress,
                                                    total=n_batches,
                                                    leave=False)
            for dfs in table_iterator:
                df = dfs[0]  # we contrained batch_size=1 above
                keys = df.index.unique(level=0).intersection(subset)
                if len(keys) == 0:
                    continue
                if drop_columns != None:
                    df.drop(columns=drop_columns,inplace=True)
                
                x_e, y_e, x_f, y_f = freeze_columns_from_df(df,
                                                keys,
                                                self.n_elements,
                                                self.mask,
                                                self.frozen_c,
                                                self.col_idx,
                                                energy_key=energy_key,
                                                sample_weights=sample_weights,
                                                )    
                # if we allowed more than one batch, we would need to stack them here
                e_variance.update(y_e)
                f_variance.update(y_f)
                data_coverage_2b_e = (np.sum(x_e[:, self.n_elements:], axis=0) != 0).reshape(self.n_pairtypes, self.n_basis)
                data_coverage_2b_e = np.any(data_coverage_2b_e, axis=0)
                data_coverage_2b_f = (np.sum(x_f[:, self.n_elements:], axis=0) != 0).reshape(self.n_pairtypes, self.n_basis)
                data_coverage_2b_f = np.any(data_coverage_2b_f, axis=0)
                self.frozen_2b_data_coverage = np.logical_or(self.frozen_2b_data_coverage,
                                                            data_coverage_2b_e)
                self.frozen_2b_data_coverage = np.logical_or(self.frozen_2b_data_coverage,
                                                            data_coverage_2b_f)
                
                # Accumulate losses
                x_e = torch.tensor(x_e, device=self.device, dtype=self.dtype,
                                   requires_grad=False)
                y_e = torch.tensor(y_e, device=self.device, dtype=self.dtype,
                                   requires_grad=False)
                p_e = self(x_e)
                loss_e += torch.nn.functional.mse_loss(p_e, y_e, reduction="sum")
                x_f = torch.tensor(x_f, device=self.device, dtype=self.dtype,
                                   requires_grad=False)
                y_f = torch.tensor(y_f, device=self.device, dtype=self.dtype,
                                   requires_grad=False)
                p_f = self(x_f)
                loss_f += torch.nn.functional.mse_loss(p_f, y_f, reduction="sum")
                
            # Compute total loss
            energy_weight, force_weight = calc_E_F_weights(e_variance.n,
                                                           f_variance.n,
                                                           e_variance.std,
                                                           f_variance.std)
            loss, _ = self.combine_weighted_gram(loss_e, loss_f, 0, 0,
                                                 energy_weight, force_weight, weight)
            loss += self.regularization_loss(sparsity_reg)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Record progress
            rmse_e = torch.sqrt(loss_e.data / e_variance.n)
            rmse_f = torch.sqrt(loss_f.data / f_variance.n)
            rmse_e_tracker[i] = rmse_e.item()
            rmse_f_tracker[i] = rmse_f.item()
            max_change_1b = torch.max(torch.abs(self.coeff_1b - old_coeff_1b)).item()
            max_change_2b = torch.max(torch.abs(self.coeff_2b - old_coeff_2b)).item()
            max_change_pseudo = torch.max(torch.abs(self.pseudo_weights - old_pseudo_weights)).item()
            change_1b_tracker[i] = max_change_1b
            change_2b_tracker[i] = max_change_2b
            change_pseudo_tracker[i] = max_change_pseudo
            print(f"\tMax change in 1-body coefficients: {max_change_1b:.5E}")
            print(f"\tMax change in 2-body coefficients: {max_change_2b:.5E}")
            print(f"\tMax change in pseudo weights: {max_change_pseudo:.5E}")
            print(f"\tRMSE energy (eV/atom): {rmse_e:.5E}")
            print(f"\tRMSE force (eV/A): {rmse_f:.5E}")
            
            endtime = time.time()
            elapsed_time = endtime - starttime
            time_tracker[i+1] = elapsed_time + time_tracker[i]
            print(f"\tTime elapsed (s): {elapsed_time:.3F}\n")

            # Checkpoint
            if ((i+1) % checkpoint == 0) or (i+1 == max_epochs):
                print("Checkpointing.")

                # Decompress and store to self.coefficients
                self.decompress_alchemical_parameters(write2self=True)

                if i+1 == max_epochs:
                    # Final RMSE check
                    self.eval()  # set model to evaluation mode
                    print()
                    print("Final RMSE check.")
                    _, _, _, _, rmse_e, rmse_f = self.batched_predict(filename,
                                                            keys=subset,
                                                            sparse_hdf5=sparse_hdf5,
                                                            )
                    rmse_e_tracker[-1] = rmse_e
                    rmse_f_tracker[-1] = rmse_f

                os.makedirs(get_params_dir(i+1), exist_ok=True)
                #torch.save(self.state_dict(), get_params_path(i+1))
                np.savez(get_params_path(i+1),
                         coeff_1b=self.coeff_1b.detach().cpu().numpy(),
                         coeff_2b=self.coeff_2b.detach().cpu().numpy(),
                         pseudo_weights=self.pseudo_weights.detach().cpu().numpy())

                np.savez(tracker_filename,
                            change_pseudo=change_pseudo_tracker,
                            change_1b=change_1b_tracker,
                            change_2b=change_2b_tracker,
                            rmse_e=rmse_e_tracker,
                            rmse_f=rmse_f_tracker,
                            time=time_tracker)

                with open(train_iter_filename, "w") as f:
                    f.write(f"{i+1}\n")
                print()

    def fit_from_file(self, *args, **kwargs):
        warnings.warn("Calling fit_from_file() of AlchemcalModelTorch.\n"
                      "Redirecting to train_from_file().")
        return self.train_from_file(*args, **kwargs)


def get_spline_taylor_expansion(r_target,
                                r,
                                coefficients,
                                knot_sequence,
                                min_curvature=0.0):
    nd3 = ndsplines.NDSpline([knot_sequence], coefficients, 3)
    y_trace = nd3(r_target, nus=0)
    d1_trace = nd3(r_target, nus=1)
    d2_trace = nd3(r_target, nus=2)
    if min_curvature is not None:
        d2_trace = max(d2_trace, min_curvature)
    dr = r - r_target
    y = y_trace + (d1_trace * dr) + (0.5 * d2_trace * dr ** 2)
    return y


def dataframe_to_tuples(df_features,
                        n_elements=None,
                        energy_key='energy',
                        sample_weights=None,
                        ):
    """
    Extract energy/force inputs/outputs from DataFrame.

    Args:
        df_features (pd.DataFrame): dataframe with target vector (y) as the
            first column and feature vectors (x) as remaining columns.
        n_elements (int): number of leading columns to consider for size
            normalization.
        energy_key (str): key for energy samples, used to slice df_features
            into energies and forces for weight generation.
        sample_weights (dict):

    Returns:
        x (np.ndarray): features for machine learning.
        y (np.ndarray): target vector.
        w (np.ndarray): weight vector for machine learning.
    """
    names = df_features.index.get_level_values(0)
    y_index = df_features.index.get_level_values(-1)
    energy_mask = (y_index == energy_key)
    force_mask = np.logical_not(energy_mask)
    data = df_features.to_numpy()
    y = data[:, [0]]
    x = data[:, 1:]
    y_e = y[energy_mask]
    y_f = y[force_mask]

    if n_elements is not None:        
        s = np.sum(x[energy_mask, :n_elements], axis=1).reshape(-1, 1)
        x_e = x[energy_mask] / s  # row-wise normalization
        y_e = y_e / s  # both y_e and s are 2D arrays with shape (n, 1) here
    else:
        x_e = x[energy_mask]
    x_f = x[force_mask]

    if sample_weights is not None:
        w = np.array([sample_weights.get(name, 1.0) for name in names]).\
                reshape(-1, 1)
        w_e = w[energy_mask]
        w_f = w[force_mask]
        x_e *= w_e
        y_e *= w_e
        x_f *= w_f
        y_f *= w_f

    # Flatten y to 1D
    y_e = y_e.flatten()
    y_f = y_f.flatten()

    return x_e, y_e, x_f, y_f


def moore_penrose_components(x, y):
    """
    Compute gram matrix (x^T x) and ordinate (x^T y).

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.

    Returns:
        a: Gram matrix (X'X)
        b: ordinate (X'y)
    """
    a = np.dot(x.T, x)
    b = np.dot(x.T, y)
    return a, b


def batched_moore_penrose(x, y, batch_size=2500):
    """
    Batched evaluation of gram matrix (x^T x) and ordinate (x^T y).

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.
        batch_size: maximum batch size, default 2500 rows. This option
            should be adjusted based on efficiency/memory tradeoffs.

    Returns:
        a: Gram matrix (X'X)
        b: ordinate (X'y)
    """

    n_samples, n_features = np.shape(x)
    n_batches = int(n_samples / batch_size)
    if n_batches <= 1:
        return moore_penrose_components(x, y)
    else:
        batched_idx = np.array_split(np.arange(len(y)), n_batches)
        gram = np.zeros((n_features, n_features))
        ordinate = np.zeros(n_features)
        for j, batch in enumerate(batched_idx):
            x_x, x_y = moore_penrose_components(x[batch], y[batch])
            gram += x_x
            ordinate += x_y
        return gram, ordinate


def lu_factorization(a, b):
    """
    LU factorization for least-squares solution using np.linalg.solve().

    Args:
        a: coefficients (X) or Gram matrix (X'X)
        b: ordinate (X'y)
    """
    return np.linalg.solve(a, b)


def linear_least_squares(x, y):
    """
    Solves the linear least-squares problem Ax=y. Regularizer matrix
    should be concatenated to x and zero-values padded to y.

    Args:
        x (np.ndarray): input matrix of shape (n_samples, n_features).
        y (np.ndarray): output vector of length n_samples.

    Returns:
        solution (np.ndarray): coefficients.
    """
    a, b = moore_penrose_components(x, y)
    return lu_factorization(a, b)


def weighted_least_squares(x, y, weights=None, regularizer=None):
    """
    Solves the linear least-squares problem with optional Tikhonov regularizer
    matrix and optional weighting.
    TODO: Remove (deprecated)

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        weights (np.ndarray): sample weights (optional).
        regularizer (np.ndarray): Tikhonov regularizer matrix.

    Returns:
        solution (np.ndarray): coefficients.
        predictions (list of np.ndarray): predictions.
    """
    x_fit, y_fit = apply_weights(x, y, weights)
    n_feats = len(x[0])
    if regularizer is not None:  # append regularizer
        # validate_regularizer(regularizer, n_feats)
        reg_zeros = np.zeros(len(regularizer))
        x_fit = np.concatenate([x_fit, regularizer])
        y_fit = np.concatenate([y_fit, reg_zeros])
    solution = linear_least_squares(x_fit, y_fit)
    return solution


def get_freezing_mask(n_feats: int, col_idx: np.ndarray) -> np.ndarray:
    """
    Freezing mask is the set difference between the range of feature indices
    and the indices to be excluded (col_idx).

    Args:
        n_feats (int): number of features.
        col_idx (list): list of indices to be masked.

    Returns:
        mask (np.ndarray): set of non-frozen indices.
    """
    mask = np.setdiff1d(np.arange(n_feats), col_idx)
    return mask


def freeze_columns(x: np.ndarray,
                   y: np.ndarray,
                   mask: np.ndarray,
                   frozen_c: np.ndarray,
                   col_idx: np.ndarray,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Freeze coefficients of the solution (e.g. forcing to zero) by
    simultaneously eliminating columns of the input and their assumed
    contribution to the output.

    Args:
        x (np.ndarray): input matrix.
        y (np.ndarray): output vector.
        mask (np.ndarray): set of non-frozen indices.
        frozen_c (np.ndarray): values of coefficients to be frozen.
        col_idx (np.ndarray): indices of coefficients to be frozen.

    Returns:
        x (np.ndarray): input matrix without frozen columns.
        y (np.ndarray): output vector, minus frozen contributions.
    """
    x_fixed = x[:, col_idx]
    x = x[:, mask]
    y = np.subtract(y, np.dot(x_fixed, frozen_c))
    return x, y


def freeze_columns_from_df(df: pd.DataFrame,
                           keys: Collection,
                           n_elements: int,
                           mask: np.ndarray,
                           frozen_c: np.ndarray,
                           col_idx: np.ndarray,
                           energy_key: str = "energy",
                           sample_weights: Dict = None,
                           ) -> Tuple[np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """
    Convenience function for freezing columns from DataFrame.

    Args:
        df (pd.DataFrame): DataFrame of energy/force features.
        keys (list): keys to query from df (e.g. training subset).
        n_elements (int): number of leading columns to consider for size
            normalization.
        mask (np.ndarray): set of non-frozen indices.
        frozen_c (np.ndarray): values of coefficients to be frozen.
        col_idx (np.ndarray): indices of coefficients to be frozen.
        energy_key (str): column name for energies, default "energy".
        sample_weights (dict): sample weights (optional).

    Returns:
        x_e (np.ndarray): input energy matrix without frozen columns.
        y_e (np.ndarray): output energy vector, minus frozen contributions.
        x_f (np.ndarray): input force matrix without frozen columns.
        y_f (np.ndarray): output force vector, minus frozen contributions.
    """
    x_e, y_e, x_f, y_f = dataframe_to_tuples(df.loc[keys],
                                             n_elements=n_elements,
                                             energy_key=energy_key,
                                             sample_weights=sample_weights,
                                             )
    x_e, y_e = freeze_columns(x_e,
                              y_e,
                              mask,
                              frozen_c,
                              col_idx)
    x_f, y_f = freeze_columns(x_f,
                              y_f,
                              mask,
                              frozen_c,
                              col_idx)
    return x_e, y_e, x_f, y_f


def freeze_regularizer(regularizer: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """Apply freezing mask to regularizer, eliminating masked columns."""
    regularizer = regularizer[:, mask]
    return regularizer


def revert_frozen_coefficients(solution: np.ndarray,
                               n_coeff: int,
                               mask: Collection[bool],
                               frozen_c: Collection[float],
                               frozen_idx: Collection[int],) -> np.ndarray:
    """
    Reverse freezing operations by arranging learned coefficients
    and frozen coefficients using the mask.

    Args:
        solution: learned solution, excluding frozen coefficients
        n_coeff: number of columns in full (unfrozen) solution
        mask: indices of remaining columns in x.
        frozen_idx: column indices of fixed coefficients.
        frozen_c: frozen coefficients.

    Returns:
        full_solution (np.ndarray)
    """
    full_solution = np.zeros(n_coeff)
    np.put_along_axis(full_solution, mask, solution, 0)
    np.put_along_axis(full_solution, frozen_idx, frozen_c, 0)
    return full_solution


def apply_weighted_gram(gram_matrix: np.ndarray,
                        weight: float) -> np.ndarray:
    """Deprecated utility function for weighting gram matrix."""
    return gram_matrix * weight**2


def apply_weights(x, y, weights):
    """Deprecated utility function for weighting inputs/outputs."""
    if weights is not None:
        if len(weights) != len(x):
            raise ValueError(
                'Number of weights does not match number of samples.')
        if not np.all(weights >= 0):
            raise ValueError('Negative weights provided.')
        w = np.sqrt(weights)
        x_fit = np.multiply(x.T, w).T
        y_fit = np.multiply(y, w)
    else:
        x_fit = x
        y_fit = y
    return x_fit, y_fit


def validate_regularizer(regularizer: np.ndarray, n_feats: int):
    """
    Check for consistency between regularizer matrix and number of features.

    Args:
        regularizer (np.ndarray): regularizer matrix.
        n_feats (int): number of features.
    """
    n_row, n_col = regularizer.shape
    if n_col != n_feats:
        shape_comparison = "N x {0}. Provided: {1} x {2}".format(n_feats,
                                                                 n_row,
                                                                 n_col)
        raise ValueError(
            "Expected regularizer shape: " + shape_comparison)


def subset_prediction(df: pd.DataFrame,
                      model: WeightedLinearModel,
                      subset_keys: Collection = None,
                      **kwargs
                      ) -> Tuple:
    """
    Convenience function for optimization workflow. Read inputs/outputs
    from DataFrame and predict using fitted model.

    Args:
        df (pd.DataFrame): DataFrame of inputs/outputs.
        model (WeightedLinearModel): fitted model.
        subset_keys (list): list of keys to query from DataFrame.

    Returns:
        y_e (np.ndarray): target values for energies.
        p_e (np.ndarray): prediction values for energies.
        y_f (np.ndarray): target values for forces.
        p_f (np.ndarray): prediction values for forces.
    """
    if subset_keys is not None:
        idx = df.index.unique(level=0).intersection(subset_keys)
        if len(idx) == 0:
            return list(), list(), list(), list()
        df = df.loc[idx]
    x_e, y_e, x_f, y_f = dataframe_to_tuples(df,
                                             **kwargs)
    p_e = model.predict(x_e)
    p_f = model.predict(x_f)
    return y_e, p_e, y_f, p_f


def batched_prediction(table_names: Collection,
                       model: WeightedLinearModel,
                       filename: str,
                       subset_keys: Collection = None,
                       drop_columns: List[str] = None,
                       sparse_hdf5: bool = False,
                       **kwargs):
    """
    Convenience function for optimization workflow. Read inputs/outputs
    from HDF5 file and predict using fitted model.

    Args:
        model (WeightedLinearModel): fitted model.
        filename (str): path to HDF5 file.
        table_names (list): list of table names to query from HDF5 file.
        subset_keys (list): list of keys to query from DataFrame.
        drop_columns (list): list of columns to drop. Used when modifying
            the cutoffs of the feature vectors from HDF5 file. No internal
            checks are performed to see if dropping provided columns produce
            features of the intended cutoffs. Use with Caution.
        sparse_hdf5 (bool): whether the HDF5 features file is in sparse format.

    Returns:
        y_e (np.ndarray): target values for energies.
        p_e (np.ndarray): prediction values for energies.
        y_f (np.ndarray): target values for forces.
        p_f (np.ndarray): prediction values for forces.
    """
    df_batches = io.dataframe_batch_loader(filename, table_names, sparse_hdf5=sparse_hdf5)
    y_e = []
    p_e = []
    y_f = []
    p_f = []
    for df in df_batches:
        if drop_columns != None:
            df.drop(columns=drop_columns,inplace=True)

        predictions = subset_prediction(df,
                                        model,
                                        subset_keys=subset_keys,
                                        **kwargs)
        y_e.append(predictions[0])
        p_e.append(predictions[1])
        y_f.append(predictions[2])
        p_f.append(predictions[3])
    y_e = np.concatenate(y_e)
    p_e = np.concatenate(p_e)
    y_f = np.concatenate(y_f)
    p_f = np.concatenate(p_f)
    return y_e, p_e, y_f, p_f


def batched_prediction_parallel(model: WeightedLinearModel,
                                filename: str,
                                table_names: Collection = None,
                                subset_keys: Collection = None,
                                drop_columns: List[str] = None,
                                sparse_hdf5: bool = False,
                                client = None,
                                shuffle: bool = False,
                                progress: str = "bar",
                                **kwargs):
    """
    Parallelized version of batched_prediction().

    Args:
        filename (str): path to HDF5 file.
        model (WeightedLinearModel): fitted model.
        table_names (list): list of table names to query from HDF5 file.
        subset_keys (list): list of keys to query from DataFrame.
        drop_columns (list): list of columns to drop. Used when modifying
            the cutoffs of the feature vectors from HDF5 file. No internal
            checks are performed to see if dropping provided columns produce
            features of the intended cutoffs. Use with Caution.
        sparse_hdf5 (bool): whether the HDF5 features file is in sparse format.
        client (concurrent.futures.Executor, dask.distributed.Client)
        shuffle (bool): shuffle the dataset during prediction.
        progress (str): style for progress indicators.

    Returns:
        y_e (np.ndarray): target values for energies.
        p_e (np.ndarray): prediction values for energies.
        y_f (np.ndarray): target values for forces.
        p_f (np.ndarray): prediction values for forces.
    """
    if table_names is None:
        _, _, table_names, _ = io.analyze_hdf_tables(filename)
    else:
        table_names = copy.copy(table_names)

    if parallel.USE_DASK:
        client_info = client.scheduler_info()
        n_jobs = client_info['workers'] * client_info['nthreads']
    elif client is None:
        n_jobs = 1
    else:
        n_jobs = client._max_workers

    if n_jobs < 2 or client is None:
        warnings.warn("Processing in serial.", RuntimeWarning)
        return batched_prediction(table_names,
                                  model,
                                  filename,
                                  subset_keys=subset_keys,
                                  drop_columns=drop_columns,
                                  sparse_hdf5=sparse_hdf5,
                                  **kwargs,
                                  )
    if shuffle:
        np.random.shuffle(table_names)
    batches = parallel.split_zip(n_jobs, table_names)[0]
    try:
        batches = [client.scatter(batch) for batch in batches]
    except AttributeError:
        pass

    future_list = parallel.batch_submit(batched_prediction,
                                        batches,
                                        client,
                                        model=model,
                                        filename=filename,
                                        subset_keys=subset_keys,
                                        drop_columns=drop_columns,
                                        sparse_hdf5=sparse_hdf5,
                                        **kwargs,
                                        )
    results_tuple = parallel.gather_and_merge(future_list,
                                              client=client,
                                              cancel=True,
                                              progress=progress)
    try:
        for batch in batches:
            client.cancel(batch)
    except AttributeError:
        pass
    return results_tuple  # y_e, p_e, y_f, p_f


def rmse_metric(predicted: Collection,
                actual: Collection) -> float:
    """
    Root-mean-square error metric.

    Args:
        predicted (list): prediction values.
        actual (list): reference values.

    Returns:
        root-mean-square-error metric.
    """
    return np.sqrt(np.mean(np.subtract(predicted, actual) ** 2))


def mae_metric(predicted, actual):
    """
    Mean-absolute error metric.

    Args:
        predicted (list): prediction values.
        actual (list): reference values.

    Returns:
        mean-absolute error metric.
    """
    return np.mean(np.abs(np.subtract(predicted, actual)))


def arrange_coefficients(coefficients, bspline_config):
    """
    Arrange coefficients by degree of interaction.

    Args:
        coefficients (np.ndarray): Flattened vector of coefficients.
            Partitioned by provided bspline_config per degree.
        bspline_config (bspline.BSplineBasis)

    Returns:
        solutions (dict): fit coefficients per degree.
    """
    split_indices = np.cumsum(bspline_config.partition_sizes)[:-1]
    solutions_list = np.array_split(coefficients,
                                    split_indices)
    element_list = bspline_config.element_list
    solutions = {element: value[0] for element, value
                 in zip(element_list, solutions_list[:len(element_list)])}
    solutions_list = solutions_list[len(element_list):]

    j = 0
    for d in range(2, bspline_config.degree + 1):
        interactions_map = bspline_config.interactions_map[d]
        for interaction in interactions_map:
            solutions[interaction] = solutions_list[j]
            j += 1
    return solutions


def postprocess_coefficients_2b(coefficients,
                                core_hardness=2.0,
                                min_core=2.0,
                                min_slope=0.1,
                                rounding_factor=3,
                                smooth_cutoff=False,
                                in_place=False):
    """
    Postprocess 2B coefficients to enforce repulsive core.

    Args:
        coefficients (np.ndarray): vector of 2B coefficients.
        core_hardness (float): power base factor for hard-core correction.
        min_core (float): minimum energy barrier at the lower-bound (eV).
        min_slope (float): minimum core slope at peak (eV).
        rounding_factor (float): decimal for rounding in extrema search.
        smooth_cutoff (bool): whether to fix the last two coefficients to
            zero, forcing the second derivative to be zero at the upper bound.
        in_place (bool): whether to modify in-place or make a copy.

    Returns:
        coefficients (np.ndarray): new vector of coefficients.
    """
    if not in_place:  # apply corrections to a copy
        coefficients = np.array(coefficients)
    well_idx = find_pair_potential_well(coefficients, rounding_factor)
    if well_idx > 1:
        # search for maximum left of potential well, rounding to meV (default)
        peak_search = np.round(coefficients[:well_idx], rounding_factor)
        # bias towards well with imperceptible slope to deal with plateau
        peak_search += np.arange(len(peak_search)) * 10**(-2 * rounding_factor)
        gradient = np.gradient(peak_search)
        peak_idx = np.argmax(peak_search)
        if np.all(gradient[:peak_idx] >= 0):
            # correction for case where lower-bound is far below
            # observations and coefficients are nearly zero.
            for i in np.arange(peak_idx)[::-1]:
                value = np.abs(coefficients[i + 1]) * core_hardness
                value = max(value, min_slope)
                coefficients[i] = value
    if coefficients[0] < min_core:
        # fail-safe hard core by simply fixing the first coefficient
        coefficients[0] = min_core
    if smooth_cutoff:
        coefficients[-2:] = 0
    return coefficients


def find_pair_potential_well(coefficients, rounding_factor):
    """
    Identify coefficient index corresponding to possible potential well.
    Intermediate function for postprocess_coefficients_2b().

    Args:
        coefficients: vector of two-body coefficients.
        rounding_factor: decimal for rounding in extrema search.

    Returns:
        well_idx: approximate location of potential well in coefficients
    """
    peak_idx = np.argmax(coefficients)
    well_idx = np.argmin(coefficients)
    if well_idx < peak_idx:
        # if well is left of peak, either core may not be well-defined
        # or well may not be well-defined
        well_search = np.round(coefficients[:peak_idx], rounding_factor)
        if np.ptp(well_search) < 10 ** -(rounding_factor - 1):
            # no actual well
            well_idx = peak_idx + 1
    return well_idx


def calc_E_F_weights(n_e, n_f, std_e, std_f):
    """
    Calculates weights applied to energy and force components of the
    least-squares problem (excluding kappa, which is applied in
    self.combine_weighted_gram()).

    Args:
        n_e (int): number of energy samples.
        n_f (int): number of force samples.
        e_stddev (float): standard deviation of energy samples.
        f_stddev (float): standard deviation of force samples.

    Returns:
        energy_weight (float): weight applied to energy components.
        force_weight (float): weight applied to force components.
    """
    if std_e == 0:  # single point or really bad dataset
        energy_weight = 1.0
        force_weight = 1 / np.sqrt(n_f)
    else:
        energy_weight = 1 / np.sqrt(n_e) / std_e
        force_weight = 1 / np.sqrt(n_f) / std_f
    return energy_weight, force_weight


def broad_row_krp_sum(A, B):
    """
    Broadcasted Khatri-Rao product of rows between A and slices of B along its
    0-th axis, with a summation along the 1st axis and a squeeze at the end.
    Used in the AlchemicalModel for computating the feature matrix for the
    alchemical spline coefficient fitting.

    Args:
        A (np.ndarray): first matrix of shape (m, n).
        B (np.ndarray): second matrix of shape (r, m, s).

    Returns:
        result (np.ndarray): the result of shape (r, n * s).
    """
    result = np.vstack(
        [
            np.sum(scipy.linalg.khatri_rao(A.T, B[i, :, :].T).T, axis=0)
            for i in range(B.shape[0])
            ]
        )
    return result


def sparsity_reg_matrix(params, strength, epsilon=1e-12):
    """
    Generate a sparsity regularization matrix for given parameters (L1 reg).

    Args:
        params (np.ndarray): parameters to be regularized.
        strength (float): regularization strength.
        epsilon (float): small value to prevent division by zero.

    Returns:
        reg_matrix (np.ndarray): regularization matrix.
    """
    params = np.where(params < epsilon, epsilon, params)
    reg_matrix = np.diag(1/params) * strength
    return reg_matrix


def expand_alchemical_init_params(params_dict, n_pseudo):
    """
    Using the pseudo-interaction coefficients and weighting factors for a
    small `n_pseudo`, create initial parameters for a larger `n_pseudo`.

    Args:
        params_dict (dict | np.lib.npyio.NpzFile'): dictionary of initial parameters.
            Keys should include `coeff_2b` and `pseudo_weights` (others optional)
        n_pseudo (int): number of pseudo-interactions in the expanded system.
            
    Returns:
        expanded_params (dict): dictionary of expanded parameters.
    """
    assert "coeff_2b" in params_dict
    assert "pseudo_weights" in params_dict
    coeff_2b_old = params_dict["coeff_2b"]
    pseudo_weights_old = params_dict["pseudo_weights"]
    n_pseudo_old = pseudo_weights_old.shape[1]
    assert n_pseudo_old == coeff_2b_old.shape[1]
    expanded_params = dict()
    for key, value in params_dict.items():
        expanded_params[key] = value.copy()
    if n_pseudo_old >= n_pseudo:
        return expanded_params
    n_basis = coeff_2b_old.shape[0]
    n_pairtypes = pseudo_weights_old.shape[0]
    coeff_2b_new = np.zeros((n_basis, n_pseudo))
    pseudo_weights_new = np.zeros((n_pairtypes, n_pseudo))
    coeff_2b_new[:, :n_pseudo_old] = coeff_2b_old
    pseudo_weights_new[:, :n_pseudo_old] = pseudo_weights_old

    # generate new pseudo-interaction coefficients using a convolution method
    for j in range(n_pseudo_old, n_pseudo):
        random_pseudo_idx = np.random.randint(0, n_pseudo_old)
        convolve_width = np.random.randint(2, int(n_basis / 4))
        convolve_window = np.random.rand(convolve_width) * 2 / convolve_width  # expectation of sum is 1
        convolved = np.convolve(coeff_2b_old[:, random_pseudo_idx], convolve_window, mode="full")
        random_cut = np.random.randint(0, convolve_width)
        coeff_2b_new[:, j] = convolved[random_cut:random_cut+n_basis]
    
    expanded_params["coeff_2b"] = coeff_2b_new
    expanded_params["pseudo_weights"] = pseudo_weights_new  # new weights are 0

    return expanded_params