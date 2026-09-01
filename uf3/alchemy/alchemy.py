from typing import List, Dict, Collection, Callable, Union
import os, time, warnings, re, gc, datetime, mmap
import numpy as np
import scipy
import scipy.optimize
from numba import jit
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: torch import failed. AlchemicalModelTorch will not be available.")
    TORCH_AVAILABLE = False
import tables
import fasteners
try:
    from mpi4py import MPI
    GLOBAL_USE_MPI = True
except ImportError:
    print("MPI not available: trouble importing mpi4py.")
    GLOBAL_USE_MPI = False
from uf3.data import io
from uf3.representation import bspline, process
from uf3.regression import least_squares as ls
from uf3.regression import regularize
from uf3.util import user_config, parallel
if TORCH_AVAILABLE:
    from uf3.util import torch_util


class AlchemicalModel(ls.WeightedLinearModel):
    """
    Alchemical learning ("pseudo-interaction") model for fitting energies and
    forces. All pseudo-interactions must have the same spline construction and
    offsets are fit.

    Note about spline coefficient regularization:
    * Alchemical spline coefficient regularizers ("C_regularizers"):
        * Should be in the form of a dictionary with keys as interaction orders
            (1, 2, ..., degree) and values as 2D arrays of shape (_, n_elements)
            for 1-body and (_, n_basis[i]) for i-body. For i>1, the same regularizer
            will be applied to all alchemical interactions of that order.
        * The regularizer matrices should already have frozen coefficients removed.

    Args:
        bspline_config (bspline.BSplineBasis): basis set configuration.
        n_pseudo (int | dict): number of pseudo-interactions to fit.
            If int, the same n_pseudo is used for all interaction order.
            If dict, the keys are interaction orders and values are n_pseudo.
        data_coverage (np.ndarray): boolean array for data coverage.
    """
    def __init__(self,
                 bspline_config: bspline.BSplineBasis,
                 n_pseudo: Union[int, Dict[int, int]],
                 data_coverage: np.ndarray = None,
                 **args):
        regularizer = 'n/a'  # loaded during training
        super().__init__(bspline_config, regularizer, data_coverage, **args)
        default_n_pseudo = {2: 3, 3: 2}
        self.n_pseudo = user_config.process_order_dict(n_pseudo, default_n_pseudo)
        component_sizes = self.bspline_config.get_interaction_partitions()[0]

        # temporary sanity checks
        for pair in self.bspline_config.interactions_map[2]:
            if not component_sizes[pair] == self.n_basis[2] + self.bspline_config.leading_trim + self.bspline_config.trailing_trim:
                raise ValueError("Inconsistent component sizes.")
        for i in range(3, self.degree+1):
            for ituple in self.bspline_config.interactions_map[i]:
                if not component_sizes[ituple] == self.n_basis[i]:
                    raise ValueError("Inconsistent component sizes.")
        assert self.bspline_config.offset_1b  # fit 1-body

    @property
    def degree(self):
        return self.bspline_config.degree

    @property
    def n_basis(self):
        component_sizes = self.bspline_config.get_interaction_partitions()[0]
        n_basis = {i: component_sizes[self.bspline_config.interactions_map[i][0]]
                   for i in range(2, self.degree+1)}
        n_basis[2] = n_basis[2] - self.bspline_config.leading_trim - \
                self.bspline_config.trailing_trim
        return n_basis

    @property
    def n_elements(self):
        return len(self.bspline_config.element_list)
    
    @property
    def n_ituples(self):
        return {i: len(self.bspline_config.interactions_map[i]) for i in
                range(2, self.degree+1)}

    @property
    def real_coeff_offsets(self):
        """
        Offset indices for coefficients of different interaction orders.
        """
        body_order_sizes = np.array([self.n_basis[i] * self.n_ituples[i]
                                     for i in range(2, self.degree+1)])
        body_order_offsets = np.cumsum(body_order_sizes)
        body_order_offsets = np.insert(body_order_offsets, 0, 0) + self.n_elements
        return {i: body_order_offsets[i-2] for i in range(2, self.degree+2)}

    @property
    def alchemical_coeff_offsets(self):
        """
        Offset indices for alchemical coefficients of different interaction orders.
        """
        body_order_sizes = np.array([self.n_pseudo[i] * self.n_basis[i]
                                     for i in range(2, self.degree+1)])
        body_order_offsets = np.cumsum(body_order_sizes)
        body_order_offsets = np.insert(body_order_offsets, 0, 0) + self.n_elements
        return {i: body_order_offsets[i-2] for i in range(2, self.degree+2)}
    
    @property
    def pseudo_offsets(self):
        """
        Offset indices for pseudo-interactions of different interaction orders.
        """
        body_order_sizes = np.array([self.n_pseudo[i] * self.n_ituples[i]
                                     for i in range(2, self.degree+1)])
        body_order_offsets = np.cumsum(body_order_sizes)
        body_order_offsets = np.insert(body_order_offsets, 0, 0)
        return {i: body_order_offsets[i-2] for i in range(2, self.degree+2)}

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
            self.coeff = {1: np.zeros(self.n_elements)}
            self.coeff.update({ i: np.zeros((self.n_basis[i], self.n_pseudo[i]))
                               for i in range(2, self.degree+1)})
            self.pseudo_weights = { i: np.random.rand(self.n_ituples[i], self.n_pseudo[i])*2-1
                                   for i in range(2, self.degree+1)}
        else:
            self.coeff = dict()
            self.pseudo_weights = dict()
            self.coeff[1] = init_params['coeff_1b']
            if self.coeff[1].shape != (self.n_elements,):
                raise ValueError("Incorrect shape for 1-body coefficients.\n"
                                 f"\tExpected: ({self.n_elements},)\n"
                                 f"\tProvided: {self.coeff[1].shape}")
            for i in range(2, self.degree+1):
                self.coeff[i] = init_params[f'coeff_{i}b']
                if self.coeff[i].shape != (self.n_basis[i], self.n_pseudo[i]):
                    raise ValueError(f"Incorrect shape for {i}-body coefficients.\n"
                                     f"\tExpected: ({self.n_basis[i]}, {self.n_pseudo[i]})\n"
                                     f"\tProvided: {self.coeff[i].shape}")
                self.pseudo_weights[i] = init_params[f'pseudo_weights_{i}b']
                if self.pseudo_weights[i].shape != (self.n_ituples[i], self.n_pseudo[i]):
                    raise ValueError(f"Incorrect shape for pseudo weights.\n"
                                     f"\tExpected: ({self.n_ituples[i]}, {self.n_pseudo[i]})\n"
                                     f"\tProvided: {self.pseudo_weights[i].shape}")

    def save_alchemical_params(self, path):
        """Save alchemical parameters to a file."""
        np.savez(path,
                 **{f"coeff_{k}b": self.coeff[k]
                     for k in range(1, self.degree+1)},
                 **{f"pseudo_weights_{k}b": self.pseudo_weights[k]
                     for k in range(2, self.degree+1)},
                 )

    def load_alchemical_params(self, path):
        """Load alchemical parameters from a file."""
        params = np.load(path)
        self.initialize_parameters(params)

    def save_data_coverage(self, path):
        """Save data coverage arrays to a file."""
        np.savez(path,
            data_coverage=self.data_coverage,
            **{f'frozen_{k}b_data_coverage': self.frozen_data_coverage[k]
            for k in range(1, self.degree+1)},
            )

    def load_data_coverage(self, path):
        """Load data coverage arrays from a file and set class attributes."""
        data = np.load(path)
        data_coverage = data['data_coverage']
        frozen_data_coverage = {i: data[f'frozen_{i}b_data_coverage']
                                for i in range(1, self.degree+1)}
        assert len(data_coverage) == self.n_feats
        assert all([len(frozen_data_coverage[i]) == self.n_basis[i]
                    for i in range(2, self.degree+1)])
        assert len(frozen_data_coverage[1]) == self.n_elements
        self.data_coverage = data_coverage
        self.frozen_data_coverage = frozen_data_coverage

    def preprocess_for_training(self,
                                filename: str,
                                subset: Collection,
                                weight: float = 0.5,
                                e_variance: ls.VarianceRecorder = None,
                                f_variance: ls.VarianceRecorder = None,
                                sample_weights: Dict = None,
                                energy_key: str = "energy",
                                progress: str = "bar",
                                drop_columns: List[str] = None,
                                sparse_hdf5: bool = False,
                                epsilon: float = 1e-12,
                                preprocessed_file: str = 'preprocessed.h5',
                                coverage_file: str = 'coverage.npz',
                                metadata_file: str = 'metadata.npz',
                                USE_MPI: bool = False,
                                ):
        """
        Preprocess data and collect metadata for efficient training, including:
            - extracting only the training subset
            - removing/freezing appropriate feature columns to remove them from
                the dataset
            - splitting dataset according to interaction order
            - ensuring C-style contiguous arrays
            - accumulating statistics for energies and forces, including means
                and variances, and calculating weights for energy and forces
            - accumulating data coverage and frozen data coverage for each
                interaction order
            - saving statistics and data coverages to disk
            - XXX: append real regularizer to dataset (do later)
            - saving preprocessed data to disk (data as CSR arrays)

        Args:
            filename (str): path to HDF5 file.
            subset (list): list of keys for training.
            weight (float): parameter balancing contribution from energies
                vs. forces. Higher values favor energies; defaults to 0.5.
            e_variance (ls.VarianceRecorder): handler for accumulating
                statistics for energies (mean and variance).
            f_variance (ls.VarianceRecorder): handler for accumulating
                statistics for forces (mean and variance).
            sample_weights (dict):
            energy_key (str): column name for energies, default "energy".
            progress (str): style for progress indicators.
            drop_columns (list): list of columns to drop. Used when modifying
                the cutoffs of the feature vectors from HDF5 file. No internal
                checks are performed to see if dropping provided columns produce
                features of the intended cutoffs. Use with Caution.
            sparse_hdf5 (bool): whether the HDF5 features file is in sparse format.
            epsilon (float): small value for filtering frozen data coverage.
            preprocessed_file (str): filename to save preprocessed data.
            coverage_file (str): filename to save data coverage arrays.
            metadata_file (str): filename to save dataset statistics.
            USE_MPI (bool): whether to use MPI for parallel processing.
        """
        lock = fasteners.InterProcessLock(preprocessed_file+".lock")  # for parallel writing
        if USE_MPI and not GLOBAL_USE_MPI:
            raise ImportError("MPI not available. Exiting...")
        if USE_MPI:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1

        ### FIRST PASS ###
        # initialize variables
        if e_variance is None:
            e_variance = ls.VarianceRecorder()
        if f_variance is None:
            f_variance = ls.VarianceRecorder()
        frozen_data_coverage = {i: np.zeros(self.n_basis[i], dtype=bool)
                                for i in range(2, self.degree+1)}
        frozen_data_coverage[1] = np.zeros(self.n_elements, dtype=bool)
        # total number of columns after removing unused columns
        # contrast with self.n_feats, which is before removing unused columns
        n_columns_total = self.n_elements + sum([self.n_basis[i] * self.n_ituples[i]
                                                 for i in range(2, self.degree+1)])
        data_coverage = np.zeros(n_columns_total, dtype=bool)


        # loop over batches
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        _, _, table_names, _ = io.analyze_hdf_tables(filename)
        if USE_MPI:  # distribute table names
            if rank == 0:
                np.random.shuffle(table_names)
                sublists = np.array_split(table_names, size)
            else:
                sublists = None
            table_names = comm.scatter(sublists, root=0)
        n_tables = len(table_names)
        if rank == 0:  # show progress only on rank 0
            table_iterator = parallel.progress_iter(np.arange(n_tables),
                                                    style=progress,
                                                    total=n_tables,
                                                    leave=True)
        else:
            table_iterator = np.arange(n_tables)
        skip_tables = []
        for j in table_iterator:
            table_name = table_names[j]
            df = process.load_feature_db(filename, table_name, sparse_hdf5=sparse_hdf5)
            keys = df.index.unique(level=0).intersection(subset)
            if len(keys) == 0:
                skip_tables.append(table_name)
                continue
            if drop_columns != None:
                df.drop(columns=drop_columns,inplace=True)

            x_e, y_e, x_f, y_f = ls.dataframe_to_tuples(df.loc[keys],
                                                        n_elements=self.n_elements,
                                                        energy_key=energy_key,
                                                        sample_weights=sample_weights,
                                                        )
            x_e, y_e = ls.freeze_columns(x_e,
                                         y_e,
                                         self.mask,
                                         self.frozen_c,
                                         self.col_idx)
            x_f, y_f = ls.freeze_columns(x_f,
                                         y_f,
                                         self.mask,
                                         self.frozen_c,
                                         self.col_idx)
            e_variance.update(y_e)
            f_variance.update(y_f)

            x_es = {1: x_e[:, :self.n_elements]}
            data_coverage_e = np.any(np.abs(x_es[1]) > epsilon, axis=0)
            frozen_data_coverage[1] = np.logical_or(frozen_data_coverage[1],
                                                    data_coverage_e)
            data_coverage[:self.n_elements] = np.logical_or(data_coverage[:self.n_elements],
                                                            data_coverage_e)
            x_fs = {1: x_f[:, :self.n_elements]}  # should be all zeros, but leave here for readability

            for i in range(2, self.degree+1):
                idx_lo = self.real_coeff_offsets[i]
                idx_hi = self.real_coeff_offsets[i+1]
                x_es[i] = x_e[:, idx_lo:idx_hi]
                x_fs[i] = x_f[:, idx_lo:idx_hi]

                data_coverage_e = np.any(np.abs(x_es[i]) > epsilon, axis=0)
                data_coverage[idx_lo:idx_hi] = np.logical_or(data_coverage[idx_lo:idx_hi],
                                                             data_coverage_e)
                data_coverage_e = data_coverage_e.reshape(self.n_ituples[i], self.n_basis[i])
                data_coverage_e = np.any(data_coverage_e, axis=0)
                data_coverage_f = np.any(np.abs(x_fs[i]) > epsilon, axis=0)
                data_coverage[idx_lo:idx_hi] = np.logical_or(data_coverage[idx_lo:idx_hi],
                                                             data_coverage_f)
                data_coverage_f = data_coverage_f.reshape(self.n_ituples[i], self.n_basis[i])
                data_coverage_f = np.any(data_coverage_f, axis=0)
                frozen_data_coverage[i] = np.logical_or(frozen_data_coverage[i],
                                                        data_coverage_e)
                frozen_data_coverage[i] = np.logical_or(frozen_data_coverage[i],
                                                        data_coverage_f)

            # save preprocessed energy and force features
            lock.acquire()
            save_preprocessed_db(x_es, y_e, x_fs, y_f, preprocessed_file,
                                degree=self.degree,
                                batch_name=table_name,
                                )
            lock.release()

        # set data coverages as class attributes
        # reduce data_coverage
        if USE_MPI:
            data_coverage = comm.reduce(data_coverage, op=np.logical_or, root=0)
            for i in frozen_data_coverage:
                frozen_data_coverage[i] = comm.reduce(frozen_data_coverage[i], op=np.logical_or, root=0)
        if rank == 0:
            data_coverage = ls.revert_frozen_coefficients(data_coverage,
                                                          self.n_feats,
                                                          self.mask,
                                                          self.frozen_c,
                                                          self.col_idx)
            self.data_coverage = np.logical_or(self.data_coverage, data_coverage)
            self.frozen_data_coverage = frozen_data_coverage
            self.save_data_coverage(coverage_file)

        ### SECOND PASS ###
        # gather n, mean, and std from e_variance and f_variance
        if USE_MPI:
            e_ns = np.array(comm.allgather(e_variance.n))
            e_means = np.array(comm.allgather(e_variance.mean))
            e_stds = np.array(comm.allgather(e_variance.std))
            e_variance = ls.VarianceRecorder()
            e_variance.update_with_stats(e_means, e_stds, e_ns)
            f_ns = np.array(comm.allgather(f_variance.n))
            f_means = np.array(comm.allgather(f_variance.mean))
            f_stds = np.array(comm.allgather(f_variance.std))
            f_variance = ls.VarianceRecorder()
            f_variance.update_with_stats(f_means, f_stds, f_ns)

        # calculate energy and force weights and store to metadata file
        if rank == 0:
            energy_weight, force_weight = ls.calc_E_F_weights(e_variance.n,
                                                            f_variance.n,
                                                            e_variance.std,
                                                            f_variance.std)
            energy_weight *= np.sqrt(weight)
            force_weight *= np.sqrt(1 - weight)
            np.savez(metadata_file,
                     w_e=energy_weight,
                     w_f=force_weight,
                     n_e=e_variance.n,
                     mean_e=e_variance.mean,
                     std_e=e_variance.std,
                     n_f=f_variance.n,
                     mean_f=f_variance.mean,
                     std_f=f_variance.std,
                     )

    def initialize_training(self,
                            preprocessed_file: str,
                            coverage_file: str,
                            metadata_file: str,
                            init_params: Union[dict, np.lib.npyio.NpzFile],
                            max_iter: int,
                            checkpoint: int,
                            checkpoint_dir: str,
                            params_filename: str,
                            tracker_filename: str,
                            train_iter_filename: str,
                            resume: bool,
                            fit_first: str,
                            ):
        """
        Initialization for training.
        """
        if not os.path.isfile(preprocessed_file):
            raise FileNotFoundError(f"'{preprocessed_file}' not found.\n"
                "Please run `preprocess_for_training()` first.\n"
                "Exiting...")
        if not os.path.isfile(coverage_file):
            raise FileNotFoundError(f"'{coverage_file}' not found.\n"
                "Please run `preprocess_for_training()` first.\n"
                "Exiting...")
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(f"'{metadata_file}' not found.\n"
                "Please run `preprocess_for_training()` first.\n"
                "Exiting...")
        metadata = np.load(metadata_file)
        w_e = metadata['w_e']
        w_f = metadata['w_f']
        n_e = metadata['n_e']
        n_f = metadata['n_f']
        self.load_data_coverage(coverage_file)  # load and set data coverage attributes
        with tables.open_file(preprocessed_file, mode="r") as f:
            table_names = [group._v_name for group in f.list_nodes("/")]

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
            change_pw_tracker = {i: train_tracker[f'change_pw_{i}b']
                                 for i in range(2, self.degree+1)}
            if not all([len(change_pw_tracker[i]) == max_iter for i in range(2, self.degree+1)]):
                warnings.warn(f"Attempting to resume with a different `max_iter` than before.\n"
                                f"Before: {len(change_pw_tracker[2])} iterations\n"
                                f"Now: {max_iter} iterations")
                for i in range(2, self.degree+1):
                    change_pw_tracker[i] = np.concatenate((change_pw_tracker[i],
                                                np.full(max_iter - len(change_pw_tracker[i]),
                                                        np.nan)))
            change_coeff_tracker = {i: train_tracker[f'change_coeff_{i}b']
                                    for i in range(1, self.degree+1)}
            if not all([len(change_coeff_tracker[i]) == max_iter for i in range(1, self.degree+1)]):
                warnings.warn(f"Attempting to resume with a different `max_iter` than before.\n"
                                f"Before: {len(change_coeff_tracker[1])} iterations\n"
                                f"Now: {max_iter} iterations")
                for i in range(1, self.degree+1):
                    change_coeff_tracker[i] = np.concatenate((change_coeff_tracker[i],
                                                np.full(max_iter - len(change_coeff_tracker[i]),
                                                        np.nan)))
            time_tracker = train_tracker["time"]
            if not len(time_tracker) == max_iter + 1:
                warnings.warn(f"Attempting to resume with a different `max_iter` than before.\n"
                                f"Before: {len(time_tracker)-1} iterations\n"
                                f"Now: {max_iter} iterations")
                time_tracker = np.concatenate((time_tracker,
                                                np.full(max_iter - len(time_tracker) + 1,
                                                        np.nan)))
            rmse_e_tracker = train_tracker["rmse_e"]
            if not len(rmse_e_tracker) == 2*max_iter:
                warnings.warn(f"Attempting to resume with a different `max_iter` than before.\n"
                                f"Before: {len(rmse_e_tracker)//2} iterations\n"
                                f"Now: {max_iter} iterations")
                rmse_e_tracker = np.concatenate((rmse_e_tracker,
                                                np.full(2*max_iter - len(rmse_e_tracker),
                                                        np.nan)))
            rmse_f_tracker = train_tracker["rmse_f"]
            if not len(rmse_f_tracker) == 2*max_iter:
                warnings.warn(f"Attempting to resume with a different `max_iter` than before.\n"
                                f"Before: {len(rmse_f_tracker)//2} iterations\n"
                                f"Now: {max_iter} iterations")
                rmse_f_tracker = np.concatenate((rmse_f_tracker,
                                                np.full(2*max_iter - len(rmse_f_tracker),
                                                        np.nan)))
            data_loss_tracker = train_tracker["data_loss"]
            if not len(data_loss_tracker) == 2*max_iter:
                warnings.warn(f"Attempting to resume with a different `max_iter` than before.\n"
                                f"Before: {len(data_loss_tracker)//2} iterations\n"
                                f"Now: {max_iter} iterations")
                data_loss_tracker = np.concatenate((data_loss_tracker,
                                                np.full(2*max_iter - len(data_loss_tracker),
                                                        np.nan)))
            total_loss_tracker = train_tracker["total_loss"]
            if not len(total_loss_tracker) == 2*max_iter:
                warnings.warn(f"Attempting to resume with a different `max_iter` than before.\n"
                                f"Before: {len(total_loss_tracker)//2} iterations\n"
                                f"Now: {max_iter} iterations")
                total_loss_tracker = np.concatenate((total_loss_tracker,
                                                np.full(2*max_iter - len(total_loss_tracker),
                                                        np.nan)))

            # find iteration to resume from
            try:
                init_iter = np.where(np.isnan(time_tracker))[0][0] - 1
            except IndexError:
                raise ValueError(f"Could not find a NaN in {time_tracker}.\n"
                                 f"Please set `resume=False` to start from scratch.\n"
                                    "Exiting...")
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

            # Parameter arrays for training.
            # After training, they will be stored to self.coefficients.
            self.initialize_parameters(init_params)
            os.makedirs(get_params_dir(0), exist_ok=True)
            self.save_alchemical_params(get_params_path(0))

            change_pw_tracker = {i: np.full(max_iter, np.nan) for i in range(2, self.degree+1)}
            change_coeff_tracker = {i: np.full(max_iter, np.nan) for i in range(1, self.degree+1)}
            time_tracker = np.full((max_iter+1,), np.nan)
            time_tracker[0] = 0

            rmse_e_tracker = np.full((2*max_iter,), np.nan)
            rmse_f_tracker = np.full((2*max_iter,), np.nan)
            data_loss_tracker = np.full((2*max_iter,), np.nan)
            total_loss_tracker = np.full((2*max_iter,), np.nan)

        if fit_first == "C":
            params_fit_order = ("coeff", "pseudo_weights")
        elif fit_first == "W":
            params_fit_order = ("pseudo_weights", "coeff")
        else:
            raise ValueError(f"Unrecognized value for `fit_first`: {fit_first}\n"
                                "Expected 'C' or 'W'.")

        return init_iter, get_params_dir, get_params_path, tracker_filename, \
               train_iter_filename, change_pw_tracker, \
               change_coeff_tracker, time_tracker, rmse_e_tracker, rmse_f_tracker, \
               data_loss_tracker, total_loss_tracker, params_fit_order, table_names, \
               w_e, w_f, n_e, n_f

    def fit_from_file(self,
                      preprocessed_file: str = "preprocessed.h5",
                      coverage_file: str = "coverage.npz",
                      metadata_file: str = "metadata.npz",
                      init_params: Union[dict, np.lib.npyio.NpzFile] = None,
                      C_regularizers: Dict = None,
                      C_reg_free: bool = False,
                      W_sparsity_reg: float = 0.0,
                      W_sparsity_epsilon: float = 1e-12,
                      progress: str = "bar",
                      max_iter: int = 1,
                      checkpoint: int = 10,
                      checkpoint_dir: str = "./checkpoint",
                      params_filename: str = "alchemical_model_params.npz",
                      tracker_filename: str = "train_tracker.npz",
                      train_iter_filename: str = ".train_iter",
                      resume: bool = False,
                      fit_first: str = "C",
                      solver: Callable = np.linalg.solve,
                      sparse_tables: bool = False,
                      cache_tables: bool = False,
                      ):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and train the model parameters using alternating least-squaures
        of the alchemical spline coefficients and weighting factors.

        Args:
            preprocessed_file (str): path to preprocessed HDF5 data file created
                by `preprocess_for_training()`.
            coverage_file (str): path to data coverages file created by
                `preprocess_for_training()`.
            metadata_file (str): path to metadata file created by
                `preprocess_for_training()`.
            init_params (dict | np.lib.npyio.NpzFile ): initial parameters for training.
                Should be an object with keys 'coeff_1b', 'coeff_2b', 'pseudo_weights_2b'
                (2b) and 'coeff_3b', 'pseudo_weights_3b' (3b).
            C_regularizers (Dict): dictionary of regularization matrices for
                alchemical spline coefficients. See the class docstring for
                more information about the format.
            C_reg_free (bool): whether the c_i is regularized in the regularization
                term of the loss function (True) or c_i * ||w_i|| (False). Defaults
                to False.
            W_sparse_reg (float): regularization strength for sparsity of
                pseudo-weights (L1 penalty). Defaults to 0.0.
            W_sparsity_epsilon (float): small value for L1 penalty to avoid
                division by zero. Defaults to 1e-12.
            progress (str): style for progress indicators.
            max_iter (int): maximum number of iterations for alternating
                least-squares optimization.
            checkpoint (int): frequency of saving parameters to disk.
            checkpoint_dir (str): directory for saving checkpoints.
            params_filename (str): filename for saving parameters during
                checkpoints. Will be saved to `checkpoint_dir/<iteration>/`.
            train_tracker (str): filename for saving training tracker during
                checkpoints. Will be saved to `checkpoint_dir/`.
            train_iter_filename (str): filename for writing current iteration
                number during checkpoints. Will be saved to `checkpoint_dir/`.
            resume (bool): whether to resume training from a previous checkpoint.
            fit_first (str): which parameters to fit first. Options are "C" for
                coefficients and "W" for pseudo-weights. Defaults to "C".
            solver (Callable): linear algebra solver to use. Defaults to
                `np.linalg.solve`.
            sparse_tables (bool): keep feature tables sparse and use sparse
                contractions to build the effective feature matrices.
            cache_tables (bool): keep loaded tables in memory across ALS
                iterations instead of re-reading them from disk.
        """
        ### Initialize training ###
        init_iter, get_params_dir, get_params_path, tracker_filename, \
        train_iter_filename, change_pw_tracker, \
        change_coeff_tracker, time_tracker, rmse_e_tracker, rmse_f_tracker, \
        data_loss_tracker, total_loss_tracker, params_fit_order, table_names, \
        w_e, w_f, n_e, n_f = \
            self.initialize_training(preprocessed_file=preprocessed_file,
                                     coverage_file=coverage_file,
                                     metadata_file=metadata_file,
                                     init_params=init_params,
                                     max_iter=max_iter,
                                     checkpoint=checkpoint,
                                     checkpoint_dir=checkpoint_dir,
                                     params_filename=params_filename,
                                     tracker_filename=tracker_filename,
                                     train_iter_filename=train_iter_filename,
                                     resume=resume,
                                     fit_first=fit_first,
                                     )
        
        table_cache = {}

        ### Outer-most ALS loop ###
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

            for j, param_to_fit in enumerate(params_fit_order):
                # initialize gram and ordinate
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if param_to_fit == "coeff":
                    print(f"\tFitting alchemical spline coefficients ({current_time}).")
                    gram, ordinate = self.initialize_gramC_ordinateC()
                elif param_to_fit == "pseudo_weights":
                    print(f"\tFitting pseudo weights ({current_time}).")
                    gram, ordinate = self.initialize_gramW_ordinateW()
                    #gram += np.eye(gram.shape[0]) * 1e-7  # XXX: temporary
                else:
                    raise ValueError("Something went wrong.")

                # initialization for loss calculation
                sse_e = 0.0
                #sse_f = 0.0
                flat_params = self.flattened_C() if param_to_fit == "coeff" \
                   else self.flattened_W()
                yTy = 0.0

                # loop over batches
                table_iterator = parallel.progress_iter(table_names,
                                                        style=progress,
                                                        leave=False)
                for table_name in table_iterator:
                    if table_name in table_cache:
                        x_es, y_e, x_fs, y_f = table_cache[table_name]
                    else:
                        x_es, y_e, x_fs, y_f = load_preprocessed_db(preprocessed_file,
                                                                    table_name,
                                                                    load_sparse=sparse_tables)
                        if sparse_tables:
                            x_es[1] = x_es[1].toarray()
                            x_fs[1] = x_fs[1].toarray()
                        if cache_tables:
                            table_cache[table_name] = (x_es, y_e, x_fs, y_f)
                    y_e = y_e.copy()  # mutated below; keep cached originals intact
                    y_f = y_f.copy()

                    if param_to_fit == "coeff":
                        feature_matrix_e = self.feature_matrixC(x_es, self.pseudo_weights)
                        feature_matrix_f = self.feature_matrixC(x_fs, self.pseudo_weights)
                    elif param_to_fit == "pseudo_weights":
                        feature_matrix_e, yhat_e_1b = self.feature_matrixW(x_es,
                                                                           self.coeff,
                                                                           return_1b=True)
                        y_e -= yhat_e_1b
                        feature_matrix_f = self.feature_matrixW(x_fs,
                                                                self.coeff,
                                                                return_1b=False)
                    else:
                        raise ValueError("Something went wrong.")

                    # update loss, gram, and ordinate
                    sse_e += np.sum((y_e - feature_matrix_e @ flat_params)**2)
                    feature_matrix_e *= w_e
                    y_e *= w_e
                    gram += feature_matrix_e.T @ feature_matrix_e
                    ordinate += feature_matrix_e.T @ y_e
                    #sse_f += np.sum((y_f - feature_matrix_f @ flat_params)**2)
                    feature_matrix_f *= w_f
                    y_f *= w_f
                    gram += feature_matrix_f.T @ feature_matrix_f
                    ordinate += feature_matrix_f.T @ y_f
                    yTy += np.sum(y_e**2) + np.sum(y_f**2)

                # calculate RMSE and data loss
                rmse_e = np.sqrt(sse_e / n_e)
                #rmse_f = np.sqrt(sse_f / n_f)
                #data_loss = sse_e * w_e**2 + sse_f * w_f**2
                data_loss = sse_from_gram_ordinate(gram, ordinate, yTy, flat_params)
                sse_f = (data_loss - sse_e * w_e**2) / w_f**2
                rmse_f = np.sqrt(sse_f / n_f)
                rmse_e_tracker[2*i+j] = rmse_e
                rmse_f_tracker[2*i+j] = rmse_f
                data_loss_tracker[2*i+j] = data_loss
                print(f"\t\tRMSE (energy): {rmse_e}")
                print(f"\t\tRMSE (forces): {rmse_f}")
                print(f"\t\tData loss: {data_loss}")


                # add regularizers to gram
                if param_to_fit == "coeff":
                    self.update_reg_gramC(gram, C_regularizers, C_reg_free=C_reg_free)
                elif param_to_fit == "pseudo_weights":
                    self.update_reg_gramW(gram, W_sparsity_reg, W_sparsity_epsilon,
                                          C_regularizers, C_reg_free=C_reg_free)
                else:
                    raise ValueError("Something went wrong.")

                # manual total loss (for debugging)
                #reg_loss = 0
                #reg_loss += ((C_regularizers[1] @ self.coeff[1])**2).sum()
                #for k in range(2, self.degree+1):
                #    L2_norm_W = np.linalg.norm(self.pseudo_weights[k], axis=0)
                #    C_reg_part = C_regularizers[k] @ self.coeff[k]
                #    C_reg_part = C_reg_part if C_reg_free else C_reg_part * L2_norm_W
                #    reg_loss += (C_reg_part**2).sum()
                #reg_loss += np.abs(self.flattened_W()).sum() * W_sparsity_reg
                #print(f"\t\tTotal loss manual: {data_loss + reg_loss}")

                # calculate total loss (data loss + regularizer loss)
                total_loss = sse_from_gram_ordinate(gram, ordinate, yTy, flat_params)
                # add missing terms to total loss
                if param_to_fit == "coeff":
                    # missing from pseudo-weight sparsity penalty
                    if W_sparsity_reg > 0:
                        total_loss += np.abs(self.flattened_W()).sum() * W_sparsity_reg
                elif param_to_fit == "pseudo_weights":
                    # missing from 1b C_regularizers
                    if 1 in C_regularizers:
                        total_loss += ((C_regularizers[1] @ self.coeff[1])**2).sum()
                    if C_reg_free:
                        # missing from 2b and 3b C_regularizers
                        for k in range(2, self.degree+1):
                            if k in C_regularizers:
                                C_reg_part = C_regularizers[k] @ self.coeff[k]
                                total_loss += (C_reg_part**2).sum()
                else:
                    raise ValueError("Something went wrong.")
                total_loss_tracker[2*i+j] = total_loss
                print(f"\t\tTotal loss: {total_loss}")
                
                # solve
                fitted_params = solver(gram, ordinate)
                del x_es, y_e, x_fs, y_f, gram, ordinate
                gc.collect()

                ### Update ###
                # NOTE: self.coeff and self.pseudo_weights must be updated every
                # half iteration because the updated values are used in the next
                # half iteration.
                if param_to_fit == params_fit_order[0]:
                    old_coeff = self.coeff
                    old_pseudo_weights = self.pseudo_weights
                if param_to_fit == "coeff":
                    self.coeff = {1: fitted_params[:self.n_elements]}
                    for k in range(2, self.degree+1):
                        idx_lo = self.alchemical_coeff_offsets[k]
                        idx_hi = self.alchemical_coeff_offsets[k+1]
                        self.coeff[k] = fitted_params[idx_lo:idx_hi].\
                             reshape(self.n_pseudo[k], self.n_basis[k]).T
                elif param_to_fit == "pseudo_weights":
                    self.pseudo_weights = {}
                    for k in range(2, self.degree+1):
                        idx_lo = self.pseudo_offsets[k]
                        idx_hi = self.pseudo_offsets[k+1]
                        self.pseudo_weights[k] = fitted_params[idx_lo:idx_hi].\
                                reshape(self.n_ituples[k], self.n_pseudo[k])
                        # normalize weights s.t. each column proportional its L2 norm
                        normalization_factor = \
                            np.linalg.norm(self.pseudo_weights[k], axis=0) / np.sqrt(self.n_ituples[k])
                        self.pseudo_weights[k] /= normalization_factor
                        self.coeff[k] *= normalization_factor  # not necessary if coeffs are trained again

                ### Record change at the end of every whole iteration ###
                print()
                if param_to_fit == params_fit_order[-1]:
                    for k in range(2, self.degree+1):
                        max_change_pseudo = np.max(np.abs(self.pseudo_weights[k] - old_pseudo_weights[k]))
                        max_change_coeff = np.max(np.abs(
                            self.coeff[k][self.frozen_data_coverage[k]] - 
                            old_coeff[k][self.frozen_data_coverage[k]]
                            ))
                        change_coeff_tracker[k][i] = max_change_coeff
                        change_pw_tracker[k][i] = max_change_pseudo
                        print(f"\tMax change in {k}-body coefficients: {max_change_coeff:.3E}")
                        print(f"\tMax change in {k}-body pseudo weights: {max_change_pseudo:.3E}")
                    # 1b
                    max_change_coeff = np.max(np.abs(self.coeff[1][self.frozen_data_coverage[1]] -
                                                     old_coeff[1][self.frozen_data_coverage[1]]))
                    change_coeff_tracker[1][i] = max_change_coeff
                    print(f"\tMax change in 1-body coefficients: {max_change_coeff:.3E}")
                print()
            
            endtime = time.time()
            elapsed_time = endtime - starttime
            time_tracker[i+1] = elapsed_time + time_tracker[i]
            print(f"\tTime elapsed: {elapsed_time:.3F} seconds\n")
        
            ### Checkpoint ###
            if ((i+1) % checkpoint == 0) or (i+1 == max_iter):
                print("Checkpointing.")

                # Decompress and store to self.coefficients
                self.decompress_alchemical_parameters()

                os.makedirs(get_params_dir(i+1), exist_ok=True)
                self.save_alchemical_params(get_params_path(i+1))
                np.savez(tracker_filename,
                         **{f"change_pw_{k}b": change_pw_tracker[k]
                            for k in range(2, self.degree+1)},
                         **{f"change_coeff_{k}b": change_coeff_tracker[k]
                            for k in range(1, self.degree+1)},
                         time=time_tracker,
                         rmse_e=rmse_e_tracker,
                         rmse_f=rmse_f_tracker,
                         data_loss=data_loss_tracker,
                         total_loss=total_loss_tracker,
                         )

                with open(train_iter_filename, "w") as f:
                    f.write(f"{i+1}\n")
                print()




    def decompress_alchemical_parameters(self):
        """Decompress the alchemical spline coefficients and store to self.coefficients."""
        coefficients = [self.coeff[1]]
        for i in range(2, self.degree+1):
            coefficients.append( (self.coeff[i] @ self.pseudo_weights[i].T).flatten(order="F") )
        coefficients = np.concatenate(coefficients)
        coefficients = ls.revert_frozen_coefficients(coefficients,
                                                     self.n_feats,
                                                     self.mask,
                                                     self.frozen_c,
                                                     self.col_idx)
        self.coefficients = coefficients

    def flattened_C(self):
        """Return the flattened alchemical spline coefficients."""
        return np.concatenate([self.coeff[i].flatten(order="F")
                               for i in range(1, self.degree+1)])
 
    def flattened_W(self):
        """Return the flattened pseudo-weights."""
        return np.concatenate([self.pseudo_weights[i].flatten()  # order="C"
                               for i in range(2, self.degree+1)])

    def tensorizeX(self, X, k):
        """Tensorize the input array X for k-body interactions."""
        n_data, _ = np.shape(X)
        X_tensor = X.reshape(n_data, self.n_ituples[k], self.n_basis[k])
        return X_tensor

    def initialize_gramC_ordinateC(self):
        """Initialize empty gram matrices and ordinates for fitting the
        alchemical spline coefficients."""
        n_columns = self.n_elements + sum(self.n_basis[i] * self.n_pseudo[i]
                                          for i in range(2, self.degree+1))
        gram = np.zeros((n_columns, n_columns))
        ordinate = np.zeros(n_columns)
        return gram, ordinate
    
    def feature_matrixC(self, Xs, Ws):
        """
        Given the preprocessed feature dictionary Xs with feature matrices of
        shape (n_data, n_basis[i] * n_ituples[i] for valid i>1) and
        (n_elements for i==1) for each interaction order, and the pseudo_weights
        dictionary Ws with pseudo_weight arrays of shape
        (n_ituples[i], n_pseudo[i]) for each interaction order, compute the
        feature matrix WX for training the alchemical spline coefficients C.

        Args:
            Xs (Dict[int, np.ndarray]): preprocessed feature dictionary with integer
                keys (interaction order, >=1) and feature matrix values
            Ws (Dict[int, np.ndarray]): pseudo_weights dictionary with integer keys
                (interaction order, >=2) and pseudo_weight arrays

        Returns:
            WX (np.ndarray): feature matrix for training the alchemical spline
                coefficients C
        """
        WXs = [Xs[1]]
        for i in range(2, self.degree+1):
            if scipy.sparse.issparse(Xs[i]):
                K = scipy.sparse.kron(scipy.sparse.csr_matrix(Ws[i]),
                                      scipy.sparse.identity(self.n_basis[i]),
                                      format="csr")
                WX_i = np.asarray((Xs[i] @ K).todense())
            else:
                X_i_tensor = self.tensorizeX(Xs[i], i)
                WX_i = broad_row_krp_sum(Ws[i], X_i_tensor)
            WXs.append(WX_i)
        WX = np.hstack(WXs)
        return WX


    def update_reg_gramC(self, gram, C_regularizers, C_reg_free=False):
        """
        Update the gram matrix for fitting the alchemical spline coefficients C
        with alchemical regularizers.

        Args:
            gram (np.ndarray): current gram matrix
            C_regularizers (Dict): dictionary of regularization matrices for
                alchemical spline coefficients. See the class docstring for
                more information about the format.
            C_reg_free (bool): whether the c_i is regularized in the regularization
                term of the loss function (True) or c_i * ||w_i|| (False).
        """
        if C_regularizers is None:
            return

        # 1b
        if 1 in C_regularizers:
            reg = C_regularizers[1]
            gram[:self.n_elements, :self.n_elements] += reg.T @ reg
        
        # 2b, 3b, ...
        for i in range(2, self.degree+1):
            if i not in C_regularizers:
                continue
            reg = C_regularizers[i]
            reg_gram = reg.T @ reg
            L2_norm_W = np.linalg.norm(self.pseudo_weights[i], axis=0)
            for k in range(self.n_pseudo[i]):
                idx_lo = self.alchemical_coeff_offsets[i] + k * self.n_basis[i]
                idx_hi = self.alchemical_coeff_offsets[i] + (k+1) * self.n_basis[i] 
                if C_reg_free:
                    gram[idx_lo:idx_hi, idx_lo:idx_hi] += reg_gram
                else:
                    gram[idx_lo:idx_hi, idx_lo:idx_hi] += reg_gram * L2_norm_W[k]**2

    def initialize_gramW_ordinateW(self):
        """Initialize gram matrices and ordinates for fitting
        the pseudo_weights with regularizer."""
        n_columns = sum(self.n_ituples[i] * self.n_pseudo[i] for i in range(2, self.degree+1))
        gram = np.zeros((n_columns, n_columns))
        ordinate = np.zeros(n_columns)
        return gram, ordinate

    def feature_matrixW(self, Xs, Cs, return_1b=False):
        """
        Given the preprocessed feature dictionary Xs with feature matrices of
        shape (n_data, n_basis[i] * n_ituples[i] for valid i>1) and
        (n_elements for i==1) for each interaction order, and the coefficient
        dictionary Cs with coefficient arrays of shape (n_basis[i], n_pseudo[i])
        for each interaction order, compute the feature matrix XC for training
        the pseudo_weights W.

        Also computes the 1-body contribution to be subtracted off.

        Args:
            Xs (Dict[int, np.ndarray]): preprocessed feature dictionary with integer
                keys (interaction order, >=1) and feature matrix values
            Cs (Dict[int, np.ndarray]): i-body alchemical spline coefficients of
                shape (n_basis[i], n_pseudo[i]), for valid i>=1
            return_1b (bool): whether to return the 1-body contribution.

        Returns:
            XC (np.ndarray): feature matrix for training the pseudo_weights W
            Y_hat_1b (np.ndarray): 1-body contribution
        """
        n_data = np.shape(Xs[2])[0]
        XCs = []
        for i in range(2, self.degree+1):
            if scipy.sparse.issparse(Xs[i]):
                K = scipy.sparse.kron(scipy.sparse.identity(self.n_ituples[i]),
                                      scipy.sparse.csr_matrix(Cs[i]),
                                      format="csr")
                XC_i = np.asarray((Xs[i] @ K).todense())
            else:
                X_i_tensor = self.tensorizeX(Xs[i], i)
                XC_i = X_i_tensor @ Cs[i]
                XC_i = XC_i.reshape(n_data, self.n_ituples[i] * self.n_pseudo[i])
            XCs.append(XC_i)
        XC = np.hstack(XCs)
        if return_1b:
            Y_hat_1b = Xs[1] @ Cs[1]
            return XC, Y_hat_1b
        return XC

    def update_reg_gramW(self, gram, W_sparsity_reg, W_sparsity_epsilon,
                         C_regularizers, C_reg_free=False):
        """
        Update the gram matrix for fitting the pseudo_weights W with alchemical
        regularizers.

        Args:
            gram (np.ndarray): current gram matrix
            W_sparsity_reg (float): regularization strength for sparsity of
                pseudo-weights (L1 penalty).
            W_sparsity_epsilon (float): small value for L1 penalty to avoid
                division by zero.
            C_regularizers (Dict): dictionary of regularization matrices for
                alchemical spline coefficients. See the class docstring for
                more information about the format.
            C_reg_free (bool): whether the c_i is regularized in the regularization
                term of the loss function (True) or c_i * ||w_i|| (False).
        """
        # pseudo-weight sparsity penalty
        if W_sparsity_reg > 0:
            pseudo_weights_flat = self.flattened_W()
            gram += sparsity_reg_matrix(pseudo_weights_flat,
                                        strength=W_sparsity_reg,
                                        epsilon=W_sparsity_epsilon)

        # contribution from C_regularizers if C_reg_free is False
        if C_reg_free:
            return
        if C_regularizers is None:
            return
        for i in range(2, self.degree+1):
            if i not in C_regularizers:
                continue
            DC_nonzero_sq = np.sum((C_regularizers[i] @ self.coeff[i])**2, axis=0)
            for k in range(self.n_ituples[i]):
                idx_lo = self.pseudo_offsets[i] + k * self.n_pseudo[i]
                idx_hi = self.pseudo_offsets[i] + (k+1) * self.n_pseudo[i]
                indices = np.arange(idx_lo, idx_hi)
                gram[indices, indices] += DC_nonzero_sq


class AlchemicalModelTorch(AlchemicalModel,
                           torch.nn.Module if TORCH_AVAILABLE else object):
    """
    Alchemical learning ("pseudo-interaction") model for fitting energies and
    forces using PyTorch. See AlchemicalModel for more information.
    """
    def __init__(self,
                 bspline_config: bspline.BSplineBasis,
                 n_pseudo: Union[int, Dict[int, int]],
                 data_coverage: np.ndarray = None,
                 dtype=torch.float64,
                 **args):
        AlchemicalModel.__init__(self,
                                 bspline_config,
                                 n_pseudo,
                                 data_coverage,
                                 **args)
        torch.nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.dtype = dtype

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
        self.coeff = {key: torch.tensor(val, device=self.device,
                                        dtype=self.dtype, requires_grad=True)
                        for key, val in self.coeff.items()
                        }
        self.pseudo_weights = {key: torch.tensor(val, device=self.device,
                                                 dtype=self.dtype, requires_grad=True)
                                for key, val in self.pseudo_weights.items()
                                }



    def convert2numpy(self):
        """Convert torch tensor attributes to numpy arrays."""
        self.coeff = {key: val.detach().cpu().numpy()
                        for key, val in self.coeff.items()
                        }
        self.pseudo_weights = {key: val.detach().cpu().numpy()
                                for key, val in self.pseudo_weights.items()
                                }

    def parameters(self):
        return list(self.coeff.values()) + list(self.pseudo_weights.values())

    def decompress_alchemical_parameters(self):
        """Ensures that convert2numpy() is called before calling superclass method.
           Then restore by calling convert2torch()."""
        self.convert2numpy()
        super().decompress_alchemical_parameters()
        self.convert2torch()

    def forward(self, xs):
        """
        Forward pass for the alchemical model.

        Args:
            xs (Dict[torch.Tensor]): input tensor dictionary with integer
                keys (interaction order, >=1) and feature matrix values

        Returns:
            y (torch.Tensor): output tensor of shape (n_data,)
        """
        y = xs[1] @ self.coeff[1]
        for i in range(2, self.degree+1):
            y += (torch.matmul(xs[i], self.coeff[i]) * self.pseudo_weights[i]).sum(dim=(1,2))
        return y
    
    def normalize_parameters(self):
        """Normalize the alchemical model parameters."""
        # XXX: Don't use yet.
        for i in range(2, self.degree+1):
            normalization_factor = torch.norm(self.pseudo_weights[i], dim=0)
            self.pseudo_weights[i] /= normalization_factor
            self.coeff[i] *= normalization_factor

    def regularization_loss(self,
                            C_regularizers: Dict = None,
                            W_sparsity_reg: float = 0.0,
                            C_reg_free: bool = False,):
        """
        Compute the regularization loss for the alchemical model.

        Args:
            C_regularizers (Dict): dictionary of regularization matrices for
                alchemical spline coefficients. See the class docstring for
                more information about the format.
            W_sparsity_reg (float): regularization strength for sparsity of
                pseudo-weights (L1 penalty). Defaults to 0.0.
            C_reg_free (bool): whether the c_i is regularized in the regularization
                term of the loss function (True) or c_i * ||w_i|| (False). Defaults
                to False.

        Returns:
            loss (torch.Tensor): regularization loss
        """
        if C_regularizers is None:
            return 0.0
        alchemical_coeffs = [self.coeff[1]]
        for i in range(2, self.degree+1):
            alchemical_coeffs.append(self.coeff[i].T.contiguous().view(-1))
        alchemical_coeffs = torch.cat(alchemical_coeffs)
        loss = torch.matmul(self.regularizer, alchemical_coeffs)
        loss = torch.sum(loss**2)

        loss = 0
        # 1b for alchemical spline coefficients
        if 1 in C_regularizers:
            loss += ((torch.tensor(C_regularizers[1]) @ self.coeff[1])**2).sum()
        # 2b, 3b, ... for alchemical spline coefficients
        for i in range(2, self.degree+1):
            if i in C_regularizers:
                C_reg_part = torch.tensor(C_regularizers[i]) @ self.coeff[i]
                if not C_reg_free:
                    L2_norm_W = torch.norm(self.pseudo_weights[i], dim=0)
                    C_reg_part = C_reg_part * L2_norm_W
                loss += (C_reg_part**2).sum()
        # sparse regularization for pseudo-weights
        loss += W_sparsity_reg * torch.abs(self.flattened_W()).sum()
        return loss

    def train_from_file(self,
                        preprocessed_file: str = "preprocessed.h5",
                        coverage_file: str = "coverage.npz",
                        metadata_file: str = "metadata.npz",
                        init_params: Union[dict, np.lib.npyio.NpzFile] = None, 
                        C_regularizers: Dict = None,
                        C_reg_free: bool = False,
                        W_sparsity_reg: float = 0.0,
                        progress: str = "bar",
                        batch_size: int = 1,
                        max_epochs: int = 1,
                        optimizer_class: torch.optim.Optimizer = None,
                        optimizer_kwargs: dict = {},
                        checkpoint: int = 10,
                        checkpoint_dir: str = ".",
                        shuffle: bool = True,
                        drop_last: bool = False,
                        dataloader_n_workers: int = 0,
                        params_filename: str = "alchemical_model_params.npz",
                        tracker_filename: str = "train_tracker.npz",
                        train_iter_filename: str = ".train_iter",
                        resume: bool = False,
                        ):
        """
        Accumulate inputs and outputs from batched parsing of HDF5 file
        and train the Alchemical model parameters using a gradient-based
        optimizer.

        Args:
            preprocessed_file (str): path to preprocessed HDF5 data file created
                by `preprocess_for_training()`.
            coverage_file (str): path to data coverages file created by
                `preprocess_for_training()`.
            metadata_file (str): path to metadata file created by
                `preprocess_for_training()`.
            init_params (dict | np.lib.npyio.NpzFile ): initial parameters for training.
                Should be an object with keys 'coeff_1b', 'coeff_2b', 'pseudo_weights_2b'
                (2b) and 'coeff_3b', 'pseudo_weights_3b' (3b).
            C_regularizers (Dict): dictionary of regularization matrices for
                alchemical spline coefficients. See the class docstring for
                more information about the format.
            C_reg_free (bool): whether the c_i is regularized in the regularization
                term of the loss function (True) or c_i * ||w_i|| (False). Defaults
                to False. Must be False for PyTorch training.
            W_sparse_reg (float): regularization strength for sparsity of
                pseudo-weights (L1 penalty). Defaults to 0.0.
            progress (str): style for progress indicators.
            batch_size (int): batch size for training.
            max_epochs (int): maximum number of iterations for alternating
                least-squares optimization.
            optimizer_class (torch.optim.Optimizer): optimizer class for training.
                If None, defaults to Adam.
            optimizer_kwargs (dict): keyword arguments for the optimizer.
            checkpoint (int): frequency of checkpointing the training RMSE and
                saving parameters to disk.
            checkpoint_dir (str): directory for saving checkpoints.
            shuffle (bool): shuffle the dataset during training.
            drop_last (bool): drop the last incomplete batch during training.
            dataloader_n_workers (int): number of workers for PyTorch DataLoader.
            params_filename (str): filename for saving parameters during
                checkpoints.
            train_iter_filename (str): filename for writing current iteration
                number during checkpoints.
            resume (bool): whether to resume training from a previous checkpoint.
        """
        if C_reg_free:
            raise ValueError("C_reg_free must be False for PyTorch training\n"
                             "because normalization at each epoch is not properly\n"
                             "implemented yet.")

        init_epoch, get_params_dir, get_params_path, tracker_filename, \
        train_iter_filename, change_pw_tracker, \
        change_coeff_tracker, time_tracker, rmse_e_tracker, rmse_f_tracker, \
        data_loss_tracker, total_loss_tracker, params_fit_order, table_names, \
        w_e, w_f, n_e, n_f = \
            self.initialize_training(preprocessed_file=preprocessed_file,
                                     coverage_file=coverage_file,
                                     metadata_file=metadata_file,
                                     init_params=init_params,
                                     max_iter=max_epochs,
                                     checkpoint=checkpoint,
                                     checkpoint_dir=checkpoint_dir,
                                     params_filename=params_filename,
                                     tracker_filename=tracker_filename,
                                     train_iter_filename=train_iter_filename,
                                     resume=resume,
                                     fit_first="C",  # doesn't matter for PyTorch
                                     )
        self.convert2torch()  # converts self.coeff and self.pseudo_weights to torch tensors

        # Optimizer
        if optimizer_class is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        else:
            optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)

        # Create PyTorch DataLoader
        assert batch_size == 1  # XXX: for now
        dataloader = torch_util.hdf5_dataloader(preprocessed_file,
                                                table_names,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                drop_last=drop_last,
                                                num_workers=dataloader_n_workers,
                                                load_fn=load_preprocessed_db,
                                                load_sparse=False,
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
        for i in range(init_epoch, max_epochs):
            print(f"Iteration {i+1}/{max_epochs}")
            starttime = time.time()

            optimizer.zero_grad()
            loss_e = 0.0
            loss_f = 0.0
            #self.normalize_parameters()
            old_pseudo_weights = {k: self.pseudo_weights[k].detach().clone()
                                    for k in range(2, self.degree+1)}
            old_coeff = {k: self.coeff[k].detach().clone()
                            for k in range(1, self.degree+1)}

            table_iterator = parallel.progress_iter(dataloader,
                                                    style=progress,
                                                    total=n_batches,
                                                    leave=False)
            for batch_list in table_iterator:
                batch = batch_list[0]  # we contrained batch_size=1 above
                x_es, y_e, x_fs, y_f = batch

                # Accumulate losses
                for key, val in x_es.items():
                    x_i_tensor = self.tensorizeX(val, int(key)) if int(key) > 1 else val
                    x_es[key] = torch.tensor(x_i_tensor, device=self.device, dtype=self.dtype,
                                             requires_grad=False)
                y_e = torch.tensor(y_e, device=self.device, dtype=self.dtype,
                                   requires_grad=False)
                p_e = self(x_es)
                loss_e += torch.nn.functional.mse_loss(p_e, y_e, reduction="sum")
                for key, val in x_fs.items():
                    x_i_tensor = self.tensorizeX(val, int(key)) if int(key) > 1 else val
                    x_fs[key] = torch.tensor(x_i_tensor, device=self.device, dtype=self.dtype,
                                             requires_grad=False)
                y_f = torch.tensor(y_f, device=self.device, dtype=self.dtype,
                                   requires_grad=False)
                p_f = self(x_fs)
                loss_f += torch.nn.functional.mse_loss(p_f, y_f, reduction="sum")
                
            # Compute total loss
            loss = loss_e * w_e**2 + loss_f * w_f**2
            data_loss_tracker[2*i: 2*i+1] = loss.item()
            loss += self.regularization_loss(C_regularizers, W_sparsity_reg,
                                             C_reg_free=C_reg_free)
            total_loss_tracker[2*i: 2*i+1] = loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Record progress
            for k in range(2, self.degree+1):
                max_change_pseudo = torch.max(torch.abs(self.pseudo_weights[k] - old_pseudo_weights[k])).item()
                max_change_coeff = torch.max(torch.abs(self.coeff[k] - old_coeff[k])).item()
                change_coeff_tracker[k][i] = max_change_coeff
                change_pw_tracker[k][i] = max_change_pseudo
                print(f"\tMax change in {k}-body coefficients: {max_change_coeff:.3E}")
                print(f"\tMax change in {k}-body pseudo weights: {max_change_pseudo:.3E}")
            # 1b
            max_change_coeff = torch.max(torch.abs(self.coeff[1] - old_coeff[1])).item()
            change_coeff_tracker[1][i] = max_change_coeff
            print(f"\tMax change in 1-body coefficients: {max_change_coeff:.5E}")
            rmse_e = torch.sqrt(loss_e.data / n_e)
            rmse_f = torch.sqrt(loss_f.data / n_f)
            rmse_e_tracker[2*i: 2*i+1] = rmse_e.item()
            rmse_f_tracker[2*i: 2*i+1] = rmse_f.item()
            print(f"\tRMSE energy (eV/atom): {rmse_e:.5E}")
            print(f"\tRMSE force (eV/A): {rmse_f:.5E}")
            
            endtime = time.time()
            elapsed_time = endtime - starttime
            time_tracker[i+1] = elapsed_time + time_tracker[i]
            print(f"\tTime elapsed (s): {elapsed_time:.3F}\n")

            # Checkpoint
            if ((i+1) % checkpoint == 0) or (i+1 == max_epochs):
                print("Checkpointing.")

                # Decompress and store to self.coefficients (as NumPy arrays)
                self.decompress_alchemical_parameters()

                os.makedirs(get_params_dir(i+1), exist_ok=True)
                #torch.save(self.state_dict(), get_params_path(i+1))
                self.save_alchemical_params(get_params_path(i+1))
                np.savez(tracker_filename,
                         **{f"change_pw_{k}b": change_pw_tracker[k]
                            for k in range(2, self.degree+1)},
                         **{f"change_coeff_{k}b": change_coeff_tracker[k]
                            for k in range(1, self.degree+1)},
                         time=time_tracker,
                         rmse_e=rmse_e_tracker,
                         rmse_f=rmse_f_tracker,
                         data_loss=data_loss_tracker,
                         total_loss=total_loss_tracker,
                         )

                with open(train_iter_filename, "w") as f:
                    f.write(f"{i+1}\n")
                print()

    def fit_from_file(self, *args, **kwargs):
        warnings.warn("Calling fit_from_file() of AlchemcalModelTorch.\n"
                      "Redirecting to train_from_file().")
        return self.train_from_file(*args, **kwargs)


def save_preprocessed_db(feature_dict_e: dict,
                         y_e: np.ndarray,
                         feature_dict_f: dict,
                         y_f: np.ndarray,
                         filename: str,
                         degree: int,
                         batch_name: str = 'batch',
                         ):
    """
    Save preprocessed database to HDF5 file in a sparse CSR matrix format.
    Used during alchemical model fitting.

    Args:
        feature_dict_e (dict): dictionary of energy feature matrices for each
            interaction order (1, 2, ..., degree).
        y_e (np.ndarray): vector of target energies.
        feature_dict_f (dict): dictionary of force feature matrices for each
            interaction order (2, 3, ..., degree).
        y_f (np.ndarray): vector of target forces.
        filename (str): path to HDF5 file.
        degree (int): maximum interaction order.
        batch_name (str): name of batch group in HDF5 file.
    """
    # sanity checks
    expected_keys = range(1, degree + 1)
    assert set(feature_dict_e.keys()).issubset(expected_keys)
    assert set(feature_dict_f.keys()).issubset(expected_keys)

    # save to HDF5
    with tables.open_file(filename, mode='a') as f:
        if ('/' + batch_name) in f:
            warnings.warn(f"Table {batch_name} already exists in {filename}. Skipping...")
        else:
            batch_group = f.create_group("/", batch_name, batch_name)

            for dtype, feature_dict, y in zip(['energy', 'force'],
                                              [feature_dict_e, feature_dict_f],
                                              [y_e, y_f]):
                group = f.create_group(batch_group, dtype, dtype)
                for key, arr in feature_dict.items():
                    subgroup_name = f"_{key}"
                    subgroup = f.create_group(group, subgroup_name, subgroup_name)
                    csr_arr = scipy.sparse.csr_array(arr)
                    process.cs_array_to_h5_group(csr_arr, f, subgroup)
                f.create_array(group, 'y', y)


def load_preprocessed_db(filename: str,
                         batch_name: str = 'batch',
                         load_sparse: bool = False,
                         ):
    """
    Load preprocessed database from HDF5 file in an array.
    Used during alchemical model fitting.

    Args:
        filename (str): path to HDF5 file.
        batch_name (str): name of batch group in HDF5 file.
        load_sparse (bool): whether to load as sparse matrix (CSR format).

    Returns:
        feature_dict_e (dict): dictionary of energy feature matrices for each
            interaction order (1, 2, ..., degree).
        y_e (np.ndarray): vector of target energies.
        feature_dict_f (dict): dictionary of force feature matrices for each
            interaction order (2, 3, ..., degree).
        y_f (np.ndarray): vector of target forces.
    """
    feature_dict_e = {}
    feature_dict_f = {}
    ys = []
    with tables.open_file(filename, mode='r') as f:
        batch_group = f.get_node("/" + batch_name)

        for dtype, feature_dict in zip(['energy', 'force'],
                                       [feature_dict_e, feature_dict_f]):
            group = f.get_node(batch_group, dtype)
            group_names = [node._v_name for node in group._f_list_nodes() if node._v_name != 'y']
            for subgroup_name in group_names:
                subgroup = f.get_node(group, subgroup_name)
                arr = process.cs_array_from_h5_group(f, subgroup, cstype='csr')
                if not load_sparse:
                    arr = arr.toarray()
                key = int(subgroup_name[1:])
                feature_dict[int(key)] = arr
            y = f.get_node(group, 'y')[:]
            ys.append(y)

    return feature_dict_e, ys[0], feature_dict_f, ys[1]








def legacy_broad_row_krp_sum(A, B):
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


@jit(nopython=True, nogil=True)
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
    r, m, s = B.shape
    n = A.shape[1]  # will assume that A.shape[0] == m
    result = np.zeros((r, n * s))
    for i in range(r):
        tmp = np.zeros(n * s)
        for j in range(m):
            tmp += np.kron(A[j], B[i, j])
        result[i] = tmp
    return result


def get_C_regularizers(bspline_config: bspline.BSplineBasis,
                       ridge_map={},
                       curvature_map={},
                       **kwargs):
    """
    Generate regularization matrices for the alchemical spline coefficients C.
    For each interaction order (>1), the regularization matrix of only a single pseudo
    interaction is generated, which will be applied to all pseudo interactions
    during fitting. For 1-body terms, the matrix is the same as the typical 1-body
    regularization matrix (number of columns == number of elements).

    Args:
        bspline_config (bspline.BSplineBasis): bspline configuration object.
        ridge_map (dict): dictionary of ridge regularization strengths for each
            interaction order. Defaults to {}.
        curvature_map (dict): dictionary of curvature regularization strengths
            for each interaction order. Defaults to {}.
        **kwargs: additional keyword arguments for ridge_map and curvature_map.

    Returns:
        C_reg_dict (dict): dictionary of regularization matrices for the alchemical
            spline coefficients. The keys are the interaction orders and the values
            are the regularization matrices of shape (_, n_basis).
    """
    # determine AlchemicalModel structure
    n_elements = len(bspline_config.element_list)
    lead_trim = bspline_config.leading_trim
    trail_trim = bspline_config.trailing_trim

    component_sizes = bspline_config.get_interaction_partitions()[0]
    n_basis = {i: component_sizes[bspline_config.interactions_map[i][0]]
                for i in range(2, bspline_config.degree+1)}
    n_basis[2] = n_basis[2] - bspline_config.leading_trim - \
            bspline_config.trailing_trim
    # temporary sanity checks
    for pair in bspline_config.interactions_map[2]:
        if not component_sizes[pair] == n_basis[2] + lead_trim + trail_trim:
            raise ValueError("Inconsistent component sizes.")
    for i in range(3, bspline_config.degree+1):
        for ituple in bspline_config.interactions_map[i]:
            if not component_sizes[ituple] == n_basis[i]:
                raise ValueError("Inconsistent component sizes.")
    assert bspline_config.offset_1b  # fit 1-body

    for k in kwargs:
        if k.lower()[0] == 'r':
            ridge_map[int(re.sub('[^0-9]', '', k))] = float(kwargs[k])
        elif k.lower()[0] == 'c':
            curvature_map[int(re.sub('[^0-9]', '', k))] = float(kwargs[k])

    ridge_map = {1: regularize.DEFAULT_REGULARIZER_GRID["ridge_1b"],
                    2: regularize.DEFAULT_REGULARIZER_GRID["ridge_2b"],
                    3: regularize.DEFAULT_REGULARIZER_GRID["ridge_3b"],
                    **ridge_map}
    curvature_map = {1: 0.0,
                        2: regularize.DEFAULT_REGULARIZER_GRID["curve_2b"],
                        3: regularize.DEFAULT_REGULARIZER_GRID["curve_3b"],
                        **curvature_map}
    # one-body element terms
    matrix = bspline_config.get_regularization_matrix_1b(n_elements, ridge=ridge_map[1])
    C_reg_dict = {1: matrix}
    # two- and three-body terms
    for degree in range(2, bspline_config.degree + 1):
        r = ridge_map[degree]
        c = curvature_map[degree]
        if degree == 2:
            #matrix = bspline_config.get_regularization_matrix_2b(interaction,
            #                                            ridge=r,
            #                                            curvature=c)
            matrix = regularize.get_ridge_penalty_matrix(n_basis[degree] + lead_trim + trail_trim)
            matrix *= np.sqrt(r)
            if c > 0:
                matrix_c = regularize.get_curvature_penalty_matrix_1D(n_basis[degree] + lead_trim + trail_trim)
                matrix_c *= np.sqrt(c)
                matrix = np.vstack((matrix, matrix_c))
            mask = np.arange(lead_trim, lead_trim + n_basis[degree])
            matrix = ls.freeze_regularizer(matrix, mask)
        elif degree == 3:
            #matrix = bspline_config.get_regularization_matrix_3b(interaction,
            #                                               ridge=r,
            #                                               curvature=c)
            dummy_interaction = bspline_config.interactions_map[degree][0]  # random choice
            mask = bspline_config.template_mask[dummy_interaction]  # template must be the same for all

            # Ridge regularization for compressed coefficients
            matrix = regularize.get_ridge_penalty_matrix(len(mask))
            matrix *= np.sqrt(r)

            # Curvature regularization
            if c > 0:
                size = bspline_config.resolution_map[dummy_interaction]
                matrix_c = regularize.get_curvature_penalty_matrix_3D(
                    size[0] + 3,
                    size[1] + 3,
                    size[2] + 3,
                    flatten=False,
                    )
                # compress each row of matrix_c
                matrix_c_compressed = np.zeros((len(mask), len(mask)))
                for compressed_i, uncompressed_i in enumerate(mask):
                    row = matrix_c[uncompressed_i]
                    matrix_c_compressed[compressed_i] = \
                        bspline_config.compress_3B(row, dummy_interaction)
                matrix_c_compressed *= np.sqrt(c)

                matrix = np.vstack((matrix, matrix_c_compressed))
        else:
            raise ValueError(
                "Four-body terms and beyond are not yet implemented.")
        C_reg_dict[degree] = matrix
    return C_reg_dict


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
    params = np.where(np.abs(params) < epsilon, epsilon, params)
    reg_matrix = np.diag(1/params) * strength
    return reg_matrix


def sse_from_gram_ordinate(gram, ordinate, yTy, beta):
    """
    Compute the sum of squared errors when gram and ordiante matrices have
    already been computed.

    SSE = (y - X @ beta)^T @ (y - X @ beta)
        = y.T @ y - 2 * y.T @ X @ beta + beta.T @ X.T @ X @ beta
        = y.T @ y - 2 * ordinate @ beta + beta.T @ gram @ beta

    Args:
        gram (np.ndarray): gram matrix of shape (n x n).
        ordinate (np.ndarray): ordinate vector of shape (n,).
        yTy (float): y.T @ y.
        beta (np.ndarray): parameter vector of shape (n,)
    """
    return yTy - 2 * ordinate @ beta + beta @ gram @ beta


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
