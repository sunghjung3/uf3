from ase.io import read
from ase.io import trajectory
from ase.atoms import Atoms
import ase.data as ase_data

from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
try:
    from ase.calculators.vasp import Vasp
    from pymatgen.io.vasp.outputs import Vasprun
except ImportError:
    Warning.warn("Trouble importing Vasp-related packages. Vasp cannot be used.")

#from ase.optimize.sciopt import SciPyFminCG
from ase.optimize import FIRE
#from myopts import GD, MGD

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import spatial

import uf3_run
from uf3.regression import least_squares
from uf3.util import plotting, user_config
from uf3.forcefield import calculator
from uf3.data import io
from uf3.representation import process
from uf3.data import geometry, composition
import r_uq
from preprocess import preprocess

import copy, sys, time, os, glob, pickle, gc, concurrent

#from memory_profiler import profile


def generate_sample_weights(current_image_prefix, strength, nimages_start=1):
    """
    Generate sample weights for training data.

    Args:
        current_image_prefix (int): Current image prefix.
        strength (int): How strongly to weigh the most recent sample vs the oldest.
            The most recent datapoint will be weighted by 2**(strength).
            Each previous datapoint will be weighted half as much until a weight of 1.0 is reached.
            All datapoints before that will be weighted 1.0.

    Returns:
        sample_weights (dict): Sample weights for training data.
    """
    sample_weights = dict()
    for i in range(current_image_prefix, -2, -1):
        key = str(i) + "_" + '0'  # the '0' is assuming that UF3 names each configuration by its index in the trajetory and that we only have one new configuration per forcecall
        sample_weights[key] = 2 ** max(strength - (current_image_prefix - i), 0)

    for i in range(1, nimages_start):
        key = '-1_' + str(i)
        sample_weights[key] = sample_weights["-1_0"]

    return sample_weights


class Pseudodict(dict):
    """
    An object that appears like a dictionary but returns one value for any
    existing and nonexistent keys.
    """
    def __init__(self, value):
        self.value = value
    def __getitem__(self, key):
        return self.value
    def get(self, key, default=None):
        return self.value

def strip_calc(atoms, e_val, f_val, inplace=False):
    """
    Given an atoms object and energy and force values, strip the calculator
    and replace it with a single point calculator with the same energy and force
    values.
    """
    if inplace:
        ret_atoms = atoms
    else:
        ret_atoms = copy.deepcopy(atoms)
    ret_atoms.calc = SinglePointCalculator(ret_atoms, energy=e_val, forces=f_val)
    return ret_atoms

def check_vasp_convergence(vasprun_file="vasprun.xml") -> bool:
    if Vasprun(vasprun_file).converged_electronic:
        return True
    raise ValueError("VASP calculation did not converge. Terminating UFMin.")

def initial_data_prep(structure_input,
                      nimages_start,
                      bspline_config,
                      true_calc,
                      true_calc_type,
                      conv_fmax,
                      ):
    '''
    Read in initial data/structure and calculate energy and forces (if necessary).
    '''
    if isinstance(structure_input, str):
        # structure file or trajectory file
        try:
            structure_input = trajectory.Trajectory(structure_input, mode='r')
        except:
            atoms = read(structure_input)
            structure_input = [atoms]
    else:
        # ase.Atoms object or List[ase.Atoms] or trajectory.Trajectory
        if isinstance(structure_input, Atoms):
            structure_input = [copy.deepcopy(structure_input)]
        elif isinstance(structure_input, trajectory.TrajectoryReader):
            structure_input = copy.deepcopy(structure_input)
        else:
            assert isinstance(structure_input, list)
            tmp = list()
            for atoms in structure_input:
                assert isinstance(atoms, Atoms)
                tmp.append(copy.deepcopy(atoms))
            structure_input = tmp

    # preprocess all image in traj
    pair_tuples = bspline_config.interactions_map[2]
    traj = list()
    for i, atoms in enumerate(structure_input):
        print("Preprocessing image", i)
        preprocessed_atoms = preprocess(atoms, pair_tuples, strength=preprocess_strength)
        traj.append(preprocessed_atoms)

    # calculate energy and forces for all images in traj
    for i, atoms in enumerate(traj):
        try:
            most_recent_E_eval = atoms.get_potential_energy()
            most_recent_F_eval = atoms.get_forces()
            print("Using provided energy and forces for image", i)
        except RuntimeError:
            atoms.calc = true_calc
            most_recent_E_eval = atoms.get_potential_energy()
            most_recent_F_eval = atoms.get_forces()
            if true_calc_type == "vasp":
                check_vasp_convergence()
            print("Calculated energy and forces for image", i)
        strip_calc(atoms, most_recent_E_eval, most_recent_F_eval, inplace=True)

    # generate additional images if necessary
    n = len(traj)
    for i in range(n, nimages_start):
        print("Generating image", i)
        atoms = copy.deepcopy(traj[-1])
        atoms.calc = true_calc
        dyn = optimizer(atoms)
        dyn.run(steps=1, fmax=conv_fmax)
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
        if true_calc_type == "vasp":
            check_vasp_convergence()
        strip_calc(atoms, most_recent_E_eval, most_recent_F_eval, inplace=True)
        traj.append(atoms)

    return traj


#@profile
def ufmin(structure_input = "POSCAR",
          nimages_start = 1,  # number of images to start with
          initial_data_output_file = "initial_data.traj",
          live_features_file = "live_features.h5",
          model_file_prefix = "model",  # store model from each step
          settings_file = "settings.yaml",
          opt_traj_file = "ufmin.traj",  # array of images at all real force evaluations
          model_traj_file = "ufmin_model.traj",  # array of images at each UF3 minimization
          model_calc_file = "model_calc.pckl",  # store energy and forces from UF3 calls
          status_update_file = "ufmin_status.out",
          ufmin_true_fmax = 0.05,  # force tolerance for the actual optimization
          ufmin_uf3_fmax = 0.05,  # force tolerance for the optimization on the uf3 surface
          optimizer = FIRE,
          max_forcecalls = 200,
          max_uf3_calls = 1000,
          verbose = 0,
          true_calc = None,
          true_calc_type = None,
          resolution_map = None,
          learning_weight = 0.5,
          regularization_values = None,
          uq_tolerance = 1.0,
          preprocess_strength = 0.5,
          sample_weight_strength = 6,
          pretrained_models = None,
          normalize_forces = False,
          true_forcecall_always = True,
          resume = 0  # start from scratch if 0. Do `resume` additional forcecalls if non-zero. If non-zero, overrides `max_forcecalls` parameter.
          ):
    """
    Energy minimization using UF3 surrogate model.

    Intended trajectory and iteration numbering system:
        * `initial_data_output_file` file: trajectory of initial data, including the starting point of optimization
            * These are data points that are used to train the first surrogate UF3 model (model 0)
            * These will have the image prefix of -1.
        * `opt_traj_file` file: trajectory of all images at all real force evaluations, excluding the starting point of optimization and initial data.
            * With the zero-indexing system, the i-th image is the resulting structure from minimizing upon model i
        * `traj` variable: union of images in `initial_data_output_file` and `opt_traj_file`

    Args:
        structure_input (str | ase.Atoms | List[ase.Atoms] | ase.io.trajectory.Trajectory): Initial dataset to start optimization from. Will be ignored if `resume` is non-zero.
            If str, then it is a path to a structure file or a trajectory file.
                If a structure file, then it is the initial structure. Optimization will start here.
                If a trajectory file, then it is a trajectory object. Optimization will start from the last structure in the trajectory.
            If ase.Atoms, then it is the structure itself. Optimization will start here.
            If List[ase.Atoms], then it is a list of structures. Optimization will start from the last structure in the list.
                If energy and forces are provided, then they will be used. Otherwise, they will be calculated.
            If ase.io.trajectory.Trajectory, then it is a trajectory object. Optimization will start from the last structure in the trajectory.
                If energy and forces are provided, then they will be used. Otherwise, they will be calculated.
        nimages_start (int): Minimum number of images to start with. Used only if `resume` is zero.
            If `structure_input` contains less than `nimages_start` images, then the remaining images will be generated by performing a single optimization step on the true energy surface.
            If `structure_input` contains more than `nimages_start` images, then all provided images will be used.
        initial_data_output_file (str): Path to ASE trajectory file to store images used to train the first surrogate UF3 model.
        live_features_file (str): Path to HDF5 file to store live features during optimization.
        model_file_prefix (str): Prefix for model files.
            Each model will be saved as `model_file_prefix`_`forcecall_counter`.json
        settings_file (str): Path to settings file.
        opt_traj_file (str): Path to ASE trajectory file to store optimization trajectory at all real force evaluations.
        model_traj_file (str): Path to ASE trajectory file to store optimization trajectory at each UF3 minimization for all real force evaluations.
        model_calc_file (str): Path to pickle file to store energy and forces from UF3 calls (real and MLFF).
        status_update_file (str): Path to file to store status updates.
        ufmin_true_fmax (float): Force tolerance for the actual optimization.
        ufmin_uf3_fmax (float): Force tolerance for the optimization on the uf3 surface.
        optimizer (ase.optimize.Optimizer): ASE optimizer to use for UF3 minimization.
        max_forcecalls (int): Maximum number of force calls to make on the true energy surface.
        max_uf3_calls (int): Maximum number of force calls to make on the UF3 surface.
        verbose (int): Verbosity level.
        true_calc (ase.calculator.Calculator): ASE calculator to use for true energy and force evaluations.
        true_calc_type (str): Type of true calculator.
            e.g. "vasp" for VASP
        resolution_map (dict): Resolution map for features.
        learning_weight (float): Weight for learning term in loss function.
        regularization_values (list): ridge and curvature regularization values.
        uq_tolerance (float): How tolerable this workflow is to uncertainty.
            See `r_uq.R_UQ` class for more details.
        preprocess_strength (float): Strength of preprocessing (0.0 to 1.0)
        sample_weight_strength (int): how strongly to weigh the most recent sample vs the oldest.
            See `generate_sample_weights` function for more details.
        pretrained_models (dict): List of pretrained models to use for training. 
            forcecall number -> model file path
        normalize_forces (bool): Whether or not to normalize features by max true force value.
        true_forcecall_always (bool): If true, evaluates the true calculator even during sub-optimization on the UF3 surface.
            For analysis purposes.
        resume (int): Start from scratch if 0. If non-zero, then `structure_input` is ignored.
    """

    ufmin_true_fmax_squared = ufmin_true_fmax ** 2
    ufmin_uf3_fmax_squared = ufmin_uf3_fmax ** 2

    if resume:
        # load existing files to resume process
        print(f"***RESUMING***\n`structure_input` = {structure_input} will be ignored.")

        try:
            init_traj = trajectory.Trajectory(initial_data_output_file, mode='r')
            nimages_start = len(init_traj)
            traj = [image for image in init_traj]
            init_traj.close()
        except FileNotFoundError:
            sys.exit(f"Initial data file {initial_data_output_file} does not exist. Cannot resume.")

        try:
            opt_traj = trajectory.Trajectory(opt_traj_file, mode='r')
            for image in opt_traj:
                traj.append(image)
            opt_traj.close()
            opt_traj = trajectory.Trajectory(opt_traj_file, mode='a')  # open in append mode to add on
        except FileNotFoundError:
            sys.exit(f"Optimization trajectory file {opt_traj_file} does not exist. Cannot resume.")
        
        if not os.path.isfile(model_traj_file):
            sys.exit(f"Model trajectory file {model_traj_file} does not exist. Cannot resume.")
        model_traj_file = open(model_traj_file, 'ab')

        if not (os.path.isfile(model_calc_file)) :
            sys.exit(f"Calculation pickle files do not exist. Cannot resume.")
        model_calc_file = open(model_calc_file, 'ab')

        if not os.path.isfile(live_features_file):
            sys.exit(f"Live features file {live_features_file} does not exist. Cannot resume.")


    else:  # start from scratch
        if os.path.isfile(live_features_file):
            sys.exit("Remove the live features file before running this script.")
        if model_file_prefix is not None:
            if glob.glob(model_file_prefix + "_*.json"):
                sys.exit("Remove model files before running this script.")
        if os.path.isfile(opt_traj_file) or os.path.isfile(model_traj_file):
            sys.exit("Remove the minimization traj files before running this script.")
        if os.path.isfile(model_calc_file):
            sys.exit("Remove the calculation pickle files before running this script.")
        if os.path.isfile(status_update_file):
            os.remove(status_update_file)
        opt_traj = trajectory.Trajectory(opt_traj_file, mode='w')  # to save optimization traj
        model_traj_file = open(model_traj_file, 'wb')
        model_calc_file = open(model_calc_file, 'wb')


    ### Set up ###

    # one-time bspline configuration
    bspline_config = uf3_run.initialize(settings_file, verbose=verbose, resolution_map=resolution_map)

    settings = user_config.read_config(settings_file)
    try:
        n_cores = settings["features"]['n_cores']
    except KeyError:
        n_cores = 1

    if true_calc is None:
        r_min = 2.22
        r_cut = 8 * r_min
        well_depth = 9
        true_calc = LennardJones(sigma=r_min*(2 ** (-1/6)), epsilon=well_depth, rc=r_cut)
        true_calc_type = 'lj'
    print("True calc:", true_calc_type)
    if true_calc_type == "vasp" and structure_input == "POSCAR":
        tmp_poscar_name = "POSCAR_" + str(time.time_ns())
        os.rename("POSCAR", tmp_poscar_name)

    if not resume:
        traj = initial_data_prep(structure_input,
                                 nimages_start,
                                 bspline_config,
                                 true_calc,
                                 true_calc_type,
                                 ufmin_true_fmax,
                                 )
        nimages_start = len(traj)
        init_traj = trajectory.Trajectory(initial_data_output_file, mode='w')
        for atoms in traj:
            init_traj.write(atoms)
        init_traj.close()

        # featurize all but the last image in traj
        #if len(traj) > 1:
        #    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        #        df_features = executor.submit(uf3_run.featurize,
        #                                    bspline_config,
        #                                    traj[:-1],
        #                                    settings_file=settings_file,
        #                                    data_prefix='-1',  # match the initial image_prefix
        #                                    verbose=verbose
        #                                    ).result()
        #    process.save_feature_db(dataframe=df_features, filename=live_features_file)
        #    del df_features
        #    combine_features = True
        #else:
        #    combine_features = False

    atoms = copy.deepcopy(traj[-1])
    most_recent_E_eval = atoms.get_potential_energy()
    most_recent_F_eval = atoms.get_forces()


    '''
    # 2-body preconditioning
    preconditioner_traj = list()
    for pair in bspline_config.interactions_map[2]:
        composition = ''.join(pair)
        estimate_bond_length = \
            ase_data.covalent_radii[ase_data.atomic_numbers[pair[0]]] + ase_data.covalent_radii[ase_data.atomic_numbers[pair[1]]]
        sigma = estimate_bond_length * 2**(-1/6)
        preconditioner_rcut = bspline_config.r_max_map[pair]
        preconditioner_sample_res = 12
        for r in np.linspace(sigma, preconditioner_rcut, preconditioner_sample_res):
            preconditioner_calc = LennardJones(sigma=sigma, epsilon=1.0, rc=preconditioner_rcut)
            preconditioner_traj.append(Atoms(composition,
                                             positions=[[0, 0, 0], [r, 0, 0]],
                                             pbc=[False, False, False],
                                             calculator=preconditioner_calc,
                                             )
                                       )
            preconditioner_traj[-1].get_forces()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        preconditioner_df_features = executor.submit(uf3_run.featurize,
                                                     bspline_config,
                                                     preconditioner_traj,
                                                     settings_file=settings_file,
                                                     data_prefix='preconditioner',
                                                     verbose=verbose
                                                     ).result()
    preconditioner_weight = 0.001
    _, _, preconditioner_x_f, preconditioner_y_f = \
        least_squares.dataframe_to_tuples(preconditioner_df_features,
                                          sample_weights=Pseudodict(preconditioner_weight),
                                          )
    preconditioner_f = [preconditioner_x_f, preconditioner_y_f]
    '''
    preconditioner_f = None

    # initialize UQ object
    r_uq_obj = r_uq.R_UQ(atoms, traj, bspline_config, uq_tolerance)


    ### Optimization Loop ###
    if resume:
        forcecall_counter = len(traj) - nimages_start + 1
        max_forcecalls = forcecall_counter + resume - 1
        combine_features = True
    else:
        forcecall_counter = 1
        combine_features = False
    image_prefix = forcecall_counter - 2
    model_number = forcecall_counter - 1


    while True:  # minimization on true energy surface
        # train UF3
        if combine_features:
            atoms_to_featurize = traj[-1]  # only featurize the newest image to save computation on previously computed images
        else:
            atoms_to_featurize = traj
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            df_features = executor.submit(uf3_run.featurize,
                                          bspline_config,
                                          atoms_to_featurize,
                                          settings_file=settings_file,
                                          data_prefix=str(image_prefix),
                                          verbose=verbose
                                          ).result()
        #df_features = uf3_run.featurize(bspline_config, atoms_to_featurize, settings_file=settings_file, data_prefix=str(forcecall_counter), verbose=verbose)

        # load previously generated features file and combine with new features
        if combine_features:
            prev_features = uf3_run.load_all_features(live_features_file)
            df_features = pd.concat( [prev_features, df_features] )
            os.remove(live_features_file)
            del prev_features
        else:
            combine_features = True
        process.save_feature_db(dataframe=df_features, filename=live_features_file)

        if pretrained_models is None or pretrained_models.get(model_number, None) is None:
            sample_weights = generate_sample_weights(image_prefix, sample_weight_strength, nimages_start=nimages_start)
            if model_file_prefix is None:
                model_file = None
            else:
                model_file = model_file_prefix + "_" + str(model_number) + ".json"
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                model = executor.submit(uf3_run.train,
                                        df_features,
                                        bspline_config,
                                        model_file=model_file,
                                        settings_file=settings_file,
                                        verbose=verbose,
                                        learning_weight=learning_weight,
                                        regularization_values=regularization_values,
                                        sample_weights=sample_weights,
                                        normalize_forces=normalize_forces,
                                        extra_data_f=preconditioner_f,
                                        ).result()
            #model = uf3_run.train(df_features, bspline_config, model_file=model_file, settings_file=settings_file, verbose=verbose,
            #                      learning_weight=learning_weight,
            #                      regularization_values=regularization_values)
        else:
            model_file = pretrained_models[model_number]
            model = least_squares.WeightedLinearModel.from_json(model_file)
        del df_features
        #model = "entire_traj_training/model.json"
        #model = least_squares.WeightedLinearModel.from_json(model)
        #y_e, p_e, y_f, p_f, rmse_e, rmse_f, mae_e, mae_f = uf3_run.calculate_errors(model, df_features)


        ### Minimize on UF3 surface ###
        '''  Primitive atomic displacement UQ method
        def too_uncertain(ref_image, current_image):
            move_threshold = 0.2  # angstroms
            displacements = np.abs(current_image.positions - ref_image.positions)
            print("max disp:", np.max(displacements))
            if np.max(displacements) > move_threshold:
                return True
            return False
        '''

        model_calc = calculator.UFCalculator(model)
        del model
        atoms.calc = model_calc
        dyn = optimizer(atoms)
        step_model_calc_E = list()  # store model calc values for this UF3 minimization step. Will eventually be appended to model_calc_E
        step_model_calc_F = list()
        step_traj = list()
        tmp_atoms = copy.deepcopy(atoms)
        del tmp_atoms.calc
        step_traj.append(tmp_atoms)

        # minimization on UF3
        ufmin_counter = 0
        while True:
            #dyn.run(steps=1, fmax=0.00000000001)  # set force tolerance to a super small number so that this doesn't get in the way of this while loop  FOR SCIPY OPTS
            dyn.run(steps=ufmin_counter+1, fmax=0.000000000001)
            ufmin_counter += 1

            # compare with true energy and froces at this point
            e_val = atoms.get_potential_energy()
            f_val = atoms.get_forces()
            if true_forcecall_always:
                true_atoms = copy.deepcopy(atoms)
                true_atoms.calc = true_calc
                true_e_val = true_atoms.get_potential_energy()
                true_f_val = true_atoms.get_forces()
                if true_calc_type == "vasp":
                    check_vasp_convergence()
                del true_atoms[:]
                del true_atoms
            else:
                true_e_val = most_recent_E_eval
                true_f_val = most_recent_F_eval
            step_model_calc_E.append( (e_val, true_e_val) )
            step_model_calc_F.append( (f_val, true_f_val) )


            uf3_forces_squared = np.sum( np.square(step_model_calc_F[-1][0]), axis=1 )
            uf3_fmax_squared = np.max(uf3_forces_squared)
            tmp_atoms = copy.deepcopy(atoms)
            del tmp_atoms.calc
            step_traj.append(tmp_atoms)

            with open(status_update_file, 'a') as f:
                status = str(forcecall_counter) + ", " + str(ufmin_counter) + ", " + str(step_model_calc_E[-1]) + ", " + str(most_recent_E_eval) + ", " + str(np.sqrt(uf3_fmax_squared)) + "\n"
                f.write(status)

            
            #if too_uncertain(traj[-1], atoms) or uf3_fmax_squared < ufmin_uf3_fmax_squared or ufmin_counter > max_uf3_calls:
            high_uncertainty = r_uq_obj.too_uncertain()
            if high_uncertainty:
                print("HIGH UNCERTAINTY")
            if high_uncertainty or uf3_fmax_squared < ufmin_uf3_fmax_squared or ufmin_counter > max_uf3_calls:
                pickle.dump(step_traj, model_traj_file)
                break
        
        pickle.dump( (step_model_calc_E, step_model_calc_F), model_calc_file )

        # evaluate true energy
        atoms.calc = true_calc
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
        if true_calc_type == "vasp":
            check_vasp_convergence()
        print(forcecall_counter, " ; true E =", most_recent_E_eval)
        true_forces_squared = np.sum( np.square(most_recent_F_eval), axis=1 )
        true_fmax_squared = np.max(true_forces_squared)
        print(forcecall_counter, " ; true F =", np.sqrt(true_fmax_squared))
        stripped_calc_atoms = strip_calc(atoms, most_recent_E_eval, most_recent_F_eval)
        traj.append(stripped_calc_atoms)
        opt_traj.write(stripped_calc_atoms)

        forcecall_counter += 1
        image_prefix += 1
        model_number += 1

        # explicit garbage collection
        del step_model_calc_E[:]
        del step_model_calc_F[:]
        del step_traj[:]
        del step_model_calc_E
        del step_model_calc_F
        del step_traj

        
        gc.collect()

        if true_fmax_squared < ufmin_true_fmax_squared or forcecall_counter > max_forcecalls:
            break


    print("True force calls:", forcecall_counter-1)

    # close open files
    model_traj_file.close()
    model_calc_file.close()
    opt_traj.close()


    # explicit garbage collection
    del traj[:]
    del traj

    del atoms[:]
    del atoms

    gc.collect()

    if true_calc_type == "vasp" and structure_input == "POSCAR":
        os.rename(tmp_poscar_name, "POSCAR")


if __name__ == "__main__":
    ### VARIABLES ###
    structure_input = "POSCAR"
    nimages_start = 1  # number of images to start with
    initial_data_output_file = "initial_data.traj"
    live_features_file = "live_features.h5"
    model_file_prefix = "model"  # store model from each step
    settings_file = "settings.yaml"
    opt_traj_file = "ufmin.traj"  # array of images at all real force evaluations
    model_traj_file = "ufmin_model.traj"  # array of images at each UF3 minimization
    model_calc_file = "model_calc.pckl"  # store energy and forces from UF3 calls
    status_update_file = "ufmin_status.out"
    ufmin_true_fmax = 0.01  # force tolerance for the actual optimization
    ufmin_uf3_fmax = 0.0001  # force tolerance for the optimization on the uf3 surface
    uq_tolerance = 1.0  # how tolerable this workflow is to uncertainty
    preprocess_strength = 0.5
    optimizer = FIRE
    max_forcecalls = 200
    max_uf3_calls = 1000
    verbose = 0
    pretrained_models = None
    normalize_forces = False
    true_forcecall_always = True
    resume = 0

    ### ==================== ###
    ### TRUE CALCULATORS ###
    # Lennard-Jones
    r_min = 2.22
    r_cut = 8 * r_min
    well_depth = 9
    true_calc = LennardJones(sigma=r_min*(2 ** (-1/6)), epsilon=well_depth, rc=r_cut)
    true_calc_type = "lj"

    # Morse
    #r_e = 2.897
    #D_e = 0.7102
    #exp_prefactor = 1.6047
    #rho0 = exp_prefactor * r_e 
    #true_calc = MorsePotential(epsilon=D_e, r0=r_e, rho0=rho0, rcut1=4.0, rcut2=7.0)
    #true_calc_type = "morse"

    # EMT
    #true_calc = EMT()
    #true_calc_type = "emt"

    # UF3
    #true_model = least_squares.WeightedLinearModel.from_json("/home/sung/UFMin/sung/representability_test/fit_3b/trimer_fit/true_model_3b.json")
    #true_calc = calculator.UFCalculator(true_model)
    #true_calc_type = "uf3"

    # VASP
    #true_calc = Vasp(prec = 'Medium',
    #                 xc = 'PBE',
    #                 setups = {'Pt':''},
    #                 ediff = 1e-6,
    #                 #ediffg = -0.01,
    #                 #kpts = 28,
    #                 kspacing = 0.16,
    #                 lcharg = False,
    #                 lwave = False,
    #                 isym = 0,
    #                 ispin = 1,
    #                 ncore = 8,
    #                 algo = 'Normal',
    #                 lreal = 'Auto',
    #                 #encut = 250,
    #                 ismear = 1,
    #                 sigma = 0.2,
    #                 nsw = 0,
    #                 ibrion = -1,
    #                 isif = 2,  # cell changes not yet supported by UFMin
    #                 #icharg = 1,
    #                 #lorbit = 11,
    #                 lasph = True,
    #                 nelm = 40,
    #                 #amix = 0.02,
    #                 #bmix = 0.2,
    #                 #lmaxmix = 4
    #                 )
    #vasp_ntasks = 48
    #os.environ['ASE_VASP_COMMAND'] = f"mpirun -np {vasp_ntasks} vasp_std"
    #os.environ['VASP_PP_PATH'] = "/home/graeme/vasp/"  # dir with potpaw
    #true_calc_type = "vasp"

    ### ==================== ###

    tmp = ufmin(structure_input,
                nimages_start,
                initial_data_output_file,
                live_features_file,
                model_file_prefix,
                settings_file,
                opt_traj_file,
                model_traj_file,
                model_calc_file,
                status_update_file,
                ufmin_true_fmax,
                ufmin_uf3_fmax,
                optimizer,
                max_forcecalls,
                max_uf3_calls,
                verbose,
                true_calc,
                true_calc_type,
                normalize_forces=normalize_forces,
                preprocess_strength=preprocess_strength,
                uq_tolerance=uq_tolerance,
                pretrained_models=pretrained_models,
                true_forcecall_always=true_forcecall_always,
                resume=resume,
                )
    del tmp
    gc.collect()

