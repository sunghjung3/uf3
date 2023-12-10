from ase.io import read

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
from ase.io import trajectory
from ase.atoms import Atoms
import ase.data as ase_data

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
import delta_uq, r_uq
from preprocess import preprocess

import copy, sys, time, os, glob, pickle, gc, concurrent

#from memory_profiler import profile


def generate_sample_weights(current_forcecall, strength):
    """
    Generate sample weights for training data.

    Args:
        current_forcecall (int): Current forcecall number.
        strength (int): How strongly to weigh the most recent sample vs the oldest.
            The most recent datapoint will be weighted by 2**(strength).
            Each previous datapoint will be weighted half as much until a weight of 1.0 is reached.
            All datapoints before that will be weighted 1.0.

    Returns:
        sample_weights (dict): Sample weights for training data.
    """
    sample_weights = dict()
    for i in range(current_forcecall, 0, -1):
        key = str(i) + "_" + '0'  # the '0' is assuming that UF3 names each configuration by its index in the trajetory and that we only have one new configuration per forcecall
        sample_weights[key] = 2 ** max(strength - (current_forcecall - i), 0)
    sample_weights["1_1"] = sample_weights["1_0"]  # we start with 2 forcecalls
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

def strip_calc(atoms, e_val, f_val):
    """
    Given an atoms object and energy and force values, strip the calculator
    and replace it with a single point calculator with the same energy and force
    values.
    """
    ret_atoms = copy.deepcopy(atoms)
    ret_atoms.calc = SinglePointCalculator(ret_atoms, energy=e_val, forces=f_val)
    return ret_atoms

def check_vasp_convergence(vasprun_file="vasprun.xml") -> bool:
    if Vasprun(vasprun_file).converged_electronic:
        return True
    raise ValueError("VASP calculation did not converge. Terminating UFMin.")

#@profile
def ufmin(initial_structure = "POSCAR",
          live_features_file = "live_features.h5",
          model_file_prefix = "model",  # store model from each step
          settings_file = "settings.yaml",
          opt_traj_file = "ufmin.traj",  # array of images at all real force evaluations
          model_traj_file = "ufmin_model.traj",  # array of images at each UF3 minimization
          true_calc_file = "true_calc.pckl",  # store energy and forces from each true force call
          model_calc_file = "model_calc.pckl",  # store energy and forces from UF3 calls
          train_uq_file = "train_uq.pckl",  # store UQ from training data
          test_uq_file = "test_uq.pckl",  # store UQ from testing data (each structure in UF3 minimization steps)
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

    Args:
        initial_structure (str | ase.Atoms): Initial structure to start optimization from.
            If str, then it is a path to a structure file.
            If ase.Atoms, then it is the structure itself.
        live_features_file (str): Path to HDF5 file to store live features during optimization.
        model_file_prefix (str): Prefix for model files.
            Each model will be saved as `model_file_prefix`_`forcecall_counter`.json
        settings_file (str): Path to settings file.
        opt_traj_file (str): Path to ASE trajectory file to store optimization trajectory at all real force evaluations.
        model_traj_file (str): Path to ASE trajectory file to store optimization trajectory at each UF3 minimization for all real force evaluations.
        true_calc_file (str): Path to pickle file to store energy and forces from each true force call.
        model_calc_file (str): Path to pickle file to store energy and forces from UF3 calls (real and MLFF).
        train_uq_file (str): Path to pickle file to store UQ from training data (only for delta_uq).
        test_uq_file (str): Path to pickle file to store UQ from testing data (each structure in UF3 minimization steps) (only for delta_uq).
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
        resume (int): Start from scratch if 0.
    """

    ufmin_true_fmax_squared = ufmin_true_fmax ** 2
    ufmin_uf3_fmax_squared = ufmin_uf3_fmax ** 2

    if resume:
        # load existing files to resume process

        try:
            opt_traj = trajectory.Trajectory(opt_traj_file, mode='r')
            traj = [image for image in opt_traj]
            opt_traj.close()
        except FileNotFoundError:
            sys.exit(f"Optimization trajectory file {opt_traj_file} does not exist. Cannot resume.")
        
        if not os.path.isfile(model_traj_file):
            sys.exit(f"Model trajectory file {model_traj_file} does not exist. Cannot resume.")
        model_traj_file = open(model_traj_file, 'ab')

        if not (os.path.isfile(true_calc_file) and os.path.isfile(model_calc_file)) :
            sys.exit(f"Calculation pickle files do not exist. Cannot resume.")
        true_calc_file = open(true_calc_file, 'ab')
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
        if os.path.isfile(true_calc_file) or os.path.isfile(model_calc_file):
            sys.exit("Remove the calculation pickle files before running this script.")
        if os.path.isfile(status_update_file):
            os.remove(status_update_file)
        traj = list()  # will be training data
        model_traj_file = open(model_traj_file, 'wb')
        true_calc_file = open(true_calc_file, 'wb')
        model_calc_file = open(model_calc_file, 'wb')
        #uq_e_train_list = list()
        #uq_f_train_list = list()
        #uq_e_test_list = list()
        #uq_f_test_list = list()


    ### Set up ###

    # one-time bspline configuration
    bspline_config = uf3_run.initialize(settings_file, verbose=verbose, resolution_map=resolution_map)

    settings = user_config.read_config(settings_file)
    try:
        n_cores = settings["features"]['n_cores']
    except KeyError:
        n_cores = 1

    if resume:
        atoms = copy.deepcopy(traj[-1])
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
    else:

        # initial structure
        if isinstance(initial_structure, str):
            atoms = read(initial_structure)
        else:
            assert isinstance(initial_structure, Atoms)
            atoms = copy.deepcopy(initial_structure)

        # preprocess
        pair_tuples = bspline_config.interactions_map[2]
        atoms = preprocess(atoms, pair_tuples, strength=preprocess_strength)

        #atoms = "emt_pt38.traj"
        #atoms = read(atoms, index="587")
        if true_calc is None:
            r_min = 2.22
            r_cut = 8 * r_min
            well_depth = 9
            true_calc = LennardJones(sigma=r_min*(2 ** (-1/6)), epsilon=well_depth, rc=r_cut)
            true_calc_type = 'lj'
        print("True calc:", true_calc_type)
        if true_calc_type == "vasp" and initial_structure == "POSCAR":
            tmp_poscar_name = "POSCAR_" + str(time.time_ns())
            os.rename("POSCAR", tmp_poscar_name)
        atoms.calc = true_calc
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
        if true_calc_type == "vasp":
            check_vasp_convergence()
        pickle.dump( (most_recent_E_eval, most_recent_F_eval), true_calc_file )
        #traj.append(copy.deepcopy(atoms))
        traj.append(strip_calc(atoms, most_recent_E_eval, most_recent_F_eval))

        # UF3 cannot train with a single structure (look at loss function). So take 1 real optimization step to gain another image
        #dyn = SciPyFminCG(atoms)
        dyn = optimizer(atoms)
        dyn.run(steps=1)
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
        if true_calc_type == "vasp":
            check_vasp_convergence()
        pickle.dump( (most_recent_E_eval, most_recent_F_eval), true_calc_file )
        #traj.append(copy.deepcopy(atoms))
        traj.append(strip_calc(atoms, most_recent_E_eval, most_recent_F_eval))

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
        forcecall_counter = len(traj) - 1
        max_forcecalls = forcecall_counter + resume - 1  # -1 because of the way the outer while loop termination was written
    else:
        forcecall_counter = 1  # already have forcecalls 0 and 1 from the setup stage

    while True:  # minimization on true energy surface
        # train UF3
        if forcecall_counter > 1:
            atoms_to_featurize = traj[-1]  # only featurize the newest image to save computation on previously computed images
            combine_features = True
        else:  # 1st iteration of this loop
            atoms_to_featurize = traj
            combine_features = False
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            df_features = executor.submit(uf3_run.featurize,
                                          bspline_config,
                                          atoms_to_featurize,
                                          settings_file=settings_file,
                                          data_prefix=str(forcecall_counter),
                                          verbose=verbose
                                          ).result()
        #df_features = uf3_run.featurize(bspline_config, atoms_to_featurize, settings_file=settings_file, data_prefix=str(forcecall_counter), verbose=verbose)
        if combine_features:
            # load previously generated features file and combine with new features
            prev_features = uf3_run.load_all_features(live_features_file)
            df_features = pd.concat( [prev_features, df_features] )
            os.remove(live_features_file)
            del prev_features
        process.save_feature_db(dataframe=df_features, filename=live_features_file)

        if pretrained_models is None or pretrained_models.get(forcecall_counter, None) is None:
            sample_weights = generate_sample_weights(forcecall_counter, sample_weight_strength)
            if model_file_prefix is None:
                model_file = None
            else:
                model_file = model_file_prefix + "_" + str(forcecall_counter) + ".json"
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
            model_file = pretrained_models[forcecall_counter]
            model = least_squares.WeightedLinearModel.from_json(model_file)
        del df_features
        #model = "entire_traj_training/model.json"
        #model = least_squares.WeightedLinearModel.from_json(model)
        #y_e, p_e, y_f, p_f, rmse_e, rmse_f, mae_e, mae_f = uf3_run.calculate_errors(model, df_features)

        '''
        # preprocessing for delta method UQ
        x_e_train, y_e_train, x_f_train, y_f_train = delta_uq.get_energy_force(df_features)
        p_e_train = model.predict(x_e_train)
        p_f_train = model.predict(x_f_train)
        mae_e = least_squares.mae_metric(p_e_train, y_e_train)
        mae_f = least_squares.mae_metric(p_f_train, y_f_train)
        p = delta_uq.get_hessian_inv(model.coefficients, np.vstack((x_e_train,x_f_train)), np.concatenate((y_e_train,y_f_train)))
        uq_e_train = delta_uq.get_uncertainty(x_e_train, model.coefficients, p, scale=mae_e)
        uq_f_train = delta_uq.get_uncertainty(x_f_train, model.coefficients, p, scale=mae_f)
        uq_e_train_list.append(uq_e_train)
        uq_f_train_list.append(uq_f_train)'''


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
        #step_uq_e_list = list()
        #step_uq_f_list = list()
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

            '''
            # delta UQ method
            uq_features = uf3_run.featurize(bspline_config, true_atoms, settings_file=settings_file, verbose=verbose)  # using true atoms instead of atoms so that forces will also be featurized
            x_e_test, y_e_test, x_f_test, y_f_test = delta_uq.get_energy_force(uq_features)
            uq_e_test = delta_uq.get_uncertainty(x_e_test, model.coefficients, p, scale=mae_e)
            step_uq_e_list.append(uq_e_test)
            uq_f_test = delta_uq.get_uncertainty(x_f_test, model.coefficients, p, scale=mae_f)
            step_uq_f_list.append(uq_f_test)
            '''
            
            #if too_uncertain(traj[-1], atoms) or uf3_fmax_squared < ufmin_uf3_fmax_squared or ufmin_counter > max_uf3_calls:
            high_uncertainty = r_uq_obj.too_uncertain()
            if high_uncertainty:
                print("HIGH UNCERTAINTY")
            if high_uncertainty or uf3_fmax_squared < ufmin_uf3_fmax_squared or ufmin_counter > max_uf3_calls:
                pickle.dump(step_traj, model_traj_file)
                break
        
        pickle.dump( (step_model_calc_E, step_model_calc_F), model_calc_file )
        #uq_e_test_list.append(step_uq_e_list)
        #uq_f_test_list.append(step_uq_f_list)

        # evaluate true energy
        forcecall_counter += 1
        atoms.calc = true_calc
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
        if true_calc_type == "vasp":
            check_vasp_convergence()
        pickle.dump( (most_recent_E_eval, most_recent_F_eval), true_calc_file )
        print(forcecall_counter, " ; true E =", most_recent_E_eval)
        true_forces_squared = np.sum( np.square(most_recent_F_eval), axis=1 )
        true_fmax_squared = np.max(true_forces_squared)
        print(forcecall_counter, " ; true F =", np.sqrt(true_fmax_squared))
        #traj.append(copy.deepcopy(atoms))
        traj.append(strip_calc(atoms, most_recent_E_eval, most_recent_F_eval))

        # explicit garbage collection
        del step_model_calc_E[:]
        del step_model_calc_F[:]
        #del step_uq_e_list[:]
        #del step_uq_f_list[:]
        del step_traj[:]
        del step_model_calc_E
        del step_model_calc_F
        #del step_uq_e_list 
        #del step_uq_f_list 
        del step_traj

        
        gc.collect()

        if true_fmax_squared < ufmin_true_fmax_squared or forcecall_counter > max_forcecalls:  # should be a >=, but whatever...
            break


    print("True force calls:", forcecall_counter)

    # close open files
    model_traj_file.close()
    true_calc_file.close()
    model_calc_file.close()

    # save optimization traj
    opt_traj = trajectory.Trajectory(opt_traj_file, mode='w')
    for image in traj:
        opt_traj.write(image)
    opt_traj.close()

    '''
    # save delta UQ data
    with open(train_uq_file, 'wb') as f:
        pickle.dump([uq_e_train_list, uq_f_train_list], f)
    with open(test_uq_file, 'wb') as f:
        pickle.dump([uq_e_test_list, uq_f_test_list], f)
    '''

    # explicit garbage collection
    del traj[:]
    #del uq_e_train_list[:]
    #del uq_f_train_list[:]
    #del uq_e_test_list[:]
    #del uq_f_test_list[:]
    del traj
    #del uq_e_train_list
    #del uq_f_train_list
    #del uq_e_test_list 
    #del uq_f_test_list 

    del atoms[:]
    del atoms

    gc.collect()

    if true_calc_type == "vasp" and initial_structure == "POSCAR":
        os.rename(tmp_poscar_name, "POSCAR")


if __name__ == "__main__":
    ### VARIABLES ###
    initial_structure = "POSCAR"
    live_features_file = "live_features.h5"
    model_file_prefix = "model"  # store model from each step
    settings_file = "settings.yaml"
    opt_traj_file = "ufmin.traj"  # array of images at all real force evaluations
    model_traj_file = "ufmin_model.traj"  # array of images at each UF3 minimization
    true_calc_file = "true_calc.pckl"  # store energy and forces from each true force call
    model_calc_file = "model_calc.pckl"  # store energy and forces from UF3 calls
    train_uq_file = "train_uq.pckl"  # store UQ from training data
    test_uq_file = "test_uq.pckl"  # store UQ from testing data (each structure in UF3 minimization steps)
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

    tmp = ufmin(initial_structure,
                live_features_file,
                model_file_prefix,
                settings_file,
                opt_traj_file,
                model_traj_file,
                true_calc_file,
                model_calc_file,
                train_uq_file,
                test_uq_file,
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

