from ase.io import read
#from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
#from ase.optimize.sciopt import SciPyFminCG
from ase.optimize import FIRE
#from myopts import GD, MGD
from ase.io import trajectory
from ase.atoms import Atoms

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
from r_uq import global_dr_trust

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
          resolution_map = None,
          learning_weight = 0.5,
          regularization_values = None,
          dr_trust = global_dr_trust,
          sample_weight_strength = 6,
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
        resolution_map (dict): Resolution map for features.
        learning_weight (float): Weight for learning term in loss function.
        regularization_values (list): ridge and curvature regularization values.
        dr_trust (float): Trust distance deviation for r-based UQ.
        sample_weight_strength (int): how strongly to weigh the most recent sample vs the oldest.
            See `generate_sample_weights` function for more details.
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
        #atoms = "emt_pt38.traj"
        #atoms = read(atoms, index="587")
        if true_calc is None:
            r_min = 2.22
            r_cut = 8 * r_min
            well_depth = 9
            true_calc = LennardJones(sigma=r_min*(2 ** (-1/6)), epsilon=well_depth, rc=r_cut)
        atoms.calc = true_calc
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
        pickle.dump( (most_recent_E_eval, most_recent_F_eval), true_calc_file )
        traj.append(copy.deepcopy(atoms))

        # UF3 cannot train with a single structure (look at loss function). So take 1 real optimization step to gain another image
        #dyn = SciPyFminCG(atoms)
        dyn = optimizer(atoms)
        dyn.run(steps=1)
        most_recent_E_eval = atoms.get_potential_energy()
        most_recent_F_eval = atoms.get_forces()
        pickle.dump( (most_recent_E_eval, most_recent_F_eval), true_calc_file )
        traj.append(copy.deepcopy(atoms))


    # one-time bspline configuration
    bspline_config = uf3_run.initialize(settings_file, verbose=verbose, resolution_map=resolution_map)

    # initialize UQ object
    r_uq_obj = r_uq.R_UQ(atoms, traj, bspline_config, dr_trust)


    ### Optimization Loop ###
    if resume:
        forcecall_counter = len(traj) - 1
        max_forcecalls = forcecall_counter + resume - 1  # -1 because of the way the outer while loop termination was written
    else:
        forcecall_counter = 1  # already have forcecalls 0 and 1 from the setup stage
    settings = user_config.read_config(settings_file)
    try:
        n_cores = settings["features"]['n_cores']
    except KeyError:
        n_cores = 1
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
                                    ).result()
        #model = uf3_run.train(df_features, bspline_config, model_file=model_file, settings_file=settings_file, verbose=verbose,
        #                      learning_weight=learning_weight,
        #                      regularization_values=regularization_values)
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
            true_atoms = copy.deepcopy(atoms)
            true_atoms.calc = true_calc
            step_model_calc_E.append( (atoms.get_potential_energy(), true_atoms.get_potential_energy()) )
            step_model_calc_F.append( (atoms.get_forces(), true_atoms.get_forces()) )

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
            del true_atoms[:]
            del true_atoms

            
            #if too_uncertain(traj[-1], atoms) or uf3_fmax_squared < ufmin_uf3_fmax_squared or ufmin_counter > max_uf3_calls:
            if r_uq_obj.too_uncertain() or uf3_fmax_squared < ufmin_uf3_fmax_squared or ufmin_counter > max_uf3_calls:
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
        pickle.dump( (most_recent_E_eval, most_recent_F_eval), true_calc_file )
        print(forcecall_counter, " ; true E =", most_recent_E_eval)
        true_forces_squared = np.sum( np.square(most_recent_F_eval), axis=1 )
        true_fmax_squared = np.max(true_forces_squared)
        print(forcecall_counter, " ; true F =", np.sqrt(true_fmax_squared))
        traj.append(copy.deepcopy(atoms))

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
    ufmin_true_fmax = 0.05  # force tolerance for the actual optimization
    ufmin_uf3_fmax = 0.05  # force tolerance for the optimization on the uf3 surface
    dr_trust = 0.64  # trust distance deviation for r-based UQ
    optimizer = FIRE
    max_forcecalls = 200
    max_uf3_calls = 1000
    verbose = 0
    resume = 0

    r_min = 2.22
    r_cut = 8 * r_min
    well_depth = 9
    true_calc = LennardJones(sigma=r_min*(2 ** (-1/6)), epsilon=well_depth, rc=r_cut)

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
                dr_trust=dr_trust,
                resume=resume,
                )
    del tmp
    gc.collect()

