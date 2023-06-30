import ufmin

from ase.io import read
#from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
#from ase.optimize.sciopt import SciPyFminCG
from ase.optimize import FIRE
#from myopts import GD, MGD
from ase.io import trajectory

import os, subprocess, sys, shutil, glob, copy, itertools, random, gc, concurrent
import math

from mpi4py import MPI

import numpy as np
import pandas as pd

#from memory_profiler import profile


#@profile
def run_hpsearch():
    pair = "Pt-Pt"
    settings_template_file = "settings_template.yaml"
    results_file = "results.csv"
    initial_structure = read("POSCAR")

    # hyperparameters to scan
    resolutions = [8, 16, 32, 64, 128, 256, 512, 1024, 2049]
    learning_weights = [0.001, 0.01, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    curvatures_2b = [1e-15, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    resolution_key = "resolution"
    learning_weight_key = "lr"
    curvature_2b_key = "curvature_2b"
    file_write_frequency = 2  # write results file every 15 iterations
    ntasks = len(resolutions) * len(learning_weights) * len(curvatures_2b)

    # inputs to ufmin (file relative to the ./out directory)
    verbose = 0


    # parallelization
    nNodes = 10
    cores_per_node = 48
    total_cores = nNodes * cores_per_node
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    max_threads_per_rank = math.floor(total_cores / nranks)
    if nranks < 2:
        sys.exit("Cannot continue with 1 rank")
    print("MPI nranks:", nranks)
    WORKTAG = 1
    DIETAG = 2


    # rank 0 send out work to other ranks and collects the results
    if rank == 0:
        #with open(results_file, 'w') as rf:
        #    rf.write(resolution_key + ", " + learning_weight_key + ", " + curvature_2b_key + ", " + "forcecalls\n")
        results = pd.DataFrame(columns = [resolution_key, learning_weight_key, curvature_2b_key, "forcecalls"])
        task_counter = 1   
        recv_counter = 0
        result_tracker = dict()  # rank: hps

        # shuffle hps to reduce memory strain
        hps_combo_list = [p for p in itertools.product(resolutions, learning_weights, curvatures_2b)]
        random.shuffle(hps_combo_list)

        for hps_combo in hps_combo_list:
            gc.collect()

            resolution = hps_combo[0]
            learning_weight = hps_combo[1]
            curvature_2b = hps_combo[2]

            hps = {resolution_key: resolution, learning_weight_key: learning_weight, curvature_2b_key: curvature_2b}  # hyperparameters

            # initial work distribution
            if task_counter < nranks:
                worker_rank = task_counter

                # send hyperparameters to work on
                comm.send(hps, dest=worker_rank, tag=WORKTAG)
                print(f"master sent {hps} to {worker_rank}", flush=True)
                result_tracker[worker_rank] = copy.deepcopy(hps)

            else:
                # receive
                stat = MPI.Status()
                nForceCalls = comm.recv(source=MPI.ANY_SOURCE, status=stat)
                worker_rank = stat.Get_source()
                recv_counter += 1
                print(f"master received {nForceCalls} from {worker_rank}", flush=True)

                # write to results file
                #with open(results_file, 'a') as rf:
                #    rf.write(str(result_tracker[worker_rank][resolution_key]) + ", " +
                #             str(result_tracker[worker_rank][learning_weight_key]) + ", "+
                #             str(result_tracker[worker_rank][curvature_2b_key]) + ", " +
                #             str(nForceCalls) + "\n")
                result_tracker[worker_rank]["forcecalls"] = nForceCalls
                tmp_df = pd.DataFrame(result_tracker[worker_rank], index=[recv_counter])
                results = pd.concat([results, tmp_df])
                del tmp_df

                #send next work
                comm.send(hps, dest=worker_rank, tag=WORKTAG)
                print(f"master sent {hps} to {worker_rank}", flush=True)
                result_tracker[worker_rank] = copy.deepcopy(hps)

            task_counter += 1
            del hps

            if recv_counter % file_write_frequency == 0:
                results.to_csv(results_file, index=False)

        # no more work. receive outstanding results
        while recv_counter < ntasks:
            nForceCalls = comm.recv(source=MPI.ANY_SOURCE, status=stat)
            recv_counter += 1
            worker_rank = stat.Get_source()
            print("end: master received {nForceCalls} from {worker_rank}", flush=True)
            #with open(results_file, 'a') as rf:
            #    rf.write(str(result_tracker[worker_rank][resolution_key]) + ", " +
            #             str(result_tracker[worker_rank][learning_weight_key]) + ", "+
            #             str(result_tracker[worker_rank][curvature_2b_key]) + ", " +
            #             str(nForceCalls) + "\n")
            result_tracker[worker_rank]["forcecalls"] = nForceCalls
            tmp_df = pd.DataFrame(result_tracker[worker_rank], index=[recv_counter])
            results = pd.concat([results, tmp_df])
            del tmp_df

            if recv_counter % file_write_frequency == 0:
                results.to_csv(results_file, index=False)


        

        # Tell all ranks to terminate
        for rank in range(1, nranks):
            comm.send(-1, dest=rank, tag=DIETAG)

        
        #rf.close()
        results.to_csv(results_file, index=False)

    else:  # not rank 0
        opt_traj_file = f"ufmin_{rank}.traj"  # array of images at all real force evaluations
        model_traj_file = f"ufmin_model_{rank}.traj"  # array of images at each UF3 minimization
        true_calc_file = f"true_calc_{rank}.pckl"  # store energy and forces from each true force call
        model_calc_file = f"model_calc_{rank}.pckl"  # store energy and forces from UF3 calls
        train_uq_file = f"train_uq_{rank}.pckl"  # store UQ from training data
        test_uq_file = f"test_uq_{rank}.pckl"  # store UQ from testing data (each structure in UF3 minimization steps)
        status_update_file = f"ufmin_status_{rank}.out"
        live_features_file = f"live_features_{rank}.h5"
        settings_file = f"settings_{rank}.yaml"
        out_directory = f"./out_{rank}"

        # make settings file for this rank
        shutil.copyfile(settings_template_file, settings_file)
        with open(settings_file, 'a') as sf:
            sf.write(f"outputs_path: \"{out_directory}\"")
        
        while True:  # run until told to die
            gc.collect()
            if os.path.exists(out_directory):
                shutil.rmtree(out_directory)
            if os.path.exists(live_features_file):
                os.remove(live_features_file)
            if os.path.exists(opt_traj_file):
                os.remove(opt_traj_file)
            if os.path.exists(model_traj_file):
                os.remove(model_traj_file)
            if os.path.exists(true_calc_file):
                os.remove(true_calc_file)
            if os.path.exists(model_calc_file):
                os.remove(model_calc_file)
            if os.path.exists(train_uq_file):
                os.remove(train_uq_file)
            if os.path.exists(test_uq_file):
                os.remove(test_uq_file)
            if os.path.exists(status_update_file):
                os.remove(status_update_file)

            stat = MPI.Status()
            hps = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
            print(f"worker {rank} got {hps}", flush=True)
            if (stat.Get_tag() == DIETAG):
                print(f"worker {rank} dying", flush=True)
                break

            # run ufmin
            resolution = hps[resolution_key]
            resolution_map = {pair: resolution}
            curvature_2b = hps[curvature_2b_key]
            regularization_values = {"ridge_1b":0, "ridge_2b": 0, "curvature_2b": curvature_2b}
            learning_weight = hps[learning_weight_key]
            os.mkdir(out_directory)

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads_per_rank) as executor:
                with open( os.path.join(out_directory, ".run_info"), 'w' ) as rif:
                    rif.write(str(hps) + "\n")
                nForceCalls = executor.submit(work,
                                              initial_structure,
                                              None,
                                              resolution_map,
                                              learning_weight,
                                              regularization_values,
                                              settings_file,
                                              live_features_file,
                                              opt_traj_file,
                                              model_traj_file,
                                              true_calc_file,
                                              model_calc_file,
                                              train_uq_file,
                                              test_uq_file,
                                              status_update_file,
                                              verbose                                             
                                              ).result()

            comm.send(nForceCalls, dest=0)


def work(initial_structure,
         model_file_prefix,
         resolution_map,
         learning_weight,
         regularization_values,
         settings_file,
         live_features_file,
         opt_traj_file,
         model_traj_file,
         true_calc_file,
         model_calc_file,
         train_uq_file,
         test_uq_file,
         status_update_file,
         verbose):
    atoms = copy.deepcopy(initial_structure)
    try:
        tmp = ufmin.ufmin(initial_structure=initial_structure,
                          model_file_prefix=model_file_prefix,
                          resolution_map=resolution_map,
                          learning_weight=learning_weight,
                          regularization_values=regularization_values,
                          settings_file=settings_file,
                          live_features_file=live_features_file,
                          opt_traj_file=opt_traj_file,
                          model_traj_file=model_traj_file,
                          true_calc_file=true_calc_file,
                          model_calc_file=model_calc_file,
                          train_uq_file=train_uq_file,
                          test_uq_file=test_uq_file,
                          status_update_file=status_update_file,
                          verbose=verbose)
        with open(status_update_file, "r") as sf:
            lines = sf.readlines()
            nForceCalls = int( lines[-1].split(',')[0] ) + 2
    except np.linalg.LinAlgError:  # probably singular matrix
        nForceCalls = -1
    del atoms[:]
    del atoms

    return nForceCalls


   
if __name__ == "__main__":
    run_hpsearch()
