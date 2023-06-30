#!/usr/bin/env python

import argparse, os, warnings, functools
from typing import List, Tuple, Dict, Callable
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ase.atoms import Atoms

from uf3.data import composition, io
from uf3.representation import bspline, process
from uf3.regression import least_squares
from uf3.util import user_config, json_io


__all__ = ["load_all_features", "initialize", "featurize", "train", "calculate_errors"]  # only allow imports of these variables/functions/classes


def log(message=None, message_verbosity=1, user_verbosity=1):  # TODO: eventually want to replace with python's logging
    if user_verbosity >= message_verbosity:
        print(message)


def complete_map_construction(map: dict,
                              chemical_system: composition.ChemicalSystem,
                              map_name: str) -> Dict:
    """
    Automated construction of maps (r_min_map, r_max_map, resolution_map) from partial user inputs, including
    proper UF3 formatting and filling in user-defined default values for unspecified interactions.
    """
    if (map is None) or (not map):
        return map

    # user-defined default values (if provided)
    user_default_2b = map.pop("other_2b", None)
    if user_default_2b is not None:
        if not isinstance(user_default_2b, (int, float)):
            raise TypeError(f"Invalid format of {map_name} 'other_2b' key in the settings file.")
    user_default_3b = map.pop("other_3b", None)
    if user_default_3b is not None:
        if not ( isinstance(user_default_3b, list) and all(isinstance(n, (int, float)) for n in user_default_3b) ):
            raise TypeError(f"Invalid format of {map_name} 'other_3b' key in the settings file.")

    return_map = dict()
    reference_interactions = chemical_system.get_interactions_list()
    element_list = chemical_system.element_list
    for element in element_list:
        reference_interactions.remove(element)  # remove all 1 body terms from interactions list

    # populate map from user with correct UF3 syntax
    for key, value in map.items():  
        interaction = key.strip().split('-')  # e.g. "Si-C-N " -> ["Si", "C", "N"]
        for element in interaction:
            if element not in element_list:  # verify that extraneous elements are not present
                raise Exception(f"Extraneous element {element} found in {map_name} {key}. Only include elements in {element_list}.")
        sorted_interaction = composition.sort_interaction_symbols(interaction)  # sort order to match UF3 ordering
        if len(sorted_interaction) > 2 and interaction != sorted_interaction:  # 3 body was reordered
            sorted_value = [value[1], value[0], value[2]]
        else:
            sorted_value = value
        if isinstance(sorted_value, (int, float)) or (isinstance(sorted_value, list) and all(isinstance(n, (int, float)) for n in sorted_value)):
            return_map[sorted_interaction] = sorted_value
            try:
                reference_interactions.remove(sorted_interaction)
            except:
                raise ValueError(f"Interaction {key} not found in chemical system of elements {element_list}.")
        else:
            raise Exception(f"Invalid format of {map_name} {key} in the settings file.")

    # fill in user-defined defaults for all other unspecified interactions
    for additional_interaction in reference_interactions:
        if len(additional_interaction) == 2 and user_default_2b is not None:
            return_map[additional_interaction] = user_default_2b
        elif len(additional_interaction) == 3 and user_default_3b is not None:
            return_map[additional_interaction] = user_default_3b

    return return_map


def write_file_with_overwrite_policy(logger: Callable,
                                     overwrite: bool = False,
                                     file_path: str = None,
                                     action: Callable = None,
                                     **kwargs
                                     ):
    """
    Write file keeping in mind user's overwrite policy.

    Args:
        logger (Callable): a partial instance of the log() function. The "user_verbosity" kwarg should already be initialized.
        overwrite (bool): will overwrite a file if it already exists with the same name.
        file_path (str): path and name of the file to write.
        action (Callable): function that executes the file writing.
        **kwargs: keyword arguments to pass into the "action" function.
    """
    if file_path is None:
        return False  # exit without writing
    if os.path.isfile(file_path) and (not overwrite):
        logger(f"{file_path} already exists. Not overwriting.", message_verbosity=1)
    else:
        if os.path.isfile(file_path) and overwrite:
            warnings.warn(f"{file_path} already exists. Overwriting...")
            os.remove(file_path)
        action(**kwargs)
        logger(f"Successfully wrote {file_path}.", message_verbosity=1)


def resolve_input_conflict(passed_value, settings_dict: dict, settings_key: str, name: str, override: bool = True) -> str:
    def args_over_settings(arg_value, settings_dict: dict, settings_key: str, name: str) -> str:
        if arg_value is None:
            try:
                if name == "features file name":
                    print(settings_dict)
                arg_value = settings_dict[settings_key]
            except KeyError:
                warnings.warn(f"{name} not provided anywhere. It will not be read or written.")
                arg_value = None
        return arg_value

    def settings_over_args(arg_value, settings_dict: dict, settings_key: str, name: str) -> str:
        arg_value = settings_dict.get(settings_key, arg_value)
        if arg_value is None:
            warnings.warn(f"{name} not provided anywhere. It will not be read or written.")
        return arg_value

    if override:
        return args_over_settings(passed_value, settings_dict, settings_key, name)
    else:
        return settings_over_args(passed_value, settings_dict, settings_key, name)


def load_all_features(features_file: str) -> pd.DataFrame:
    """
    Loads the full contents of a given features file to a Pandas dataframe.

    Args:
        features_file (str): name of the features file.

    Returns:
        df_features (pd.DataFrame): features dataframe.
    """
    _, _, table_names, _ = io.analyze_hdf_tables(features_file)
    return pd.concat( [process.load_feature_db(features_file, table_name) for table_name in table_names] )


def initialize(settings_file: str = "settings.yaml",
               resolution_map: dict = None,
               knots_map: dict = None,
               override: bool = True,
               overwrite: bool = True,
               verbose: int = 1
               ) -> bspline.BSplineBasis:
    """
    Initialize b-splines for UF3 training.

    Args:
        settings_file (str): name of the settings file (default: settings.yaml).
            For an example of the content, see "defaults_options.yaml" at `uf3.__file__` (execute this in Python to see path)
        resolution_map (dict): map of resolution (number of knot intervals) per interaction (default: 20 for 2b, 5 for 3b).
        knots_map (dict): pre-generated map of knots.
            Overrides other settings.
        override (bool): these inputs override any conflicting values in the settings file in case of a conflict (default: True)
        overwrite (bool): overwrite files that exist already (default: True)
        verbose (int): verbosity level logging (default: 1)
            0: does not log anything other than errors and warnings
            1: all of 0 plus some reminders and info about the state of training
            2: all of 1 plus heavy outputs (for debugging)

    Returns:
        bspline_config (bspline.BSplineBasis): B-spline basis configurations for the given settings.
    """
    logger = functools.partial(log, user_verbosity=verbose)  # partially initialized logging function
    logger(f"* verbose level = {verbose}", message_verbosity=2)
    logger(f"* overwriting files that exist already: {overwrite}", message_verbosity=2)

    # read some things from the settings file
    settings = user_config.read_config(settings_file)
    logger(f"* Settings: {settings}", message_verbosity=2)
    output_dir_root = settings.get("outputs_path", ".")  # use current directory if not specified
    logger(message=f"* outputs_path: {output_dir_root}", message_verbosity=2)
    if not os.path.exists(output_dir_root):
        logger(f"Output directory ({output_dir_root}) does not exist. Creating...", message_verbosity=1)
        os.mkdir(output_dir_root)
    random_seed = settings.get("seed", None)
    logger(f"* random_seed: {random_seed}", message_verbosity=2)
    np.random.seed(random_seed)

    # chemical system
    element_list = settings["elements"]
    if isinstance(element_list, str):
        element_list = element_list.split()
    element_list = [element.title() for element in element_list]  # just to make sure to have all 1st character of elements capitalized
    if len(element_list) < 1:
        raise RuntimeError(f"No elements given. Check 'elements' in {settings_file}")
    logger(f"* element_list: {element_list}", message_verbosity=2)
    degree = settings["degree"]
    if degree != 2 and degree != 3:
        raise ValueError(f"UF3 only supports 2- and 3-body interactions at this moment. Detected degree={degree}. Check 'degree' in {settings_file}")
    logger(f"* degree: {degree}", message_verbosity=2)
    chemical_system = composition.ChemicalSystem(element_list=element_list, degree=degree)
    
    # B-spline basis. If not found in the settings file, get default values from uf3/uf3/default_options.yaml
    basis_settings = settings.get("basis", dict())
    logger(f"* basis settings from user: {basis_settings}", message_verbosity=2)
    r_min_map = basis_settings.get("r_min", None)
    r_max_map = basis_settings.get("r_max", None)
    #resolution_map = basis_settings.get("resolution", None)
    resolution_map = resolve_input_conflict(resolution_map, basis_settings, "resolution", "resolution", override)
    r_min_map = complete_map_construction(r_min_map, chemical_system, "r_min")
    r_max_map = complete_map_construction(r_max_map, chemical_system, "r_max")
    resolution_map = complete_map_construction(resolution_map, chemical_system, "resolution")
    logger(f"* r_min_map from user: {r_min_map}", message_verbosity=2)
    logger(f"* r_max_map from user: {r_max_map}", message_verbosity=2)
    logger(f"* resolution_map from user: {resolution_map}", message_verbosity=2)
    knot_strategy = basis_settings.get("knot_strategy", "linear")
    fit_offsets = basis_settings.get("fit_offsets", True)
    trailing_trim = basis_settings.get("trailing_trim", 3)  # default value: 3 (continuous upto 2nd derivative at r=r_max)
    load_knots = basis_settings.get("load_knots", False)
    dump_knots = basis_settings.get("dump_knots", False)
    leading_trim = 0  # discontinuous at r=r_min (repulsive region)
    if load_knots:
        try:
            logger(f"Loading knots...", message_verbosity=2)
            knots_file = basis_settings["knots_path"]
            logger(f"Knots loaded!", message_verbosity=2)
        except KeyError:
            raise Exception(f"The knots_file name must be provided in order to load knots from a file.")
        knots_map = bspline.parse_knots_file(knots_file, chemical_system=chemical_system)
        logger(f"* knots_map from file {knots_file}: {knots_map}", message_verbosity=2)
    bspline_config = bspline.BSplineBasis(chemical_system=chemical_system,
                                          r_min_map=r_min_map,
                                          r_max_map=r_max_map,
                                          resolution_map=resolution_map,
                                          knot_strategy=knot_strategy,
                                          offset_1b=fit_offsets,
                                          leading_trim=leading_trim,
                                          trailing_trim=trailing_trim,
                                          knots_map=knots_map)
    logger(bspline_config, message_verbosity=1)

    if dump_knots:  # write knots_map to a file
        try:
            knots_outfile_path = os.path.join(output_dir_root, basis_settings["knots_path"])
        except KeyError:
            raise Exception(f"The name for the knots output file must be provided in order to write knots to a file.")
        write_file_with_overwrite_policy(logger, overwrite, knots_outfile_path,
                                         json_io.dump_interaction_map,
                                         interaction_map=bspline_config.knots_map,
                                         filename=knots_outfile_path,
                                         write=True)
    
    return bspline_config


def featurize(bspline_config: bspline.BSplineBasis,
              data: Atoms | List[Atoms] | str = None,
              features_file: str = None,
              settings_file: str = "settings.yaml",
              data_prefix: str = None,
              return_df_data: bool = False,
              override: bool = True,
              overwrite: bool = True,
              verbose: int = 1
              ) -> pd.DataFrame:
    """
    Featurize dataset for UF3 training.

    Args:
        bspline_config (bspline.BSplineBasis): B-spline basis configurations for training;
            Return value from uf3_run.initialize()
        data (ase.atoms.Atoms | str): training data, either Atoms object(s), file name, or directory name.
            File can be any format supported by ase.io.read() or additional database formats supported by
            uf3.data.io.parse_trajectory().
            If passing in Atoms object(s), the current UF3 implementation recommends that Atoms.get_potential_energy() and 
            Atoms.get_forces() have been called before.
            Or instead, it may be possible to set both Atoms.calc = <calc> and Atoms.calc.atoms = <Atoms> (but not guaranteed to work).
        features_file (str): name of features file to store results from this function.
        settings_file (str): name of the settings file (default: settings.yaml).
            For an example of the content, see "defaults_options.yaml" at `uf3.__file__` (execute this in Python to see path)
        data_prefix (str): prefix for data and features labeling.
            Useful for debugging if data/features from multiple sources will eventually be combined.
            If data is read from a file, takes the filename as the default prefix.
            If data is read from an Atoms object, takes the current time (time.time_ns()) as the default prefix.
        return_df_data (bool): returns Pandas dataframe of the training data created during the featurization process if True,
            in addition to the features dataframe (default: False)
        override (bool): these inputs override any conflicting values in the settings file in case of a conflict (default: True)
        overwrite (bool): overwrite files that exist already (default: True)
        verbose (int): verbosity level logging (default: 1)
            0: does not log anything other than errors and warnings
            1: all of 0 plus some reminders and info about the state of training
            2: all of 1 plus heavy outputs (for debugging)

    Returns:
        df_features (pd.DataFrame): features dataframe
    """
    logger = functools.partial(log, user_verbosity=verbose)  # partially initialized logging function
    logger(f"* verbose level = {verbose}", message_verbosity=2)
    logger(f"* function arguments have higher precedence than settings file: {override}", message_verbosity=2)
    logger(f"* overwriting files that exist already: {overwrite}", message_verbosity=2)

    # resolve filename conflicts
    settings = user_config.read_config(settings_file)
    data_settings = settings.get("data", dict())
    features_settings = settings.get("features", dict())
    data = resolve_input_conflict(data, data_settings, "db_path", "data", override)
    if data is None:
        raise Exception("You must provide training data.")
    features_file = resolve_input_conflict(features_file, features_settings, "features_path", "features file name", override)
    logger(f"* data: {data}", message_verbosity=2)
    logger(f"* features_file: {features_file}", message_verbosity=2)

    # read training data and create dataframe
    data_coordinator = io.DataCoordinator()
    if isinstance(data, str):  # data is a file or directory
        try:  # single file
            if data_prefix is None:
                data_prefix = data
            data_coordinator.dataframe_from_trajectory(data, prefix=data_prefix)
        except:  # maybe a directory of files
            for datafile in os.listdir(data):  # iterate through all files in the directory
                if data_prefix is None:
                    data_prefix = datafile
                data_coordinator.dataframe_from_trajectory(datafile, prefix=data_prefix)
    else:  # data is an Atoms object or a list of them
        if isinstance(data, Atoms):
            data = [data]  # list of Atoms objects
        if data_prefix is None:
            data_prefix = str(time.time_ns())
        try:
            data_coordinator.dataframe_from_lists(data, prefix=data_prefix, copy=False)  # TODO: probably have to add prefix as input to function to consolidate multiple h5 files during active learning
        except:
            raise Exception("Could not read data. Check if it is a valid Atoms object or a list of Atoms objects. If they have a calculator attached, make sure it has been initialized properly.")
    df_data = data_coordinator.consolidate()
    elements_in_data = set(df_data.iloc[0][data_coordinator.atoms_key].get_chemical_symbols())
    if set(bspline_config.element_list) != elements_in_data:
        raise TypeError(f"Element types in the data ({elements_in_data}) do not match the element list provided in the settings file ({bspline_config.element_list}).")
    logger(f"Number of training images: {len(df_data)}", message_verbosity=1)
    logger(f"Data:\n{df_data.info}", message_verbosity=2)

    # featurization
    logger(f"Starting featurization...", message_verbosity=1)
    output_dir_root = settings.get("outputs_path", ".")  # use current directory if not specified
    if features_file is None:
        features_outfile_path = None
    else:
        features_outfile_path = os.path.join(output_dir_root, features_file)
        features_exist_but_overwrite = os.path.isfile(features_outfile_path) and overwrite
        if features_exist_but_overwrite:  # an additional checkpoint to prevent accidental deletion of features file, which may take long to regenerate
            warnings.warn(f"{features_outfile_path} already exists and will be overwritten. Interrupt before the completion of featurization to prevent overwriting.")
        #if features_exist_but_overwrite or (not os.path.isfile(features_outfile_path)):  # need to go through featurization
    n_cores = features_settings.get("n_cores", 16)  # number of cores to use for featurization step
    logger(f"* n_cores for featurization: {n_cores}", message_verbosity=2)
    client = ProcessPoolExecutor(max_workers=n_cores)
    fit_forces = features_settings.get("fit_forces", True)
    column_prefix = features_settings.get("column_prefix", 'x')
    representation = process.BasisFeaturizer(bspline_config=bspline_config,
                                            fit_forces=fit_forces,
                                            prefix=column_prefix) 
    df_features = representation.evaluate_parallel(df_data, client, n_jobs=n_cores*50, progress="bar")
    logger(f"Number of features: {len(df_features)}", message_verbosity=2)
    logger(f"Features:\n{df_features.info}", message_verbosity=2)
    write_file_with_overwrite_policy(logger, overwrite, features_outfile_path,
                                     process.save_feature_db,
                                     dataframe=df_features,
                                     filename=features_outfile_path
                                     )  # XXX: option to add table name as kwarg. Maybe useful during active learning?
    logger(f"Featurization finished!", message_verbosity=1)

    if return_df_data:
        return df_features, df_data
    return df_features


def train(features: pd.DataFrame | str,
          bspline_config: bspline.BSplineBasis,
          model_file: str = None,
          learning_weight: float = 0.5,
          regularization_values: dict = None,
          settings_file: str = "settings.yaml",
          override: bool = True,
          overwrite: bool = True,
          verbose: int = 1
          ) -> least_squares.WeightedLinearModel:
    """
    Trains a UF3 model.

    Args:
        features (pd.DataFrame | str): Pandas dataframe or name of .h5 file containing featurizations.
        bspline_config (bspline.BSplineBasis): B-spline basis configurations computed from the initialize() function.
        model_file (str): name of the .json file to write the trained model to.
        learning_weight (float): parameter balancing contribution from energies vs. forces (default: 0.5). Higher values favor energies.
        regularization_values (dict): dict of regularization values.
            Keys: ridge_1b, ridge_2b, ridge_3b, curvature_2b, curvature_3b
        settings_file (str): name of the settings file (default: settings.yaml).
            For an example of the content, see "defaults_options.yaml" at `uf3.__file__` (execute this in Python to see path)
        override (bool): these inputs override any conflicting values in the settings file in case of a conflict (default: True)
        overwrite (bool): overwrite files that exist already (default: True)
        verbose (int): verbosity level logging (default: 1)
            0: does not log anything other than errors and warnings
            1: all of 0 plus some reminders and info about the state of training
            2: all of 1 plus heavy outputs (for debugging)

    Returns:
        model (least_squares.WeightedLinearModel): trained UF3 model.
    """
    logger = functools.partial(log, user_verbosity=verbose)  # partially initialized logging function
    logger(f"* verbose level = {verbose}", message_verbosity=2)
    logger(f"* function arguments have higher precedence than settings file: {override}", message_verbosity=2)
    logger(f"* overwriting files that exist already: {overwrite}", message_verbosity=2)

    settings = user_config.read_config(settings_file)
    models_settings = settings.get("model", dict())
    features_settings = settings.get("features", dict())
    learning_settings = settings.get("learning", dict())
    model_file = resolve_input_conflict(model_file, models_settings, "model_path", "model file name", override)
    logger(f"* model_file: {model_file}", message_verbosity=2)
    if isinstance(features, str):
        features = resolve_input_conflict(features, features_settings, "features_path", "features file", override)
        df_features = load_all_features(features)
    elif isinstance(features, pd.DataFrame):
        df_features = features
    else:
        raise TypeError(f"Invalid data type of features. Please provide a features dataframe or file name.")

    ### Fit UF3 model from features dataframe ###
    logger("Training model...", message_verbosity=1)
    #learning_weight = learning_settings.get("weight", 0.5)
    learning_weight = resolve_input_conflict(learning_weight, learning_settings, "weight", "learning weight", override)
    #regularization_values = learning_settings.get("regularizer", dict())
    regularization_values = resolve_input_conflict(regularization_values, learning_settings, "regularizer", "regularizer", override)
    if regularization_values is None:
        regularization_values = dict()
    regularizer = bspline_config.get_regularization_matrix(**regularization_values)
    model = least_squares.WeightedLinearModel(bspline_config=bspline_config,
                                              regularizer=regularizer,
                                              )  # XXX: there is another arg called "data coverage", and I have no idea what it does
    n_elements = len(bspline_config.chemical_system.element_list)
    x_e, y_e, x_f, y_f = least_squares.dataframe_to_tuples(df_features, n_elements=n_elements)  # x: input, y: prediction
    model.fit(x_e, y_e, x_f, y_f, weight=learning_weight)
    if model_file is not None:
        output_dir_root = settings.get("outputs_path", ".")
        model_outfile_path = os.path.join(output_dir_root, model_file)
        write_file_with_overwrite_policy(logger, overwrite, model_outfile_path,
                                         model.to_json,
                                         filename=model_outfile_path)
    logger("Training finished!", message_verbosity=1)
    
    # XXX: there are some postprocessing methods in UF3 for the repulsive region (in least_squares.py). They don't seem to make a big difference
    return model


def calculate_errors(model: least_squares.WeightedLinearModel | str,
                     features: pd.DataFrame | str,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Wrapper for uf3.regression.least_squares.batched_prediction() and subset_prediction().
    Get model predictions for featurized data (either dataframe or file) and calculate RMSEs.

    Args:
        model (least_squares.WeightedLinearModel | str): UF3 model object or file name
        features (pd.DataFrame | str): featurization dataframe or file name

    Returns:
        y_e (np.ndarray): true energies
        p_e (np.ndarray): energies predicted by the model
        y_f (np.ndarray): true forces
        p_f (np.ndarray): forces predicted by the model
        rmse_e (float): energy RMSE of the model
        rmse_f (float): force RMSE of the model
        mae_e (float): energy MAE of the model
        mae_f (float): force MAE of the model
    """
    if isinstance(model, str):  # model from file
        model = least_squares.WeightedLinearModel.from_json(model)

    if isinstance(features, str):  # from file
        y_e, p_e, y_f, p_f, rmse_e, rmse_f = model.batched_predict(features)
        mae_e = least_squares.mae_metric(y_e, p_e)
        mae_f = least_squares.mae_metric(y_f, p_f)
    elif isinstance(features, pd.DataFrame):  # from dataframe
        y_e, p_e, y_f, p_f = least_squares.subset_prediction(features, model)
        rmse_e = least_squares.rmse_metric(y_e, p_e)
        rmse_f = least_squares.rmse_metric(y_f, p_f)
        mae_e = least_squares.mae_metric(y_e, p_e)
        mae_f = least_squares.mae_metric(y_f, p_f)
    else:
        raise TypeError(f"Invalid format of features data. Pass in a dataframe or file name.")
    
    return y_e, p_e, y_f, p_f, rmse_e, rmse_f, mae_e, mae_f


if __name__ == "__main__":  # if this script is called from command line instead of being imported
    # include the -h flag for help with arguments

    def parse_args():
        """
        Parses arguments passed in through the command line
        """
        parser = argparse.ArgumentParser(prog="uf3_train", formatter_class=argparse.RawTextHelpFormatter, description=   
    """Trains a UF3 model.

    NOTE: all the optional arguments regarding file names can also be specified in the settings file.
    The values passed through the command line take precedence, unless the --no_override flag is also passed.
    
    NOTE 2: calling this script from python allows more inputs (e.g. knot resolution, learning weight, knot map, etc.) to be passed through.""")
        parser.add_argument("-s", "--settings", help=
    """name of the settings file (default: settings.yaml)""",
                            default="settings.yaml")
        parser.add_argument("-d", "--data", help=
    """name of training data file. Either a file or a directory.
    Files can be any format supported by ase.io.read() or additional database formats
    supported by uf3.data.io.parse_trajectory().""",
                            default=None)
        parser.add_argument("-m", "--model", help=
    """name of file to save the trained model.""",
                            default=None)
        parser.add_argument("-f", "--features", help=
    """name of featurizations (aka \"descriptors\") file.""",
                            default=None)
        parser.add_argument("--d2f", help=
    """only run featurization but not training (d2f = "data to features").
    Will write features to the features file, unless prevented by the overwrite policy (below).""",
                            action="store_true")
        parser.add_argument("--f2m", help=
    """only run training from features (f2m = "features to model").
    Will read features from the features file. No other training data needed.""",
                            action="store_true")
        parser.add_argument("--no_override", help=
    """do not override settings file values with command-line inputs in case of a conflict (default: False).
    By default, values passed in through the command line has higher priority than the settings file.""",
                            action="store_false")  # default is True on the command line
        parser.add_argument("--overwrite", help=
    """overwrite files that already exist (default: False)""",
                            action="store_true")  # default is False on the command line
        parser.add_argument("--verbose", type=int, help=
    """Verbosity level of logging.
    Possible values:
        0: does not log anything other than errors and warnings
        1: all of 0 plus some reminders and info about the state of training (default)
        2: all of 1 plus heavy outputs (for debugging)""",
                            default=1)
        args = parser.parse_args()
        return args

    args = parse_args()
    settings_file = args.settings
    data_file = args.data
    model_file = args.model  # file to save the model after training
    features_file = args.features
    featurize_only = args.d2f
    train_only = args.f2m
    override = not(args.no_override)
    overwrite = args.overwrite
    verbose = args.verbose

    full_run = not(featurize_only ^ train_only)  # run full (data to model) only if d2f and f2m are both the same value

    bspline_config = initialize(settings_file=settings_file,
                                overwrite=overwrite,
                                verbose=verbose)

    if full_run or featurize_only:
        df_features = featurize(bspline_config,
                             data=data_file,
                             features_file=features_file,
                             settings_file=settings_file,
                             override=override,
                             overwrite=overwrite,
                             verbose=verbose)
    elif train_only:
        # load features from file
        settings = user_config.read_config(settings_file)
        features_settings = settings.get("features", dict())
        features = resolve_input_conflict(features_file, features_settings, "features_path", "features file", override)
        df_features = load_all_features(features)

    if full_run or train_only:
        model = train(df_features,
                      bspline_config,
                      model_file=model_file,
                      override=override,
                      overwrite=overwrite,
                      verbose=verbose)
        _, _, _, _, rmse_e, rmse_f, mae_e, mae_f = calculate_errors(model, df_features)
        print(f"Energy RMSE: {rmse_e}")
        print(f"Force RMSE : {rmse_f}")
        print(f"Energy MAE : {mae_e}")
        print(f"Force MAE  : {mae_f}")
