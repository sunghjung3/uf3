
from autograd import elementwise_grad, hessian
import autograd.numpy as agnp
import numpy as np
import pandas as pd
from scipy.stats.distributions import t
#from scipy.stats import norm as t
import functools



__all__ = ["get_hessian_inv", "get_uncert", "too_uncertain"]


def get_input_output(df):
    """Extracts the input and output variables from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        A tuple containing two numpy arrays: the input variables and the output variables.
    """
    y = np.asarray(df['y'])
    try:
        x = df.drop(['y'],axis=1)
    except:
        x = df.drop('y')
    return np.asarray(x), y     

def get_energy_force(df):
    """Extracts the energy and force data from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        A tuple containing four numpy arrays: the input variables for energy, the output variables for energy,
        the input variables for force, and the output variables for force, as well as a numpy array of the
        number of atoms.
    """
    idx = pd.IndexSlice
    try:
        energies = df.loc[idx[:,'energy'],idx[:]]
    except:
        energies = df.loc[idx['energy'],idx[:]]
    #size = np.asarray(energies['n_Pt'])        
    x_e,y_e = get_input_output(energies)
    forces = df.loc[~df.index.isin(energies.index)]
    x_f,y_f = get_input_output(forces)
    #return x_e,y_e,x_f,y_f,size
    return x_e, y_e, x_f, y_f

def compute_pred(theta, x):
    """Computes the predicted output of a linear regression model given the input and parameters.

    Args:
        theta (np.ndarray): A 1D array of regression coefficients.
        x (np.ndarray): A 2D array of input data, where each row represents a single data point.

    Returns:
        A 1D array of predicted output values.
    """
    return x@theta


def sse(theta, x, y):
    """Computes the sum of squared errors (SSE) of a linear regression model given the parameters.

    Args:
        theta (np.ndarray): A 1D array of regression coefficients.

    Returns:
        The sum of squared errors of the model.
    """
    return agnp.sum((compute_pred(theta,x)-y)**2)


def get_hessian_inv(theta, x, y):
    """ 
    Computes the inverse of the Hessian matrix of the sum of squared errors (sse) function
    evaluated at the given theta vector.

    Args:
    theta (numpy.ndarray): The parameter vector.

    Returns:
    numpy.ndarray: The inverse of the Hessian matrix of the sse function.
    """
    parameterized_sse = functools.partial(sse, x=x, y=y)
    h = hessian(parameterized_sse)(theta)
    w,v = np.linalg.eig(h)
    eps = max((1e-5, 1.05*np.abs(min(w))))
    h = h + eps*np.eye(len(w))
    p = sse(theta, x, y)/len(x)*np.linalg.pinv(h)   
    return p


def get_uncertainty(x, theta, p, confidence=0.975, scale=1):
    """Computes the uncertainty estimates for the coefficients of a linear regression model.

    Args:
        x (np.ndarray): A 2D array of input data, where each row represents a single data point.
        theta (np.ndarray): A 1D array of regression coefficients.
        p (numpy.ndarray): inverse Hessian matrix of the SSE loss function
        confidence (float): The confidence level for the uncertainty estimates. Defaults to 0.975.
        scale (float): A scaling factor for the uncertainty estimates. Defaults to 1.

    Returns:
        A 1D array of uncertainty estimates for each regression coefficient.
    """
    t_val = t.ppf(confidence,x.shape[1])
    uncerts = []
    for xi in x:
        gp = elementwise_grad(compute_pred,0)(theta,xi)
        uncerts += [np.sqrt(gp@(p*scale)@gp)]
    return np.array(uncerts)*t_val


def too_uncertain(x, theta, p):
    get_uncertainty(x, theta, p)
    # if uncertainty > cutoff
        # return True
    return False


if __name__ == "__main__":
    print("run at /home/sung/UFMin/sung/data/emt_pt38/1/1/1")
    import uf3_run
    from uf3.regression import least_squares
    import pickle, itertools

    from ase.calculators.emt import EMT


    settings_file = "settings.yaml"
    train_features_file = "live_features.h5"
    test_datafile = "ufmin_model.traj"
    test_features_file = "modeltraj_features.h5"
    model_file = "model_16.json"

    nAtoms = 38  # number of atoms
    model = least_squares.WeightedLinearModel.from_json(model_file)
    with open(test_datafile, 'rb') as f:
        test_data = pickle.load(f)
    test_data = list(itertools.chain.from_iterable(test_data))
    for atoms in test_data:
        atoms.calc = EMT()
        atoms.get_potential_energy()
        atoms.get_forces()

    df_features = uf3_run.load_all_features(train_features_file)

    x_e, y_e, x_f, y_f = get_energy_force(df_features)
    p_e = model.predict(x_e)
    p_f = model.predict(x_f)
    mae_e = least_squares.mae_metric(y_e, p_e)
    mae_f = least_squares.mae_metric(y_f, p_f)
    x = np.vstack((x_e,x_f))
    y = np.concatenate((y_e,y_f))

    # testing on ufmin_model.traj
    #bspline_config = uf3_run.initialize(settings_file)
    #df_features_test = uf3_run.featurize(bspline_config, test_data, test_features_file, settings_file, data_prefix="test")
    df_features_test = uf3_run.load_all_features(test_features_file)
    x_e_test, y_e_test, x_f_test, y_f_test = get_energy_force(df_features_test)
    p_e_test = model.predict(x_e_test)
    p_f_test = model.predict(x_f_test)
    #mae_e_test = least_squares.mae_metric(y_e_test, p_e_test)
    #mae_f_test = least_squares.mae_metric(y_f_test, p_f_test)

    p = get_hessian_inv(model.coefficients, x, y)

    uq_e = get_uncertainty(x_e, model.coefficients, p, scale=mae_e)
    uq_f = get_uncertainty(x_f, model.coefficients, p, scale=mae_f)
    uq_e_test = get_uncertainty(x_e_test, model.coefficients, p, scale=mae_e)
    uq_f_test = get_uncertainty(x_f_test, model.coefficients, p, scale=mae_f)


    def create_parity_plot_frame(y_i, p_i, error, units, title, img_filename):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.errorbar(p_i, y_i, xerr=error, fmt='.', c='b', zorder=3)
        #ax.scatter(p_all, y_all, s=5, c='r', zorder=2)
        #ax.scatter(p_md, y_md, s=5, c='g', zorder=1)
        ax.axline([0,0], c="k", ls="dashed", slope=1, linewidth=0.5)
        #ax.text(0.55, 0.05, f"Training RMSE = {'%.5f'%(rmse_i)}\nFull traj RMSE = {'%.5f'%(rmse_all)}\nMD traj RMSE={'%.5f'%(rmse_md)}", transform=ax.transAxes)
        ax.set_xlabel(f"Predicted ({units})")
        ax.set_ylabel(f"True ({units})")
        #ax.legend(["Training", "Full", "MD", "Ideal"])
        fig.suptitle(title)
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        plt.savefig(img_filename, transparent=False, facecolor='white')
        plt.close()
    
    create_parity_plot_frame(y_e, p_e, uq_e, "eV", "Training energy", "parity_train_e.png")
    create_parity_plot_frame(y_f, p_f, uq_f, "eV/A", "Training force", "parity_train_f.png")
    create_parity_plot_frame(y_e_test, p_e_test, uq_e_test, "eV", "Test energy", "parity_test_e.png")
    create_parity_plot_frame(y_f_test, p_f_test, uq_f_test, "eV/A", "Test force", "parity_test_f.png")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(uq_f,bins=10,alpha = 0.6,density= True, label="train")
    ax.hist(uq_f_test, bins=10, alpha=0.6, density=True, label="test")
    ax.legend(loc="upper right")
    plt.show()
