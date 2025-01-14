# UF3 Alchemical Compression

This directory contains some scripts for UF3 alchemical compression.
Below are the step-by-step instructions on how to run the files, the contents of each file, and the expected outputs.

## Files

1. `featurize.py`
2. `preprocess.py`
3.
    - a. `train.py`
    - b. `train_torch.py`
4.
    - a. `write_aborted_model.py`
    - b. `symmetrize_model.py`
    - c. `analyze.ipynb`

## TLDR

Run `featurize.py` --> `preprocess.py` --> `train.py`.

To use the trained compressed model with vanilla UF3, you may or may not want to run
`symmetrize_model.py` first, depending on what you're trying to do.

If you run out of memory when you load the model using
`least_squares.WeightedLinearModel`, it may be because vanilla UF3 is trying to assign a
default regularization matrix, which is absolutely enormous. There are some tricks to
prevent this (see [Using Compressed Model with Vanilla UF3](#using-compressed-model-with-vanilla-uf3))

## Instructions

### 1. Set Up & Featurization

**File:** `featurize.py`

**Description:** This script creates and writes the UF3 `BSplineBasis` object and the features dataframe.

**Outputs:**

* `bspline_config.pkl`
* `df_features.h5` 

**Unimportant Details:**

* The user should review and (if necessary) modify the content above `#============`.

* As of now, all interactions of the same interaction order must have the same knot construction.

* The option `sparse_hdf5 = True` is to store the features in a sparse format (`scipy.sparse.csc_matrix`).
This storage option not yet supported in vanilla UF3, but it can reduce the features filesize by more than 100x
(for 5 chemical elements) and dramatically reduce file I/O overhead. Only the storage is compressed; it is
loaded back as the typical vanilla UF3 features dataframe.

* Just FYI: as of now, all symmetries of 3-body interactions are explicitly turned off. Hence the lines:
```
for trio in bspline_config.interactions_map.get(3, []):
    bspline_config.symmetry[trio] = 1 
bspline_config.update_basis_functions()
```


### 2. Preprocessing the Vanilla UF3 Features

**File:** `preprocess.py`

**Description:** This script preprocesses the vanilla UF3 features for faster training. See description of
`AlchemicalModel.preprocess_for_training()` for more details.

**Outputs:**

* `preprocessed.h5`
* `coverage.npz`
* `metadata.npz`

**Unimportant Details:**

* The user should review and (if necessary) modify the content above `#============`.
* Some customization may be required to use only a subset of `df_features.h5` as the training set.
* `sparse_hdf5` should match the value used in the previous step (`featurize.py`) to create `df_features.h5`
* Note that the learning weight (energy vs force weight) is defined in this file.
* I keep getting this warning when `tables` is imported, but I think it's ok:
```
RuntimeWarning: overflow encountered in cast
  infinitymap['float16'] = [-np.float16(np.inf), np.float16(np.inf)]
```


### 3. Training

#### a. Option 1: Alternating Least Squares (ALS)

**File:** `train.py`

**Description:** Trains the compressed model using ALS

**Outputs:**

* `model.json`
    * If training is manually aborted, [`write_aborted_model.py`](#a-writing-aborted-model) can be used to write `model.json`.
* `checkpoint/...`

**Unimportant Details:**

* The user should review and (if necessary) modify the content above `#============`.
* `checkpoint/xxx/alchemical_model_params.npz` contains model parameters at training iteration `xxx`.
* `checkpoint/train_tracker.npz` has information like elapsed time and training RMSE for each training iteration.
* Manually aborted trainings can be resumed by simply changing `resume = True`.
* If `init_params == None`, initial model parameters are randomly initialized. Predefined initializations should be
formatted like `checkpoint/xxx/alchemical_model_params.npz`.

#### b. Option 2: PyTorch

**File:** `train_torch.py`

**Description:** Trains the compressed model using PyTorch

**Outputs:**

* `model.json`
    * Same thing with [`write_aborted_model.py`](#a-writing-aborted-model) applies here too.
* `checkpoint/...`

**Unimportant Details:**

* Same as `train.py`
* The PyTorch model seems to be much more memory inefficient. Some optimizations may be necessary.


### 4. Miscellaneous Scripts

#### a. Writing Aborted Model

**File:** `write_aborted_model.py`

**Description:** Writes `model.json` for aborted training.
Change `alchemical_params_filename` inside this script to point to the desired iteration number.

**Outputs:**

* `model.json`

**Unimportant Details:**

* The user should review and (if necessary) modify the content above `#============`.

#### b. Revert Explicit Asymmetrization

**File:** `symmetrize_model.py`

**Description:** In featurization, we had to explicitly turn off all symmetry for 3-body interactions.
This script un-does this by symmetrizing the trained coefficients for these 3-body interactions.

**Outputs:**

* `model_sym.json`

**Unimportant Details:**

* The user should review and (if necessary) modify the content above `#============`.
    * Level 1: 3-body coefficient tensor is symmetrized, but it is stored back in the full-tensor form.
        Choose this level to keep the model compatible with the featurized dataset.
    * Level 2: 3-body coefficient tensor is symmetrized, and it is stored back in a reduced form after
        redundant coefficients are removed. This would not make it compatible with already-featurized
        datasets. However, this option is probably better to use it for model evaluation (i.e. MD simulations)
        than level 1.

#### c. Analyze Training

**File:** `analyze.ipynb`

**Description:** Just a random Jupyter Notebook that I included here for reference.
It has code used to plot training curves and pseudo-interactions.

## Using Compressed Model with Vanilla UF3

The `model.json` trained by alchemical compression can be used with vanilla UF3, but there are some minor details.

### Loading the Model

#### Regarding 3-body Symmetry

Usually, the model is loaded like this:

```
from uf3.regression import least_squares
model = least_squares.WeightedLinearModel.from_json('model.json')
```

When this is called, it tries to internally reconstruct the `BSplineBasis` object using the r_min/r_max/resolution information
stored in `model.json`. However, as we explicitly turned off 3-body symmetries for alchemical compression, this will throw an error.

One way to fix this is to explicitly load the model (basically do what `from_json()` method does with explicit asymmetry):

```
dump = json_io.load_interaction_map('model.json')
bspline_config = bspline.BSplineBasis.from_dict(dump)
for trio in bspline_config.interactions_map.get(3, []):
    bspline_config.symmetry[trio] = 1
bspline_config.update_basis_functions()
regularizer = dump.get("regularizer", 0)  # we don't want it to create the huge reg matrix
data_coverage = dump.get("data_coverage", None)
model = least_squares.WeightedLinearModel(bspline_config,
                                          regularizer=regularizer,
                                          data_coverage=data_coverage)
model.load(solution=dump)
```

An alternative method is to use [`symmetrize_model.py`](#b-revert-explicit-asymmetrization) with `level=2` and then use the usual
method of loading the model.

#### Regarding Regularization

By default, the `from_json()` method tries to assign a default regularization matrix to the model's `self.regularizer` attribute.
With many elements, the size of this matrix is enormous, and it can quickly consume the available memory.
But, during evaluation (after training), we don't care about the regularizer.

The easiest way to disable the default assignment is to change the following line from `least_squares.WeightedLinearModel.from_json()` method from 

```
regularizer = config.get("regularizer", None)
```

to

```
regularizer = config.get("regularizer", 0)
```

Notice that this was also done in the explicit model loading [above](#regarding-3-body-symmetry).