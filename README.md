# eMFP

This repository contains the code used in the research article *"Embedded Morgan Fingerprints for more efficient molecular property predictions with machine learning"*.

## Datasets

The following datasets were obtained from their original sources:

| Dataset Name                          | DOI                                         |
|-------------------------------------|---------------------------------------------|
| RedDB Database                      | https://doi.org/10.1038/s41597-022-01832-2  |
| Non-Fullerene Acceptors Database    | https://doi.org/10.1016/j.joule.2017.10.006 |
| QM9 Database                       | https://doi.org/10.1038/sdata.2014.22       |

All datasets have been cleaned and preprocessed within this repository.

⚠️ **Warning**  
Before training any model, it is necessary to extract all `.csv.gz` files located in the `Datasets` directory, as well as those in its subdirectories.

--- 

## Setting up the Conda Virtual Environment

To ensure that all dependencies required to run the scripts are correctly installed, this repository includes a Conda environment configuration file named `environment.yml`.

### Steps to create and activate the environment:

1. **Create the environment:**

   Open a terminal and navigate to the root directory of this repository (where `environment.yml` is located). Then run:

   ```bash
   conda env create -f environment.yml
    ```



This command will create a new Conda environment named `emfp` (as specified in the YML file) and install all necessary packages.

2. **Activate the environment:**

   After the environment has been created, activate it with:

   ```bash
   conda activate emfp
   ```

3. **Verify the environment is active:**

   You should see `(emfp)` at the beginning of your terminal prompt, indicating the environment is active.

---

⚠️ **Warning**
Although the `environment.yml` file has been tested on Ubuntu systems, package incompatibilities may occasionally arise. If you encounter issues installing the environment with Conda, it is recommended to manually check and resolve package compatibility conflicts one by one.


## Training

Training is performed using two scripts:

- `train_cv_dnn_sys_external.py`: trains the Deep Neural Network (DNN) model.
- `train_cv_skl_sys_external.py`: trains the traditional machine learning models (`RF`, `GBR`, `MLP`, and `KNR`).

Both scripts receive the same positional arguments:

| Position | Argument | Type | Possible values / Description | Default |
|----------|----------|------|-------------------------------|---------|
| 1 | `databaseName` | string | Dataset: `rdb`, `nfa`, `qm9` | `rdb` |
| 2 | `encodingMethod` | string | Molecular encoding: `mfp`, `emfp` | `mfp` |
| 3 | `embeddingSize` | int | `1` for MFP; `8`, `16`, `32`, `64`, `128`, `256` for eMFP | `1` |
| 4 | `withDescriptors` | bool | `True` or `False` | `False` |
| 5 | `ffnnCase` | string | `none`, `linear`, `gauss` | `none` |
| 6 | `ffnnOrder` | int | FFNN order (`1`, `2`, `3`, ...) | `1` |
| 7 | `nBitsMFP` | int | Number of Morgan fingerprint bits (`1024`, `2048`, `4096`, ..., `16384`, ...) | `16384` |
| 8 | `radiusMFP` | int | Morgan fingerprint radius (`0`, `1`, `2`, `3`, `4`, ...) | `2` |
| 9a | `modelName` | string | `RF`, `GBR`, `MLP`, `KNR` | `RF`, use with train_cv_skl_sys_external.py |
| 9b | `modelName` | string | `DNN` | `DNN`, use with train_cv_dnn_sys_external.py |
| 10 | `outterKFold` | int | Outer cross-validation fold (`1` to `5`) | `1` |
| 11 | `int_ext_case` | string | `internal` or `external` | `internal` |

> **Note**
>
> The `int_ext_case` argument determines the execution mode:
>
> - `internal`: performs hyperparameter optimization (HPO) using Optuna.
> - `external`: loads the Optuna `.db` database and trains the best model found in the stored trials.

---

## Running Examples

### Traditional Machine Learning Models (RF, GBR, MLP, KNR)

Traditional machine learning models are trained using:

```text
train_cv_skl_sys_external.py
```

### Example 1: Morgan Fingerprints (MFP) with Random Forest (RF)

```bash
python train_cv_skl_sys_external.py rdb mfp 1 False linear 1 16384 2 RF 1 internal
```

### Example 2: Embedded Morgan Fingerprints (eMFP) with compression size 64

```bash
python train_cv_skl_sys_external.py rdb emfp 64 False linear 1 16384 2 RF 1 internal
```

> **Note**
>
> Use `internal` to perform hyperparameter optimization with Optuna. Use `external` to load the corresponding Optuna `.db` file and train the best model obtained during the optimization.

---

## Running the Deep Neural Network (DNN)

The DNN model is trained using:

```text
train_cv_dnn_sys_external.py
```

### Example

```bash
python train_cv_dnn_sys_external.py rdb emfp 64 False linear 1 16384 2 DNN 1 internal
```

> **Note**
>
> As with the traditional machine learning models:
>
> - `internal` performs hyperparameter optimization (HPO) using Optuna.
> - `external` loads the Optuna `.db` file and trains the best DNN model found during the optimization trials.