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

- `train_dnn.py`: for training models with Deep Neural Network model (`models.py`)
- `train_other_models.py`: for training traditional machine learning models.

The available arguments for both scripts are listed below:

| Argument      | Type      | Possible values / Description                          | Required  | Default value     |
|---------------|-----------|--------------------------------------------------------|-----------|-------------------|
| `-file`       | string    | Name of the input file containing SMILES               | Mandatory | `None`            |
| `-mfp`        | flag      | Use Morgan Fingerprint (does not require `-size`)      | Optional  | `False`           |
| `-emfp`       | flag      | Use embedded MFP (requires `-size`)                    | Optional  | `False`           |
| `-size`       | int       | Compression factor: `4`, `8`, `16`, `32`, `64`         | Optional  | `None`            |
| `-none`       | flag      | No FFNN applied                                        | Optional  | `False`           |
| `-linear`     | flag      | Linear FFNN                                            | Optional  | `False`           |
| `-order`      | int       | FFNN order: `1`, `2`, `3`, ...  (requires `-linear`)   | Optional  | `1`               |
| `-nB`         | int       | Number of bits for MFP: `1024`, `2048`, `4096`, ...    | Mandatory  | `16384`           |
| `-rd`         | int       | Radius for MFP: `2`, `3`, `4`, `5`                      | Mandatory  | `2`               |

### Additional argument for `train_other_models.py`:

| Argument      | Type      | Possible values / Description                          | Required  | Default value     |
|---------------|-----------|--------------------------------------------------------|-----------|-------------------|
| `-model`      | string    | ML model to train: `RF`, `GBR`, `KNR`, `MLP`           | Mandatory  | `None`            |


## Running Example

### Example 1: Using Morgan Fingerprints (MFP) with Random Forest (RF)

To run a calculation on the RedDB dataset with MFP and an RF model, use the following command:

```bash
python train_other_models.py -file Datasets/trainDb/clean_db_reddb.csv -mfp -linear -order 1 -nB 16384 -rd 2 -model RF
```


## Running Examples
Running a example with same parameters by changing in the use of eMFP.

### Example 1: Using Morgan Fingerprints (MFP) with Random Forest (RF)

To run a calculation on the RedDB dataset with MFP and an RF model, use the following command:

```bash
python train_other_models.py -file Datasets/trainDb/clean_db_reddb.csv -mfp -linear -order 1 -nB 16384 -rd 2 -model RF
```


### Example 2: Using embedded Morgan Fingerprints (eMFP) with compression factor 64

To run the same dataset using embedded MFP with a compression size of 64 and RF model:

```bash
python train_other_models.py -file Datasets/trainDb/clean_db_reddb.csv -emfp -size 64 -linear -order 1 -nB 16384 -rd 2 -model RF
```
