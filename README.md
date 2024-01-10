# Deep Learning (DS 330)
# American University of Armenia - Fall 2023

## 1. Python Setup

The exercises are implemented in Python 3.10, so this is what we are going to install here.

To avoid issues with different versions of Python and Python packages we recommend that you always set up a project specific virtual environment. The most common tools for a clean management of Python environments are *pyenv*, *virtualenv* and *Anaconda*. For simplicity, we are going to focus on Anaconda.

### Anaconda setup
Download and install miniconda (minimal setup with less start up libraries) or conda (full install but larger file size) from [here](https://www.anaconda.com/products/distribution#Downloads). Create an environment using the terminal command:

`conda create --name dl_homeworks python==3.10`

Next activate the environment using the command:

`conda activate dl_homeworks`

Continue with installation of requirements and starting jupyter notebook using:

`pip install -r requirements.txt` 

`jupyter notebook`

Jupyter notebooks use the python version of the current active environment so make sure to always activate the `dl_homeworks` environment before working on notebooks for this class.

## 2. Homeworks Download

The homeworks will be uploaded to to moodle. Each time we start a new homework you will have to unzip the homework and copy it into the current directory as we are utilizing some shared folders.

### The directory layout for the homeworks

    DL_homeworks
    ├── datasets      # The datasets will be stored here
    ├── homework_00   # Optional / not graded                     
    ├── homework_02                    
    ├── homework_03                    
    ├── homework_04
    ├── homework_05
    ├── homework_06
    ├── homework_07                              
    ├── homework_08
    ├── homework_09
    ├── homework_10
    ├── output         # Where you will find zipped homeworks for uploading
    └── README.md
    └── requirements.txt


## 3. Dataset Download

Datasets will generally be downloaded automatically by homework notebooks and stored in a common datasets directory shared among all homeworks. A sample directory structure for cifar10 dataset is shown below:-

    DL_homeworks
        ├── datasets                   # The datasets required for all homeworks will be downloaded here
            ├── cifar10                # Dataset directory
                ├── cifar10.p          # dataset files 

## 4. Homework Submission
After you have worked through a homework exercise, execute the notebook cells that saves and zips the homework. The output can be found in the global `DL_homeworks/output` folder. You will need to upload this zip file to moodle. homework_00 will not be graded so you don't need to upload it.
