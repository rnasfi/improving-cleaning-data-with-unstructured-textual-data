# Overview

## Description of the project
The main purpose of this repo is to propose a new approach to clean different datasets.
The data cleaning approach is based on using ML models to extract inforation from unstructured texts to fix inconsistencies in a structured dataset.
A description of use case examples can be found [here](docs/examples.md) to text the approach.

## Approach
An overview of the data cleaning data is decribed [here](docs/approach.md)


***
# Experiment
## Getting started

This project is written in Python.
To run it, as of version 3.8.10, and different python libraries (requirement.txt).

Just pull the code from this repository and install locally the used libraries such as pandas, etc.
Click on the hyperlinks to see example code.


```
cd existing_repo
git remote add origin https://gitlab.com/rnasfi/improving-data-cleaning-with-unstructured-data.git
git branch -M main
git push -uf origin main
```

***

## Set up a virtual environment using pip and Virtualenv 
(this part is inspired from Holoclean repository [here](https://github.com/HoloClean/holoclean/blob/master/README.md))

If you are familiar with `virtualenv`, you can use it to create 
a virtual environment.

For Python 3.8, create a new environment
with your preferred virtualenv wrapper, for example:

* [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) (Bourne-shells)
* [virtualfish](https://virtualfish.readthedocs.io/en/latest/) (fish-shell)


Either follow instructions [here](https://virtualenv.pypa.io/en/stable/installation/) or install via
`pip`.
```bash
$ pip install virtualenv
```

Then, create a `virtualenv` environment by creating a new directory for a Python 3.8 virtualenv environment
```bash
$ mkdir -p ump
$ virtualenv --python=python3.8 ump
```
where `python3.8` is a valid reference to a Python 3.8 executable.

Activate the environment 
```bash
$ source ump/bin/activate
```

#### Install the required python packages

*Note: make sure that the environment is activated throughout the installation process.
When you are done, deactivate it using* 
`conda deactivate`, `source deactivate`, *or* `deactivate` 
*depending on your version*.

In the project root directory, run the following to install the required packages.
Note that this commands installs the packages within the activated virtual environment.

```bash
$ pip install -r ./requirements/requirements_v8.txt
```
*Note for macOS Users:*
you may need to install XCode developer tools using `xcode-select --install`.


#### Make the start example file executable
```bash
$ chmod +x ./start_example.sh
```

#### Make the start example file executable
```bash
$ ./start_example.sh
```

***
# Complementaries

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
The project is still on progress.
