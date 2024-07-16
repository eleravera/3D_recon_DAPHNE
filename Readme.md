This Readme refers to the DAPHNE project which is a companion of the book
Panetta D, Camarlinghi N. 3D Image Reconstruction for CT and PET : A Practical Guide with Python. CRC Press; 2020.
Available from: https://www.taylorfrancis.com/books/9780429270239

# Installation
To use this project you will need to working
* python >=3.6.8
* python3-venv package
* pip or conda package manager
It is recommended to run the project within a python virtual env.
For more information on python virtual env the reader is referred
https://docs.python.org/3/library/venv.html

## 1. Create and activate the virtual env
#### UNIX
open a shell and create a python virtual env running the command

``python3 -m venv reconstruction-book``

to activate the virtual env run

``source reconstruction-book/bin/activate``
### Windows
open a command prompt and create a python virtual env running

``c:\Python3\python -m venv c:\path\to\myenv\reconstruction-book``

activate the virtual env run

``c:\path\to\myenv\reconstruction-book\Scripts\activate.bat``

## 2. Install all the packages needed for this project
#### UNIX and Windows (pip)

install all the packages needed by this project

``pip install -r requirements.txt``

## 3. Deactivate the virtual env

to deactivate the virtual env, type

### UNIX
run

``deactivate``

### Windows

``cd c:\path\to\myenv\reconstruction-book\Scripts\``

``deactivate``

### 4. Virtual env usage

If you want to reuse previously created virtual env, run

### UNIX

``source reconstruction-book/bin/activate``

### Windows

``c:\path\to\myenv\reconstruction-book\Scripts\activate.bat``

# Project organization

The project is organized into three packages:

### 1.  Algorithms
Contains all the reconstruction algorithms: ART, SIRT, MLEM, OSEM, FBP
### 2. Geometry
Contains all the classes to generate an experimental setup
### 3. Misc
Contains a number of loosely related classes, mostly used in the other packages.

Moreover the Project contains the folders:

* Data
  some data provided to be used within the project, e.g., a SheppLogan Phantom Image

* Notebook
  Jupyter notebook demonstrating how to use the classes of this project.
  Information on how to run Jupyter notebook can be found
   https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html
