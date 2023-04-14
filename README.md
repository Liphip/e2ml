# Experimentation and Evaluation in Machine Learning at University of Kassel FB16 - Personal Solutions

This repository contains personal solutions for the course *Experimentation and Evaluation in Machine Learning* (E2ML) of the 'Intelligent Embedded Systems' (IES) department at the University of Kassel FB16 in the sommer term of 2023.

### Table of Contents

- [General](#general)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Experimentation](#experimentation)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## General

**Author**: Philip Laskowicz

**E-Mail**: <philip.laskowicz@student.uni-kassel.de>

**Institute**: Intelligent Embedded Systems, University of Kassel, Germany


For more information about the E2ML course, visit the [course website](https://www.uni-kassel.de/eecs/ies/lehre/sommersemester-2023), the [Moodle course](https://moodle.uni-kassel.de/course/view.php?id=8349) or the [course catalog page](https://portal.uni-kassel.de/qisserver/rds?state=verpublish&status=init&vmfile=no&publishid=219889&moduleCall=webInfo&publishConfFile=webInfo&publishSubDir=veranstaltung&noDBAction=y&init=y).

These solutions are for reference only and should not be copied or submitted as one's own work. Please refer to the [License](#license) section for more information. If you have questions about the solutions or suggestions for improvement, please see the [Contact](#contact) section for information on how to get in touch with the repository maintainer.


## Project Structure

- `e2ml`: Python package of the Python modules implemented during this course
    - `evaluation`: Python package to evaluate and visualize experimental results
    - `experimentation`: Python package with methods of design of experiments
    - `models`: Python package of implement machine learning models
    - `preprocessing`: Python package of data preprocessing functions
    - `simulation`: Python package to execute experiments
- `notebooks`: directory of Jupyter notebooks with example code
    - `tutorial_python_solutation.ipynb`: introduction into Python including 
      important modules, such as, NumPy, SciPy, and Scikit-learn
- `LICENSE`: information about the terms under which one can use this package
- `setup.py`: Python file to install the project's package

## Setup

To install and use this project, one needs to consider the following steps.

1. Update the general section of `README.md` and the `setup.py` file by adding your credentials to the designated
text passages.
2. Install conda for Python 3.9 according to the 
   [installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
3. Create a conda environment with the name `e2ml-env`.
```shell
conda create --name e2ml-env python=3.9
```
4. Activate the created environment.
```shell
conda activate e2ml-env
```
5. Install the project's package `e2ml` in the conda environment.
```shell
pip install -e .
```
6. Now, you have installed the Python package `e2ml` and should
be able to use it. You can test it by importing it within a Python console.
```python
import e2ml
```
7. Finally, you can start to work with this project. In particular, you can view the 
   [Jupyter Notebooks](https://jupyter-notebook.readthedocs.io/en/stable/) in the folder `notebooks`
   by executing the following command.
```shell
jupyter-notebook
```
## Experimentation

TBD

## Evaluation

TBD

## Results

TBD


## License

For information about copyright and licensing, please refer to the [LICENSE](LICENSE) file in this repository.

## Contact

If you have any questions or suggestions about this repository, you can contact the repository maintainer at <philip.laskowicz@student.uni-kassel.de>.