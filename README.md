# Experimentation and Evaluation in Machine Learning at University of Kassel FB16 - Personal Solutions

This repository contains personal solutions for the course *Experimentation and Evaluation in Machine Learning* (E2ML) of the *Intelligent Embedded Systems* (IES) department at the University of Kassel FB16 in the sommer term of 2023.

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
    - [`01_python_introduction.ipynb`](notebooks/01_python_introduction.ipynb): First Exercise - Introduction into Python including 
      important modules, such as, NumPy, SciPy, and Scikit-learn
   - [`02_foundations_of_stochastic.ipynb`](notebooks/02_foundations_of_stochastic.ipynb): Second Exercise - Foundations of Stochastic including 
      discete and continuous probabilities and probability rules
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

The experimentation package contains methods to design experiments. In particular, it contains methods to design
experiments with the following design types:

- Full Factorial Design
- Halton Design
- Latin Hypercube Design
- One-Factor-at-a-Time Design

The experimentation package also contains methods to perform optimizations with the following optimization methods:

- Bayesian Optimization

## Evaluation

The evaluation package contains methods to evaluate experimental results. In particular, it contains methods to evaluate
the following metrics:

- Accuracy
- Confusion Matrix
- Cohen's Kappa
- F1-Score
- Receiver Operating Characteristic (ROC) Curve
- Kullback-Leibler Divergence

The evaluation package also contains methods to visualize experimental results. In particular, it contains the following
visualization methods:

- Lift Chart
- ROC Curve
- Plot decision boundary of a classifier

The evaluation package further contains methods to perform error estimation with the following methods:

- Cross-Validation

It also contains loss functions of the following types:

- Zero-One Loss
- Binary Cross-Entropy Loss

Finally, it contains methods to perform statistical tests with the following methods:

- One-Sample Z-Test
- One-Sample T-Test
- Paired T-Test
- Wilcoxon Signed-Rank Test

## Results

The results of the experiments are stored in the `results` directory. The results are stored in the following format:

- `results`
   - `csvs`: CSV files of the experiments
      - `models`: Models of the experiments
         - `final`: Final models
            - `final_analysis[_traindata].csv`: Final analysis of the models [on the training data]
         - `up_to_week_x`: Models trained on the data up to week _x_
            - `z_models.csv`: Models with configuration _z_ on the data up to week _x_
            - `hpo_results.csv`: Hyperparameter optimization results of the models on the data up to week _x_
      - `worst_accuracies`: Worst accuracies of the experiments
         - `up_to_week_x`: Worst accuracies of the models trained on the data up to week _x_
   - `data_requests`: Data requests of the experiments
      - `data_req_period_x.csv`: Data requests of period _x_
   - `hpo_large_scale`: Results of the large-scale hyperparameter optimization
      - `y_best_params.npy`: Best parameters of the Classifier _y_
      - `y_hpo_results.csv`: Hyperparameter optimization results of the Classifier _y_
   - `plots`: Plots of the experiments
      - `data_requests`: Plots of data requests of the experiments
         - `data_req_period_x.png`: Plot of data requests of period _x_
      - `initial_data`: Plot of the initial data
      - `probability_heatmaps`: Probability heatmaps of the experiments
         - `up_to_week_x`: Probability heatmaps of the Classifiers on the data up to week _x_
            - `y_probabilities.png`: Probability heatmap of the Classifier _y_ on the data up to week _x_
      - `worst_values`: Worst values of the experiments
         - `up_to_week_x`: Worst values of the models trained on the data up to week _x_
            - `y_worst_values.png`: Worst values of the Classifier _y_ on the data up to week _x_

## Data

The data used in the experiments is stored in the `data` directory. The data is stored in the following format:

- `data`
   - `batchx_w_labels.csv`: Data of batch _x_ for user _w_ with labels
   - `initial_molluscs_data.csv`: Initial data of the molluscs data set

## License

For information about copyright and licensing, please refer to the [LICENSE](LICENSE) file in this repository.

## Contact

If you have any questions or suggestions about this repository, you can contact the repository maintainer at <philip.laskowicz@student.uni-kassel.de>.
