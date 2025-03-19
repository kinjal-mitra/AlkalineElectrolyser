# AlkalineElectrolyser

This repository contains code and notebooks related to the development of a model for an Alkaline Electrolyser.

## Overview

The project focuses on using machine learning techniques, specifically Autoencoders and LSTM models, to analyze and potentially predict the behavior of an Alkaline Electrolyser.

## Contents

*   **`AutoEncoderModel_Notebook.ipynb`**: Jupyter Notebook containing the implementation and experimentation with Autoencoder models.
*   **`AutoEncodersModel.py`**: Python script containing the Autoencoder model.
*   **`LSTM_Model_Notebook.ipynb`**: Jupyter Notebook containing the implementation and experimentation with LSTM (Long Short-Term Memory) models.
*   **`Optuna Params_AutoEncoderModel.jpg`**: Image showing the loss for the parameters of the Autoencoder model optimized using Optuna.
*   **`Personlized Params_AutoEncoderModel.jpg`**: Image showing the loss for the personalized parameters of the Autoencoder model.
*   **`requirements.txt`**: File containing a list of Python packages required to run the code.

## Requirements

To run the code in this repository, you will need the following Python packages:

numpy
matplotlib
pandas
torch 
torchvision 
torchaudio --index-url https://download.pytorch.org/whl/cu124 (change cuda version on the basis of machine specs)
scikit-learn
optuna

The run in terminal:
``` 
pip install -r requirements.txt 

```


*(Note: A `requirements.txt` file is included in the repository. It's highly recommended to create a virtual environment before installing the requirements.)*

## Usage

1.  Clone the repository:

    ```
    git clone https://github.com/kinjal-mitra/AlkalineElectrolyser.git
    cd AlkalineElectrolyser
    ```

2.  Install the required packages:

    ```
    pip install -r requirements.txt
    ```

3.  Open and run the Jupyter Notebooks (`AutoEncoderModel_Notebook.ipynb` and `LSTM_Model_Notebook.ipynb`) to explore the models and their performance.  You can use Jupyter Lab or Jupyter Notebook.

    ```
    jupyter notebook AutoEncoderModel_Notebook.ipynb
    jupyter notebook LSTM_Model_Notebook.ipynb
    ```

    or

    ```
    jupyter lab AutoEncoderModel_Notebook.ipynb
    jupyter lab LSTM_Model_Notebook.ipynb
    ```

4.  The Python script `AutoEncodersModel.py` can be imported and used in other projects.

AlkalineElectrolyser/
├── AutoEncoderModel_Notebook.ipynb
├── AutoEncodersModel.py
├── LSTM_Model_Notebook.ipynb
├── Optuna Params_AutoEncoderModel.jpg
├── Personlized Params_AutoEncoderModel.jpg
└── requirements.txt
