# Richter's Predictor: Modeling Earthquake Damage

## Overview

[Richter's Predictor]((https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/#features_list)) is a machine learning project aims to predict earthquake damage levels based on building characteristics and geographic data. 

This project utilizes data from the Nepal Earthquake competition hosted by DrivenData. The objective is to develop a predictive model that accurately estimates the damage level for buildings in the aftermath of an earthquake.

## Project Structure


MiniCompetition  
│  
├── src  
│   ├── modeling  
│   │   ├── [evaluate.py, model.py, pipeline.py, predict.py etc...]  
│   └── [config.py, dataset.py,features.py etc...] 
│  
├── data  
│   ├── [test_values.csv,train_values.csv, train_labels.csv]  
│
├── notebooks  
│   ├── [Exploration.ipynb etc...]  
│  
├── main.py  
├── requirements.txt  
└── README.md  
├── LICENSE  
└── folderStructure.ipynb  


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PundirNeetu/MiniCompetition.git
   cd MiniCompetition 
   ```
2. Install the required packages:
```
pip install -r requirements.txt
```	

## Usage
To run the main script and execute model training and evaluation:
```
python main.py
```	
## Data Preparation
Place  datasets in the /data directory.   
The expected files are:  

train_values.csv: Contains building features.  
train_labels.csv: Contains the corresponding damage grades.  
test_values.csv: Features for the test set.  


## Model Training and Evaluation
The project implements multiple machine learning models, including Random Forest and XGBoost. The evaluation metrics used include the F1 score to assess model performance.

## Acknowledgments
- DrivenData for organizing the competition.
- Scikit-learn for machine learning tools.
