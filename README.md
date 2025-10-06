# Water Quality Potability Prediction

A machine learning model that predicts whether water is safe for human consumption based on water quality parameters.

## Overview

This project uses logistic regression to classify water samples as potable (safe to drink) or non-potable based on nine water quality features. The model can help quickly assess water safety in areas where laboratory testing may be limited or delayed.

## Features

The model analyzes the following water quality parameters:

- **pH**: Acidity/alkalinity level of water
- **Hardness**: Calcium and magnesium concentration (mg/L)
- **Solids**: Total dissolved solids (ppm)
- **Chloramines**: Chloramine concentration (ppm)
- **Sulfate**: Sulfate concentration (mg/L)
- **Conductivity**: Electrical conductivity (μS/cm)
- **Organic_carbon**: Organic carbon concentration (ppm)
- **Trihalomethanes**: Trihalomethanes concentration (μg/L)
- **Turbidity**: Water clarity measure (NTU)

## Requirements

```
pandas
scikit-learn
numpy
```

Install dependencies:
```bash
pip install pandas scikit-learn numpy
```

## Dataset

The model is trained on the [Water Quality dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability), which contains water quality metrics and potability classification.

## Usage

### Training the Model

```python
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("water_quality_potability.csv")

# Define features and target
feature_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
x = data[feature_cols]
y = data['Potability']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
```

### Making Predictions

```python
import numpy as np

# Example water sample
new_sample = np.array([[7.0, 190, 22000, 8.0, 300, 400, 10, 65, 3.5]])

# Predict potability
prediction = model.predict(new_sample)

if prediction[0] == 1:
    print("The water is predicted to be POTABLE.")
else:
    print("The water is predicted to be NOT POTABLE.")
```

## Model Performance

The model achieves the following performance on the test set:

- **Accuracy**: [Add your accuracy score here]
- **Precision, Recall, F1-Score**: See classification report in output

## Project Structure

```
water-quality-prediction/
├── water_quality_model.py      # Main model training script
├── water_quality_potability.csv # Dataset
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Future Improvements

- Handle missing values in the dataset more robustly
- Experiment with other algorithms (Random Forest)
- Implement feature scaling for better performance
- Add cross-validation for more reliable evaluation
- Create a web interface for easy predictions

## Applications

This model could be useful for:

- **Developing regions**: Quick water safety assessment in areas like Papua New Guinea
- **Emergency response**: Rapid water quality evaluation after disasters
- **Rural communities**: Screening water sources before detailed lab testing
- **Water treatment facilities**: Real-time monitoring and quality control

## Author

Gideon Kulangye  
Data Science Student, South Dakota State University

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- Dataset from Kaggle Water Quality dataset
