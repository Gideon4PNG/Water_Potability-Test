#Working with multiple regression model

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
#Load the dataset
data=pd.read_csv(r"C:\Users\UOTSTD489\Desktop\DS and ML projects via Python\Datasets\water_quality_potability.csv")

print(data[:0])

#List of feature columns
feature_cols=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

#Create data matric x and target vector y 
x=data[feature_cols]
y=data['Potability']

#Split the dataset into training and testing sets
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.2,ransom_state=42)


#Initialize the fit logistic regression model

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#Predict on the test set 
y_pred=model.predict(x_test)


#Evaluate the model
print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))