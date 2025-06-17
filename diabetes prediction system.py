# IMPORTING DEPENDENCIES

import pandas as pd
import numpy as np
import time, os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# DATA COLLECTION AND PROCESSING

# loading the data into pandas dataframe
diabetes_df = pd.read_csv("diabetes.csv")
# Splitting the data into training and testing data
X = diabetes_df.drop(columns="Outcome", axis=1)  #separating the labels
Y = diabetes_df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=7)

# FEATURE SCALING
scaler = StandardScaler()

scaler.fit(X_train)
X_train_standardized = scaler.transform(X_train)
X_test_standardized = scaler.transform(X_test)

# MODEL EVALUATION
model = LogisticRegression()

model.fit(X_train_standardized, Y_train) #Training the model
training_prediction = model.predict(X_train_standardized)
training_accuracy = accuracy_score(training_prediction, Y_train) * 100 #accuracy score on training data

testing_prediction = model.predict(X_test_standardized)
testing_accuracy = accuracy_score(testing_prediction, Y_test) * 100 #accuracy score on testing data

# PREDICTIVE SYSTEM
def predict(pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age):
    input_data = np.asarray([pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,
    age]).reshape(1, -1) #turning input data into 2d shape

    scaled_input = scaler.transform(input_data) #scaling the input

    prediction = model.predict(scaled_input)

    if prediction[0]==0:
        return 'The person is non-diabetic'
    else:
        return 'The person is diabetic'

def program_intro():
    os.system('clear')
    print("""
    1. Predict Diabetes
    2. Accuracy Score
    """)
    option = input('Enter a valid option: ')
    if option == '1':
        print('\nEnter the required inputs')
        try:
            pregnancies = float(input("Pregnancies (if none,enter 0): "))
            glucose = float(input("Glucose: "))
            blood_pressure = float(input("Blood Pressure: "))
            skin_thickness = float(input("Skin Thickness: "))
            insulin = float(input("Insulin: "))
            bmi = float(input("BMI: "))
            diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
            age = float(input("Age: "))
            prediction = predict(pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age)
            print('Calculating.....')
            time.sleep(2)
            print(prediction)
        except:
            print('Enter Valid Numbers!')
    elif option == '2':
        print('\n')
        print(f"Accuracy on training data : {training_accuracy:.2f}%")
        print(f"Accuracy on testing data : {testing_accuracy:.2f}%")
    else:
        print('Invalid Option!')

program_intro()