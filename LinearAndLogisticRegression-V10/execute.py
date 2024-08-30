import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from main import linearModel
from standardize import standardizeNewData
from logisticRegression import predictExecutionLogistic

# Load the new dataset
features = pandas.read_csv('loan_new.csv')

# print(features)

# Remove missing values
for index, row in features.iterrows():
    if row.isnull().values.any():
        features = features.drop(index)

# Replace '3+' with '3' in the 'Dependents' column, 3+ is it 3 or more dependents ??
features['Dependents'] = features['Dependents'].str.replace('3+', '3')

# print("\nAfter Removing missing values\n")

# print(features)

# encode categorical features
le = LabelEncoder()
features['Gender'] = le.fit_transform(features['Gender'])
features['Married'] = le.fit_transform(features['Married'])
features['Education'] = le.fit_transform(features['Education'])
features['Property_Area'] = le.fit_transform(features['Property_Area'])

# print('data after encoding:\n', features)

# standardize numerical features
# features['Dependents'] = standardize('Dependents', features)
# do not use the same function, as mean and std should be calc on features
features['Income'] = standardizeNewData('Income', features)
features['Coapplicant_Income'] = standardizeNewData('Coapplicant_Income', features)
features['Loan_Tenor'] = standardizeNewData('Loan_Tenor', features)
features['Credit_History'] = standardizeNewData('Credit_History', features)

features = features.drop(columns=['Loan_ID'])
# predict loan amounts
targetLoanAmounts = linearModel.predict(features)

print("Predicted Loan Amounts from the new dataset: \n", targetLoanAmounts)

# predict loan statuses

targetLoanStatus = predictExecutionLogistic(features.astype(int))

print("Predicted Loan Statuses from the new dataset: \n", targetLoanStatus)

df = pandas.DataFrame({
    'Predicted Loan Amounts': targetLoanAmounts,
    'Predicted Loan Statuses': targetLoanStatus
})

df.to_csv('predicted_results.csv', index=False)
