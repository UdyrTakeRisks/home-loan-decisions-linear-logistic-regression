import pandas
import numpy
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from analysis import visualize, visualizeStandardization
from standardize import standardizeOldData
from logisticRegression import LogisticRegression, accuracy

# Load the old dataset
data = pandas.read_csv('loan_old.csv')

# Check whether there are missing values, types of each feature, feature scale and visualization
print(data)

# Perform analysis on the dataset

visualize(data, 'hist', 'Income', 'Max_Loan_Amount')

visualize(data, 'scatter', 'Income', 'Max_Loan_Amount')

visualize(data, 'scatter', 'Coapplicant_Income', 'Max_Loan_Amount')

visualize(data, 'scatter', 'Loan_Tenor', 'Max_Loan_Amount')

visualize(data, 'scatter', 'Credit_History', 'Max_Loan_Amount')

# Data Preprocessing
# Remove missing values
for index, row in data.iterrows():
    if row.isnull().values.any():
        data = data.drop(index)

print("\nAfter Removing missing values\n")

print(data)

visualize(data, 'scatter', 'Dependents', 'Max_Loan_Amount')

# Replace '3+' with '3' in the 'Dependents' column, 3+ is it 3 or more dependents ??
data['Dependents'] = data['Dependents'].str.replace('3+', '3')

# Print the modified dataframe
# print("\nAfter Removing + symbol in Dependents\n")

# for index, row in data.iterrows():
#     print(row)

# print(data)

# separate features and targets
X = data.drop(columns=['Max_Loan_Amount', 'Loan_Status'])
print(X)
# X = X.to_numpy().reshape((-1, 10))
# print(X)

# Y = data[['Max_Loan_Amount', 'Loan_Status']].to_numpy()  # what about loan_status yes or no ?
Y = data[['Max_Loan_Amount', 'Loan_Status']]  # what about loan_status yes or no ?
print(Y)

# split - 80% train sets - 20% test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

print("x train:\n", X_train)
print("x test:\n", X_test)
print("y train:\n", Y_train)
print("y test:\n", Y_test)

# encode categorical features
le = LabelEncoder()
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_test['Gender'] = le.transform(X_test['Gender'])
X_train['Married'] = le.fit_transform(X_train['Married'])
X_test['Married'] = le.transform(X_test['Married'])
X_train['Education'] = le.fit_transform(X_train['Education'])
X_test['Education'] = le.transform(X_test['Education'])
X_train['Property_Area'] = le.fit_transform(X_train['Property_Area'])
X_test['Property_Area'] = le.transform(X_test['Property_Area'])

# encode categorical targets
Y_train['Loan_Status'] = le.fit_transform(Y_train['Loan_Status'])
Y_test['Loan_Status'] = le.transform(Y_test['Loan_Status'])

print("x train:\n", X_train)
print("x test:\n", X_test)
print("y train:\n", Y_train)
print("y test:\n", Y_test)

# property area columns after encoding for example -> Urban (0) - Semi-Urban (1) - Rural (2)

# standardized numerical features

############
# error bec of '3+'
# Dependents standardize
# mean = numpy.mean(X_train['Dependents'])
# std = numpy.std(X_train['Dependents'])
# standardize train and test sets
# X_train['Dependents'] = standardizeOldData('Dependents', X_train, mean, std)
# X_test['Dependents'] = standardizeOldData('Dependents', X_test, mean, std)
# test
# visualizeStandardization(X_test, 'Dependents', xLabel='Standardized Dependents', yLabel='Frequency')
#############

# calc mean and standard deviation on the train set

# Income standardize
mean = numpy.mean(X_train['Income'])
std = numpy.std(X_train['Income'])
X_train['Income'] = standardizeOldData('Income', X_train, mean, std)
X_test['Income'] = standardizeOldData('Income', X_test, mean, std)
# test
visualizeStandardization(X_test, 'Income', xLabel='Standardized Income', yLabel='Frequency')

# Coapplicant_Income standardize
mean = numpy.mean(X_train['Coapplicant_Income'])
std = numpy.std(X_train['Coapplicant_Income'])
X_train['Coapplicant_Income'] = standardizeOldData('Coapplicant_Income', X_train, mean, std)
X_test['Coapplicant_Income'] = standardizeOldData('Coapplicant_Income', X_test, mean, std)
# test
visualizeStandardization(X_test, 'Coapplicant_Income', xLabel='Standardized Coapplicant_Income', yLabel='Frequency')

# Loan_Tenor standardized
mean = numpy.mean(X_train['Loan_Tenor'])
std = numpy.std(X_train['Loan_Tenor'])
X_train['Loan_Tenor'] = standardizeOldData('Loan_Tenor', X_train, mean, std)
X_test['Loan_Tenor'] = standardizeOldData('Loan_Tenor', X_test, mean, std)
# test
visualizeStandardization(X_test, 'Loan_Tenor', xLabel='Standardized Loan_Tenor', yLabel='Frequency')

# Credit_History standardized
mean = numpy.mean(X_train['Credit_History'])
std = numpy.std(X_train['Credit_History'])
X_train['Credit_History'] = standardizeOldData('Credit_History', X_train, mean, std)
X_test['Credit_History'] = standardizeOldData('Credit_History', X_test, mean, std)
# test
visualizeStandardization(X_test, 'Credit_History', xLabel='Standardized Credit_History', yLabel='Frequency')

# take the encoded categorical features/targets and standardized numerical features to fit the model

# Fitting Linear Regression Model to predict the loan amount (continuous)

linearModel = linear_model.LinearRegression()
# ValueError: could not convert string to float: 'LP002894'
X_train = X_train.drop(columns=['Loan_ID'])
X_test = X_test.drop(columns=['Loan_ID'])
linearModel.fit(X_train, Y_train['Max_Loan_Amount'])

Y_pred = linearModel.predict(X_test)
# predicted_loan_amounts = Y_pred[:, 0]
# print("Predicted loan amounts: \n", predicted_loan_amounts)
print("Predicted loan amounts: \n", Y_pred)

# print('Coefficients: \n', linearModel.coef_, " ", linearModel.intercept_)
# print('Mean squared error: %.2f' % mean_squared_error(Y_test, Y_pred))

# Evaluate linear regression model
r2 = r2_score(Y_test['Max_Loan_Amount'], Y_pred)
print("Linear Regression R2 Score: ", r2)

# fitting a binary logistic regression model (build it from scratch) to predict loan status (discrete)

X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)
# Y_train[:, 1] -> ignore rows, take 2nd column Loan_Status, note: Loan_ID is dropped

Y_prediction = LogisticRegression(X_train.astype(int), X_test, Y_train[:, 1].astype(int), 0.01, 30000)
print("Predicted loan statuses: \n", Y_prediction)

acc = accuracy(Y_test['Loan_Status'], Y_prediction)
print('Logistic Regression Accuracy: ', acc)
