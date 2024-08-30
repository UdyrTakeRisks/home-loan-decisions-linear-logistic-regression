import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def CostFunction(x, y, theta):
    m = len(y)
    h = sigmoid(numpy.dot(x, theta))
    J = (-1 / m) * numpy.sum(y * numpy.log(h) + (1 - y) * numpy.log(1 - h))
    return J


global_theta = None


def fit_gradientDescent(x, y, alpha, iterations):  # works
    m = len(y)
    cost_history = []
    theta = numpy.zeros(x.shape[1])
    for _ in range(iterations):
        h = sigmoid(numpy.dot(x, theta))
        gradient = numpy.dot(x.T, (h - y)) / m
        theta -= alpha * gradient
        cost = CostFunction(x, y, theta)
        cost_history.append(cost)  # ??
    return theta, cost_history


# def fit_gradientDescent(x, y, alpha, max_iterations): # works
#     # Initialize theta array
#     theta = numpy.zeros(x.shape[1])
#
#     # Loop for gradient descent iterations
#     for _ in range(max_iterations):
#         # Compute the hypothesis function values
#         h = sigmoid(numpy.dot(x, theta))
#
#         # Loop to update each parameter
#         for j in range(theta.shape[0]):
#             # Compute the partial derivative of the error
#             par_der = (1 / len(y)) * numpy.sum((h - y) * x[:, j])
#
#             # Update the j-th parameter
#             theta[j] = theta[j] - alpha * par_der
#
#     return theta


def predictLogistic(test_set, theta, threshold=0.5):
    test_set = numpy.array(test_set)
    probability = sigmoid(test_set.dot(theta))
    prediction = (probability >= threshold).astype(int)
    return prediction


def predictExecutionLogistic(test_set):
    result = predictLogistic(test_set.astype(int), global_theta)
    return result


def accuracy(Y_true, Y_predicted):
    return numpy.mean(Y_true == Y_predicted)


# Loan_Status
# gradientDescent(X_train, Y_train['Loan_Status']) == fit
# predict
# accuracy(Y_test['Loan_Status'], Y_pred)

def LogisticRegression(x_train, x_test, y_train, learning_rate, iterations):
    global global_theta
    global_theta, _ = fit_gradientDescent(x_train, y_train, learning_rate, iterations)
    y_pred = predictLogistic(x_test.astype(int), global_theta)
    return y_pred
