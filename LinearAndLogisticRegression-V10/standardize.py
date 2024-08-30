import numpy


def standardizeOldData(column, data, mean, std):
    # standardize
    return (data[column] - mean) / std


def calculateNewMean(features, column):
    mean = numpy.mean(features[column])
    return mean


def calculateNewStd(features, column):
    std = numpy.std(features[column])
    return std


def standardizeNewData(column, data):
    # standardize
    return (data[column] - calculateNewMean(data, column)) / calculateNewStd(data, column)
