import matplotlib.pyplot as plot


def visualize(data, kind, x, y):
    data.plot(kind=kind, x=x, y=y)
    plot.show()


def visualizeStandardization(data, column, xLabel, yLabel):
    plot.hist(data[column], bins=20)
    plot.xlabel(xLabel)
    plot.ylabel(yLabel)
    plot.show()
