import csv
import math
import numpy as np

def read_csv(filename = '../prob_plots.csv'):
    """
    Reads data from csv file, removes first line and gives back a
    numpy array with the data

    Arguments:
    - filename: name of csv file

    Returns:
    - numpy array of shape (row, columns)

    """

    data = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            data.append([number(x) for x in row])

    # delete
    del data[0]
    X = np.asarray(data)

    return X

def number(s):
    """
    Tries to cast s to float, if it fails it returns a string

    Arguments:
    - s: input scalar to cast

    Returns:
    - casted input
    """

    try:
        return float(s)
    except ValueError:
        return 0.0
