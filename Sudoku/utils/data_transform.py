import numpy as np
import pandas as pd

# from .utils.utils import PATH_TO_CSV

DATASET_PATH = '../assets/nurikabe.npz'

def custom_encoder(array_of_grids):
    """ transform numpy array of sudoku grids in one hot
    array of dimension (len(arr), 9, 9, 9), with one channel
    for each value from 1 to 9. """

    if len(array_of_grids.shape) == 2:
        array_of_grids = np.array([array_of_grids])
    shape_encoded = (*array_of_grids.shape, 3)
    encoded = np.zeros(shape_encoded, dtype=np.bool)
    for i, grid in enumerate(array_of_grids):
        for r in range(9):
            for c in range(9):
                if grid[r, c] == -2:
                    encoded[i, r, c, 2] = 1
                elif grid[r, c] == -1:
                    encoded[i, r, c, 1] = 1
                else:
                    encoded[i, r, c, 0] = 1
        # for value in range(0, 3):
        #     encoded[i, :, :, value] = (np.abs(grid) == value)
    return encoded


def read_transform(NROWS=100, encode=False):
    """ If encode is true, also splits in train test and eval. """

    all_solutions = np.load(DATASET_PATH)

    train = [i for i in range( int(len(all_solutions) * 0.9) )]
    val = [i for i in range( int(len(all_solutions) * 0.9), int(len(all_solutions) * 0.99) )]
    test = [i for i in range( int(len(all_solutions) * 0.99), int(len(all_solutions)) )]

    # train = [i for i in range( int(len(all_solutions) * 0.7) )]
    # val = [i for i in range( int(len(all_solutions) * 0.7), int(len(all_solutions) * 0.85) )]
    # test = [i for i in range( int(len(all_solutions) * 0.85), int(len(all_solutions)) ) ]

    _Y_train = [all_solutions[all_solutions.files[i]] for i in train]
    _X_train = [
        np.where(
            (board == 0) | (board == -1), -2, board
        )
        for board in _Y_train
    ]

    _Y_val = [all_solutions[all_solutions.files[i]] for i in train]
    _X_val = [
        np.where(
            (board == 0) | (board == -1), -2, board
        )
        for board in _Y_val
    ]

    _Y_test = [all_solutions[all_solutions.files[i]] for i in val]
    _X_test = [
        np.where(
            (board == 0) | (board == -1), -2, board
        )
        for board in _Y_test
    ]

    Y_train = [custom_encoder(data) for data in _Y_train]
    # breakpoint()
    Y_train = np.stack(Y_train).squeeze()

    X_train = [custom_encoder(data) for data in _X_train]
    X_train = np.stack(X_train).squeeze()

    Y_val = [custom_encoder(data) for data in _Y_val]
    Y_val = np.stack(Y_val).squeeze()

    X_val = [custom_encoder(data) for data in _X_val]
    X_val = np.stack(X_val).squeeze()

    Y_test = [custom_encoder(data) for data in _Y_test]
    Y_test = np.stack(Y_test).squeeze()

    X_test = [custom_encoder(data) for data in _X_test]
    X_test = np.stack(X_test).squeeze()

    # _data_X = data["puzzle"].apply(lambda x: [int(i) if i != '.'
    #                                           else 0 for i in x])
    # _data_Y = data["solution"].apply(lambda x: [int(i) for i in x])

    # data_X = np.stack(_data_X.to_numpy()).reshape((len(data), 9, 9))
    # data_Y = np.stack(_data_Y.to_numpy()).reshape((len(data), 9, 9))

    # if not encode:
    #     return data_X, data_Y

    # data_X_encoded = custom_encoder(data_X)
    # data_Y_encoded = custom_encoder(data_Y)

    # _X_train, X_test, _Y_train, Y_test = train_test_split(
    #     data_X_encoded, data_Y_encoded, test_size=0.1, random_state=42)

    # X_train, X_val, Y_train, Y_val = train_test_split(
    #     _X_train, _Y_train, test_size=0.1, random_state=42)
    # breakpoint()
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
