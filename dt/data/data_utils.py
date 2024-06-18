import csv
import numpy as np


def load_csv_trajectory(path_to_csv):
    arr = []
    with open(path_to_csv) as file:
        reader = csv.reader(file)
        w, h = 0, 0
        for i, row in enumerate(reader):
            # this indicates the number of cells solved
            if "seconds" in row[-1]:
                continue
            row = [int(x) for x in row]
            if i == 0:
                w, h = row
            else:
                arr.append(np.array(row).reshape((h, w)))
    return np.array(arr)


def write_csv_trajectory(fp, trajectory, aux={}):
    h = aux.get("h", 9)
    w = aux.get("w", 9)
    cells_solved = aux.get("cells_solved", "-1")
    time = aux.get("time", "seconds")
    with open(fp, "w") as f:
        f.write(f"{h},{w}\n")
        for t in range(len(trajectory)):
            f.write(",".join([f"{x}" for x in trajectory[t].flatten()]) + "\n")
        f.write(f"{cells_solved},{time}\n")


def action_to_index_fixed(H, W, A, y, x, a):
    return y * W * A + x * A + a


def index_to_action_fixed(H, W, A, a):
    y = a // (W * A)
    x = (a - y * (W * A)) // A
    a_ = a - y * (W * A) - (x * A)
    return y, x, -a_
