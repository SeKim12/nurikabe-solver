import csv 
import os
import argparse

from moviepy.video.VideoClip import DataVideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import matplotlib.pyplot as plt 

def load_trajectory(path_to_csv):
    arr = []
    with open(path_to_csv) as file:
        reader = csv.reader(file)
        w, h = 0, 0
        for i, row in enumerate(reader):
            row = list(int(x) for x in row)
            # this indicates the number of cells solved
            if len(row) == 1:
                continue
            if i == 0:
                w, h = row
            else:
                arr.append(np.array(row).reshape((h, w)))
    return np.array(arr)

def data_to_frame(board):
    normalized = np.copy(board)
    normalized[board > 0] = 1
    normalized[board == -3] = 0.5
    normalized[board == -2] = 1
    normalized[board == -1] = 0

    map = plt.imshow(normalized, cmap='gray')

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] > 0:
                map.axes.text(j, i, int(board[i, j]), ha='center', va='center', color='black', fontdict={'size': 16})

    map.axes.axis('off')

    return mplfig_to_npimage(map.figure)

def render_trajectory(path_to_csv, save_dir='data'):
    vidname = path_to_csv.split('/')[-1].split('.csv')[0]

    arr = load_trajectory(path_to_csv)
    clip = DataVideoClip(arr, data_to_frame, fps=15)
    clip.write_videofile(os.path.join(save_dir, f'{vidname}.mp4'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_csv') 
    parser.add_argument('--save_dir', type=str, default='data')
    args = parser.parse_args()

    render_trajectory(args.path_to_csv, args.save_dir)
