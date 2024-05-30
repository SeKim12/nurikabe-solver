import csv 
import os
import argparse
import zipfile 
import tempfile
import gc

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
            # this indicates the number of cells solved
            if len(row) == 1 or 'seconds' in row[-1]:
                continue
            row = [int(x) for x in row]
            if i == 0:
                w, h = row
            else:
                arr.append(np.array(row).reshape((h, w)))
    return np.array(arr)

def data_to_frame(board):
    normalized = np.copy(board).astype(float)
    normalized[board > 0] = 1
    normalized[board == -2] = 0.5
    normalized[board == -1] = 0
    normalized[board == 0] = 1
    
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
    clip = DataVideoClip(arr, data_to_frame, fps=5)
    clip.write_videofile(os.path.join(save_dir, f'{vidname}.mp4'))
    clip.close()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_file', type=str) 
    parser.add_argument('--num_videos', '-n',  default=5, type=int)
    parser.add_argument('--save_dir', type=str, default='data')
    args = parser.parse_args()

    if args.path_to_file.endswith('.zip'):
        zf = zipfile.ZipFile(args.path_to_file)

        with tempfile.TemporaryDirectory() as td: 
            zf.extractall(td) 
            csv_files = [x for x in os.listdir(td) if x.endswith('.csv')]

            for i in range(args.num_videos):
                render_trajectory(os.path.join(td, csv_files[i]), args.save_dir)
    else:
        render_trajectory(args.path_to_file, args.save_dir)
