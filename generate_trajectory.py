import argparse
import pathlib
import os
import zipfile 
import tempfile
import shutil
import numpy as np
import time

from joblib import Parallel, delayed
from solvers.microsoft import wrapper as microsoft_wrapper

data_dir = pathlib.Path(__file__).parent / 'data'
print(f'saving trajectories to {data_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_archive', help='grid data .npz file')

    # if generating a subset of puzzles from archive
    # target indices or
    parser.add_argument('--puzzle_indices', type=str, default='', help='specify puzzle indices, e.g. 0,1,2')
    # randomly sample puzzles
    parser.add_argument('--sample_puzzles', type=int, default=0, help='randomly samples n puzzles from archive to generate trajectories')

    # for batched
    parser.add_argument('--max_workers', type=int, default=5)

    args = parser.parse_args()

    td = tempfile.TemporaryDirectory()
    tmp_dir = td.name

    archive = np.load(args.path_to_archive)
    target_files = np.array(archive.files)

    is_sample = len(args.puzzle_indices) > 0 or args.sample_puzzles > 0
    if is_sample: 
        if args.sample_puzzles > 0:
            indices = np.random.choice(np.arange(len(target_files)), args.sample_puzzles, replace=False)
        else:
            indices = (args.puzzle_indices + ',').split(',')[:-1]
            indices =  [int(x) for x in indices]
        target_files = target_files[indices]
        targets = [target_files] 
        names = ['sample'] 
    else:  # generate all trajectories, split train/val
        train_size = int(len(target_files) * 0.8)

        train_files = target_files[:train_size]
        val_files = target_files[train_size:] 

        targets = [train_files, val_files]
        names = ['train', 'val']

    for i in range(len(targets)): 
        target_files = targets[i] 
        target_grids = [archive[file] for file in target_files]
        # not really worth using parallel
        # if len(target_files) <= 50: 
        #     args.max_workers = 1

        index_batches = np.array_split(np.arange(len(target_files)), args.max_workers)

        start = time.perf_counter()
        results = Parallel(n_jobs=args.max_workers)(delayed(microsoft_wrapper.solve)(tmp_dir, [target_grids[j] for j in subbatch], target_files[subbatch])\
                                                     for batch in index_batches \
                                                        for subbatch in np.array_split(batch, 10))
        end = time.perf_counter() 

        artifacts = [x for y in results for x in y]
        traj_files = [x for x in artifacts if x.endswith('.csv')]

        data_src = 'logicgamesonline' if 'logicgamesonline' in artifacts[0] else 'janko'

        filename = f'{data_src}_trajectories_{names[i]}_expert720K.zip'
        print(f'Generated {len(traj_files)} trajectories for {data_src} in {end - start:.3f}s')

        with zipfile.ZipFile(os.path.join('data', filename), 'w') as f:
            for file in artifacts:
                f.write(file, arcname=os.path.basename(file))
        
    td.cleanup()

