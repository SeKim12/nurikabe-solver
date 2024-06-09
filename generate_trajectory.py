import argparse
import pathlib
import os
import zipfile 
import tempfile
import shutil
import numpy as np
import itertools 
import time
import random 
import contextlib
import joblib 
from tqdm import tqdm 

from joblib import Parallel, delayed
from solvers.microsoft import wrapper as microsoft_wrapper

data_dir = pathlib.Path(__file__).parent / 'data'
print(f'saving trajectories to {data_dir}')

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_archive', help='grid data .npz file')
    parser.add_argument('--save_dir', type=str)
    # if generating a subset of puzzles from archive
    # target indices or
    parser.add_argument('--puzzle_indices', type=str, default='', help='specify puzzle indices, e.g. 0,1,2')
    # randomly sample puzzles
    parser.add_argument('--sample_puzzles', type=int, default=0, help='randomly samples n puzzles from archive to generate trajectories')

    # for batched
    parser.add_argument('--max_workers', type=int, default=5)

    args = parser.parse_args()

    # td = tempfile.TemporaryDirectory()
    # os.makedirs(args.save_dir, exist_ok=True)

    archive = np.load(args.path_to_archive)

    # filter out the zero puzzle
    target_files = np.array([x for x in archive.files if '5336' not in x])

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
        save_dir = f'{args.save_dir}_{names[i]}'
        print(save_dir)
        target_files = targets[i] 
        target_grids = np.array([archive[file] for file in target_files])

        all_files = []
        all_grids = [] 

        for grid, file in zip(target_grids, target_files):
            for uid, check_order in enumerate(itertools.permutations([0, 1, 2, 3, 4])):
                all_files.append((uid, file))
                all_grids.append((uid, grid))

        all_files = sorted(all_files, key=lambda x: x[0])
        all_grids = sorted(all_grids, key=lambda x: x[0])

        index_batches = np.array_split(np.arange(len(all_files)), 120)

        print('Starting!')

        parallel_args = []
        for uid, batch in enumerate(index_batches):
            for subbatch in np.array_split(batch, args.max_workers):
                grids = []
                files = [] 
                
                for k in subbatch:
                    grids.append(all_grids[k])
                    files.append(all_files[k])

                    assert all_files[k][0] == all_grids[k][0] == uid
                
                parallel_args.append((save_dir, grids, files))

        start = time.perf_counter()

        all_args = parallel_args if names[i] == 'val' else parallel_args[-10:]
        with tqdm_joblib(tqdm(desc="My calculation", total=len(parallel_args))) as progress_bar:
            results = Parallel(n_jobs=args.max_workers)(
                delayed(microsoft_wrapper.solve)(*(_arg + (j%args.max_workers,))) for j, _arg in enumerate(all_args)
            )
                                                #  i) for i, batch in enumerate(index_batches))

        end = time.perf_counter() 

        artifacts = [x for y in results for x in y]
        traj_files = [x for x in artifacts if x.endswith('.csv')]

        data_src = 'logicgamesonline' # 

        # filename = f'{data_src}_trajectories_{names[i]}_expert720K.zip'
        print(f'Generated {len(traj_files)} trajectories for {data_src} in {end - start:.3f}s')

        # with zipfile.ZipFile(os.path.join('data', filename), 'w') as f:
        #     for file in artifacts:
        #         f.write(file, arcname=os.path.basename(file))

