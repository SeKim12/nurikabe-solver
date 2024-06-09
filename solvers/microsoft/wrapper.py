import os 
import pathlib
import numpy as np
import itertools 
import time 
from tqdm import tqdm 

cur_dir = pathlib.Path(__file__).parent

tmp_cwd = os.getcwd() 

# build binary if necessary
os.chdir(cur_dir)
os.system('make')
os.chdir(tmp_cwd)

# locate binary
binary = cur_dir / 'microsoft_solver'

order = {i: x for i, x in enumerate(itertools.permutations([0, 1, 2, 3, 4]))}

# trajectory data is stored in tmp directory
# os.makedirs('tmp', exist_ok=True)

def get_janko_pid(filename):
    return f'{int(filename.split("/")[-1].split(".a.html")[0]):05d}'

def get_logicgamesonline_pid(filename):
    return f'{int(filename.split("pid=")[-1]):05d}'

def construct_args(grid, filename):
    h, w = grid.shape
    grid = np.clip(grid, a_min=0, a_max=None)

    str_grid = np.char.mod('%d', grid)
    str_grid = np.where(str_grid == '0', ' ', str_grid)

    puzzle_str = [] 
    for row in str_grid:
        puzzle_str.append(''.join(list(row)))
    puzzle_str = '*'.join(puzzle_str)
    puzzle_str += '*'

    if 'logicgamesonline' in filename:
        puzzle_name = get_logicgamesonline_pid(filename)
    elif 'janko' in filename:
        puzzle_name = get_janko_pid(filename)

    return [f"'{puzzle_name}'", f"'{str(w)}'", f"'{str(h)}'", f"'{puzzle_str}'"]

def solve(save_dir, grids, filenames, worker_index=0):
    # tmp_dir = os.path.join(tmp_dir, '')  # add backslash if not there already
    if save_dir[-1] != '/':
        save_dir += '/'

    num_puzzles = len(grids)
    assert len(filenames) == num_puzzles, f'Received {len(filenames)} filenames for {num_puzzles} grids'

    # for i in range(num_puzzles):
    #     args = construct_args(grids[i], filenames[i])
    #     files.append(os.path.join(tmp_dir, args[0].strip("'")))
    #     cmd.extend(args)

    files = [] 

    act_grids = []
    act_filenames = []
    uid = None 
    for i in range(len(grids)):
        act_grids.append(grids[i][1])
        act_filenames.append(filenames[i][1])
        uid = grids[i][0]

    cmd = [f"'{x}'" for x in order[uid]]
    cmd.append(f"'{uid:05d}'")
    cmd.append(f"'{save_dir}'")
    cmd.append(f"'{num_puzzles}'")

    # breakpoint()

    for i in range(num_puzzles): 
        args = construct_args(act_grids[i], act_filenames[i]) 
        cmd.extend(args)
        
        puzzle_dir = os.path.join(save_dir, args[0].strip("'"))
        if uid == 0:
            os.makedirs(puzzle_dir, exist_ok=True)
        
        files.append(os.path.join(puzzle_dir, f'{uid:05d}.csv'))

    # for i in range(len(grids)):
    #     uid, grid = grids[i] 
    #     uid2, file = filenames[i] 

    #     assert uid == uid2, f'{uid} != {uid2}'

    #     cmd = [f"'{x}'" for x in order[uid]]

    #     cmd.append(f"'{uid:05d}'")
    #     cmd.append(f"'{save_dir}'")
    #     cmd.append(f'{num_puzzles}')

    cmd = ' '.join(cmd)
    os.system(f'{str(binary)} {cmd}')
    
    saved = set()
    for file, grid in zip(act_filenames, act_grids):
        soln_file = os.path.join(save_dir, get_logicgamesonline_pid(file), 'soln.npy')
        if soln_file in saved:
            continue 
        saved.add(soln_file)
        np.save(soln_file, grid)

    return files