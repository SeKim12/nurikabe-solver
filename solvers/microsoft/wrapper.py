import os 
import pathlib
import numpy as np
import itertools 
import time 

cur_dir = pathlib.Path(__file__).parent

tmp_cwd = os.getcwd() 

# build binary if necessary
os.chdir(cur_dir)
os.system('make')
os.chdir(tmp_cwd)

# locate binary
binary = cur_dir / 'microsoft_solver'

# trajectory data is stored in tmp directory
# os.makedirs('tmp', exist_ok=True)

def construct_args(grid, filename, uid=None):
    h, w = grid.shape
    grid = np.clip(grid, a_min=0, a_max=None)

    str_grid = np.char.mod('%d', grid)
    str_grid = np.where(str_grid == '0', ' ', str_grid)

    strs = [] 
    for row in str_grid:
        strs.append(''.join(list(row)))
    strs = '*'.join(strs)
    strs += '*'

    if 'logicgamesonline' in filename:
        res_file = f'logicgamesonline{int(filename.split("pid=")[-1]):05d}'
    elif 'janko' in filename:
        res_file = f'janko{int(filename.split("/")[-1].split(".a.htm")[0]):05d}'

    if uid is not None:
        res_file = res_file + f'_{uid:05d}'

    return [f"'{res_file}'", f"'{str(w)}'", f"'{str(h)}'", f"'{strs}'"]

def solve(tmp_dir, grids, filenames):
    tmp_dir = os.path.join(tmp_dir, '')  # add backslash if not there already
    num_puzzles = len(grids)
    assert len(filenames) == num_puzzles, f'Received {len(filenames)} filenames for {num_puzzles} grids'

    # for i in range(num_puzzles):
    #     args = construct_args(grids[i], filenames[i])
    #     files.append(os.path.join(tmp_dir, args[0].strip("'")))
    #     cmd.extend(args)

    files = [] 
    for uid, check_order in enumerate(itertools.permutations([0, 1, 2, 3, 4])):
        cmd = [f"'{x}'" for x in check_order]
        cmd.append(f"'{tmp_dir}'")
        cmd.append(f'{num_puzzles}')

        for i in range(num_puzzles): 
            args = construct_args(grids[i], filenames[i], uid) 
            files.append(os.path.join(tmp_dir, args[0].strip("'")))
            print(f'{files[-1].split("/")[-1]} STARTING...')
            cmd.extend(args)

        cmd = ' '.join(cmd)
        os.system(f'{str(binary)} {cmd}')

    # cmd = [f"'{tmp_dir}' '{num_puzzles}'"]
    # files = []
    # for i in range(num_puzzles):
    #     args = construct_args(grids[i], filenames[i])
    #     files.append(os.path.join(tmp_dir, args[0].strip("'")))
    #     cmd.extend(args)

    # cmd = ' '.join(cmd)
    # os.system(f'{str(binary)} {cmd}')

    artifacts = [] 
    for i, file in enumerate(files):  
        print(f'{file.split("/")[-1]} DONE')
        artifacts.append(f'{file}.html')
        artifacts.append(f'{file}.csv')

        np.save(f'{file}_soln.npy', grids[i // 120])
        artifacts.append(f'{file}_soln.npy')

    return artifacts