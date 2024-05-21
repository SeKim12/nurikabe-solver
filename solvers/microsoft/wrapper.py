import os 
import pathlib
import numpy as np

cur_dir = pathlib.Path(__file__).parent

tmp_cwd = os.getcwd() 

# build binary if necessary
os.chdir(cur_dir)
os.system('make')
os.chdir(tmp_cwd)

# locate binary
binary = cur_dir / 'microsoft_solver'

# trajectory data is stored in tmp directory
os.makedirs('tmp', exist_ok=True)

def construct_args(archive, idx):
    file = archive.files[idx]
    grid = archive[file]

    h, w = grid.shape
    grid = np.clip(grid, a_min=0, a_max=None)

    str_grid = np.char.mod('%d', grid)
    str_grid = np.where(str_grid == '0', ' ', str_grid)

    strs = [] 
    for row in str_grid:
        strs.append(''.join(list(row)))
    strs = '*'.join(strs)
    strs += '*'

    if 'logicgamesonline' in file:
        res_file = f'logicgamesonline{int(file.split("pid=")[-1]):05d}'
    elif 'janko' in file:
        res_file = f'janko{int(file.split("/")[-1].split(".a.htm")[0]):05d}'

    return [f"'{res_file}'", f"'{str(w)}'", f"'{str(h)}'", f"'{strs}'"]

def solve(archive, num_puzzles, start=0):
    archive = np.load(archive)
    num_puzzles = len(archive.files) if num_puzzles == -1 else num_puzzles
    args = [f"'{num_puzzles}'"]
    files = []
    for i in range(start, start + num_puzzles):
        lst = construct_args(archive, i)
        files.append('tmp/' + lst[0].strip("'"))
        args.extend(lst)
    args = ' '.join(args)

    os.system(f'{str(binary)} {args}')

    artifacts = []
    for file in files:
        artifacts.append(f'{file}.html')
        artifacts.append(f'{file}.csv')
    return artifacts