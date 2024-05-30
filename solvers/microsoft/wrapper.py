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
# os.makedirs('tmp', exist_ok=True)

def construct_args(grid, filename):
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

    return [f"'{res_file}'", f"'{str(w)}'", f"'{str(h)}'", f"'{strs}'"]

def solve(tmp_dir, grids, filenames):
    tmp_dir = os.path.join(tmp_dir, '')  # add backslash if not there already
    num_puzzles = len(grids)
    assert len(filenames) == num_puzzles, f'Received {len(filenames)} filenames for {num_puzzles} grids'

    cmd = [f"'{tmp_dir}' '{num_puzzles}'"]
    files = []
    for i in range(num_puzzles):
        args = construct_args(grids[i], filenames[i])
        files.append(os.path.join(tmp_dir, args[0].strip("'")))
        cmd.extend(args)

    cmd = ' '.join(cmd)
    os.system(f'{str(binary)} {cmd}')

    artifacts = [] 
    for i, file in enumerate(files):    
        artifacts.append(f'{file}.html')
        artifacts.append(f'{file}.csv')

        np.save(f'{file}_soln.npy', grids[i])
        artifacts.append(f'{file}_soln.npy')

    
    return artifacts
    
# def solve(archive, num_puzzles, start=0):
#     archive = np.load(archive)
#     num_puzzles = len(archive.files) if num_puzzles == -1 else num_puzzles
#     args = [f"'{num_puzzles}'"]
#     files = []
#     for i in range(start, start + num_puzzles):
#         lst = construct_args(archive, i)
#         files.append('tmp/' + lst[0].strip("'"))
#         args.extend(lst)
#     args = ' '.join(args)

#     os.system(f'{str(binary)} {args}')

#     artifacts = []
#     for file in files:
#         artifacts.append(f'{file}.html')
#         artifacts.append(f'{file}.csv')
#     return artifacts