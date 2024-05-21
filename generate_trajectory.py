import argparse
import pathlib
import os
import zipfile 

from solvers.microsoft import wrapper as microsoft_wrapper
from render import render_trajectory

data_dir = pathlib.Path(__file__).parent / 'data'
print(f'saving trajectories to {data_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', choices=['microsoft'])
    parser.add_argument('--path_to_archive', help='grid data .npz file')
    parser.add_argument('--num_puzzles', type=int, default=-1)
    parser.add_argument('--generate_videos', type=int, default=0)

    args = parser.parse_args()

    if args.solver == 'microsoft':
        artifacts = microsoft_wrapper.solve(args.path_to_archive, args.num_puzzles)
    
    artifacts = [os.path.join(os.getcwd(), x) for x in artifacts]
    csv_files = [x for x in artifacts if x.endswith('.csv')]

    print(f'Generated {len(csv_files)} trajectories')
    print(f'Rendering {args.generate_videos} sample videos of trajectories')
    for i in range(min(len(csv_files), args.generate_videos)):
        render_trajectory(csv_files[i])
    
    data_src = 'logicgamesonline' if 'logicgamesonline' in artifacts[0] else 'janko'
    
    print(f'Compressing trajectory data')

    with zipfile.ZipFile(os.path.join('data', f'{args.solver}_{data_src}_trajectories.zip'), 'w') as f:
        for file in artifacts:
            f.write(file, arcname=os.path.basename(file))
    
    for file in artifacts:
        os.remove(file)


    

