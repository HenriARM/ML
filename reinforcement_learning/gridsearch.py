import subprocess
import itertools
import json

# Example of script name and hparams grid name
script_path = './run_flappy_dqn.py'
hparams_path = './flappy_dqn.json'

# Grid search
'''
Example of grid
grid = {
    'sequence_name': ['flappy_dqn'],
    'batch_size': [32, 64],
    'learning_rate': [1E-4, 1E-3]
}
'''

with open(hparams_path) as json_file:
    grid = json.load(json_file)


def main():
    for values in itertools.product(*grid.values()):
        params = []
        grid_comb = dict(zip(grid.keys(), values))
        for hparam in grid_comb:
            params.append(f'-{hparam}={grid_comb[hparam]}')
        p = subprocess.Popen(['python', script_path, *params])
        p.wait()


if __name__ == "__main__":
    main()
