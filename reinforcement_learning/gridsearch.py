import subprocess
import itertools

script_name = './reinforcement_learning/ple_dqn'

# Grid search
grid = {
    'sequence_name': ['flappy_dqn'],
    'batch_size': [32, 64],
    'learning_rate': [1E-4, 1E-3]
}


def main():
    for values in itertools.product(*grid.values()):
        params = []
        grid_comb = dict(zip(grid.keys(), values))
        for hparam in grid_comb:
            params.append(f'-{hparam}={grid_comb[hparam]}')
        p = subprocess.Popen(['python', f'{script_name}.py', *params])
        p.wait()


if __name__ == "__main__":
    main()
