from multiprocessing import Process, Lock
import itertools
import time
import os


def train(l, hyperparams):
    l.acquire()
    try:
        time.sleep(2.0)
        print(f'Done training {hyperparams}\n')
        print(f'module name: {__name__}')
        print(f'parent process: {os.getppid()}')
        print(f'process id: {os.getpid()}')
    finally:
        l.release()


# Grid search
grid = {
    'batch_size': [32, 64, 128, 1, 2],
    'learning_rate': [1E-4, 1E-3, 1E-2]
}


def main():
    grid_combs = []
    for values in itertools.product(*grid.values()):
        grid_combs.append(dict(zip(grid.keys(), values)))

    lock = Lock()
    for grid_comb in grid_combs:
        Process(target=train, args=(lock, grid_comb)).start()


if __name__ == "__main__":
    main()
