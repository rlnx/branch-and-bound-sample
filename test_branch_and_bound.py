import os
import math
import time
import argparse
import numpy as np
import branch_and_bound as bb
from collections import namedtuple

def print_array(message, array, max_line_len=25):
    print(message, end='')
    offset = len(message) + 2
    max_dec = len(str(np.max(array)))
    print('[ ', end='')
    for i, x in enumerate(array):
        print(str(x).ljust(max_dec), end=' ')
        if (i + 1) % max_line_len == 0 and i + 1 < len(array):
            print('\n'.ljust(offset + 1), end='')
    print(']')

def read_test_case_file(filename):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'test_cases', filename)
    return np.loadtxt(path).astype(np.int32)

def read_test_case(name):
    matrix_file      = '{}_matrix.csv'.format(name)
    constraints_file = '{}_constraints.csv'.format(name)
    reference_file   = '{}_reference.csv'.format(name)
    T = read_test_case_file(matrix_file)
    d = read_test_case_file(constraints_file).flatten()
    x = read_test_case_file(reference_file).flatten()
    return T, d, x[1:], x[0]

def get_lower_bound_eval(config):
    makers = { 'default': lambda: bb.LowerBoundDefault() }
    return makers[config.lower]

def get_upper_bound_eval(config):
    makers = { 'default': lambda: bb.UpperBoundDefault(),
               'distance': lambda: bb.UpperBoundByDistance(),
               'aneal': lambda: bb.UpperBoundAneal() }
    subnames = config.upper.split('+')
    multiple_eval = [makers[n] for n in subnames]
    return try_combine_evals(multiple_eval)

def get_node_container(config):
    makers = { 'default': lambda: bb.NodeStack(),
               'queue':   lambda: bb.NodeQueue(),
               'stack':   lambda: bb.NodeStack(),
               'heap':    lambda: bb.NodeHeapUpper(),
               'uheap':   lambda: bb.NodeHeapUpper(),
               'lheap':   lambda: bb.NodeHeapLower(),
               'ulheap':  lambda: bb.NodeHeapUpperLower(),
               'luheap':  lambda: bb.NodeHeapLowerUpper() }
    return makers[config.collection]

def get_task_names(config):
    all_tasks_num = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
    all_tasks = ['task_{}'.format(i) for i in all_tasks_num]
    assert config.task == 'all' or config.task in all_tasks
    return all_tasks if config.task == 'all' else [config.task]

def try_combine_evals(multiple_eval_makers):
    if len(multiple_eval_makers) == 1:
        return multiple_eval_makers[0]
    multiple_eval = [maker() for maker in multiple_eval_makers]
    return lambda: bb.CombinedBoundEval(multiple_eval)

def warmup_solver(strategy):
    T, d = np.ones((3, 3)) - np.eye(3), np.ones(2)
    bb.minimize(T, d, strategy)

def efficiency(sum_trials):
    baseline_sum_trials = 228665
    baseline_trials_per_task = {
        'task_1':  1,
        'task_2':  1,
        'task_3':  83,
        'task_4':  60,
        'task_5':  72,
        'task_6':  628,
        'task_7':  5869,
        'task_8':  87742,
        'task_9':  1,
        'task_10': 134208
    }
    return (baseline_sum_trials / sum_trials)

def solve_task(task, strategy, config):
    T, d, x, w = read_test_case(task)
    t_0 = time.time()
    res = bb.minimize(T, d, strategy, verbose=config.verbose)
    t_1 = time.time()
    dur_ms = (t_1 - t_0) * 1000
    status = 'ok' if res.w == w else 'fail'
    print('Run {{ {}: N = {} }} ({}) {:.2f}ms'
          .format(task, res.N, status, dur_ms))
    print_array('greedy   {}  '.format(res.w_greedy), res.x_greedy)
    print_array('optimal  {}  '.format(res.w), res.x)
    print()
    Result = namedtuple('Result', ['trials', 'time'])
    return Result(res.N, dur_ms)

def main(config):
    strategy = ( get_upper_bound_eval(config),
                 get_lower_bound_eval(config),
                 get_node_container(config) )
    if not config.nowarmup:
        warmup_solver(strategy)
    tasks = get_task_names(config)
    results = [ solve_task(task, strategy, config) for task in tasks ]
    sum_time = np.sum([r.time for r in results])
    sum_trials = np.sum([r.trials for r in results])
    print('sum(time) = {:.2f}ms'.format(sum_time))
    print('sum(trials) = {}'.format(sum_trials))
    print('efficiency = {:.2f}'.format(efficiency(sum_trials)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all',
                        help='Name of the test problem')
    parser.add_argument('--upper', type=str, default='default',
                        help='Strategy of upper bound computation')
    parser.add_argument('--lower', type=str, default='default',
                        help='Strategy of lower bound computation')
    parser.add_argument('--collection', type=str, default='default',
                        help='Data structure storing nodes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print intermediate steps')
    parser.add_argument('--nowarmup', action='store_false', default=True,
                        help='Disables warmuping')
    main(parser.parse_args())
