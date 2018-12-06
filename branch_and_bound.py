import numba
import numpy as np
from collections import deque

def minimize(matrix, constraints):
    T, d = _fix_inputs(matrix, constraints)
    return _minimize_bnb(T, d)

def _minimize_bnb(T, d, strategy=None):
    x_guess = Node().greedy(T, d)
    w_guess = _evalute_criterion(T, d, x_guess)
    print('x_guess_1 = {}, w = {}'.format(x_guess, w_guess))

    # x_guess = Node().greedy(T, d)
    # w_guess = _evalute_criterion(T, d, x_guess)
    # print('x_guess_1 = {}, w = {}'.format(x_guess, w_guess))


    root = Node()
    queue = Queue()
    queue.push(root)

    counter = 0
    upper_bound_min = w_guess
    best_known_node = root
    best_known_node_w = root.upper

    while not queue.empty():
        node = queue.pop()

        counter += 1
        # if counter % 1000 == 0:
        #     print(node)

        nodes = node.branch(T, d)

        for n in nodes:
            upper_bound_min = min(upper_bound_min, n.upper)

        for n in nodes:
            if n.lower <= upper_bound_min:
                if n.lower == n.upper:
                    if n.upper < best_known_node_w:
                        best_known_node_w = n.upper
                        best_known_node = n
                else:
                    queue.push(n)


    n = best_known_node
    print('iters = {}'.format(counter))
    x = n.greedy(T, d)
    w = _evalute_criterion(T, d, x)
    print('x = {}, w = {}'.format(x, w))
    return x[1:], w

class Queue(object):
    def __init__(self):
        self.data = deque()

    def empty(self):
        return len(self.data) == 0

    def pop(self):
        return self.data.popleft()

    def push(self, node):
        self.data.append(node)

    def push_many(self, nodes):
        self.data.extend(nodes)

class Node(object):
    def __init__(self, T=None, d=None, x=None, used=None):
        self.x = x
        self.used = used
        self.leaf = False
        if x is None or used is None:
            self.lower = -np.inf
            self.upper = +np.inf
        else:
            candidates = _candidates(x, used)
            self.upper, _ = _upper_bound(T, d, x, candidates)
            self.lower = _lower_bound(T, d, x, candidates)
            self.leaf = len(x) == len(d) + 1

    def greedy(self, T, d):
        x, used = self._argument(d.shape[0])
        candidates = _candidates(x, used)
        _, x_ext = _upper_bound(T, d, x, candidates)
        return x_ext

    def branch(self, T, d):
        x, used = self._argument(d.shape[0])
        candidates = _candidates(x, used)
        nodes = []
        for cand in candidates:
            used_copy = used.copy()
            used_copy[cand] = True
            x_copy = np.hstack((x, cand))
            node = Node(T, d, x_copy, used_copy)
            nodes.append(node)
        return nodes

    def _argument(self, n):
        x, used = self.x, self.used
        if self.x is None:
            x = np.zeros(1, dtype=np.int32)
            used = np.full(n + 1, False, dtype=np.bool)
            used[0] = True
        return x, used

    def __repr__(self):
        return '{{ {}: {} {} }}'.format(str(self.x), self.lower, self.upper)


@numba.jit(nopython=True)
def _upper_bound(T, d, x, candidates):
    k = len(x) - 1
    m = len(candidates)
    n = m + k
    x_ext = np.empty(n + 1, dtype=np.int32)
    x_ext[:k + 1] = x
    z_sum = 0
    candidates_lst = [x for x in candidates]
    for i in range(0, k):
        z_sum += T[x[i], x[i + 1]]
    for i in range(0, m):
        w_min = np.iinfo(np.int32).max
        w_max = np.iinfo(np.int32).min
        best_cand = candidates_lst[0]
        for cand in candidates_lst:
            # print(cand)
            diff = d[cand - 1] - (z_sum + T[x_ext[k + i], cand])
            if diff >= 0 and w_min > diff:
                w_min = diff
                best_cand = cand
        candidates_lst.remove(best_cand)
        x_ext[k + i + 1] = best_cand
        z_sum += T[x_ext[k + i], best_cand]
    w = _evalute_criterion(T, d, x_ext)
    # print(x_ext, w) # debug
    return w, x_ext

@numba.jit(nopython=True)
def _lower_bound(T, d, x, candidates):
    k = len(x) - 1
    m = len(candidates)
    z = np.zeros(len(d) + 1, dtype=np.int32)
    z_sum = 0
    for i in range(0, k):
        z_sum += T[x[i], x[i + 1]]
        z[x[i + 1]] = z_sum
    for cand in candidates:
        z[cand] = z_sum + T[x[k], cand]
    return np.sum(z[1:] > d)

@numba.jit(nopython=True)
def _candidates(x, used):
    n = used.shape[0]
    candidates = []
    for i in range(0, n):
        if not used[i]:
            candidates.append(i)
    return np.array(candidates, dtype=np.int32)

def _minimize_naive(T, d):
    print(T)
    print(d)
    n = d.shape[0]
    w_min = n # Max value of criterion
    x_min = np.arange(n, dtype=np.int32)
    for x in _generate_permutations(n):
        w = _evalute_criterion(T, d, x)
        if w < w_min:
            w_min = w
            x_min[...] = x[1:] # Copy
    print(x_min, w_min)
    return x_min, w_min

@numba.jit(nopython=True)
def _evalute_criterion(T, d, x):
    n = d.shape[0]
    z = np.zeros(n + 1, dtype=np.int32)
    z_sum = 0
    for i in range(0, n):
        z_sum += T[x[i], x[i + 1]]
        z[x[i + 1]] = z_sum
    return np.sum(z[1:] > d)

def _generate_permutations(n):
    p_full = np.arange(n + 1, dtype=np.int32)
    p = p_full[1:]
    c = np.zeros(n, dtype=np.int32)
    yield p_full
    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                p[0], p[i] = p[i], p[0]
            else:
                p[c[i]], p[i] = p[i], p[c[i]]
            yield p_full
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i +=1

def _fix_inputs(matrix, constraints):
    T = np.asarray(matrix).astype(np.int32)
    d = np.asarray(constraints).astype(np.int32)
    assert len(d.shape) == 1
    assert len(T.shape) == 2
    assert d.shape[0] > 0
    assert T.shape[0] == T.shape[1]
    assert T.shape[0] == d.shape[0] + 1
    assert np.all(T == T.T)
    return T, d
