import numba
import numpy as np
from collections import namedtuple

xint = np.uint16
wint = np.uint32
wint_max = np.iinfo(wint).max
xint_max = np.iinfo(xint).max

class BoundEval(object):
    """Evaluates lower and upper bound"""
    def eval(self, T, d, x, candidates) -> (int, np.array): pass


class NodeContainer(object):
    """Container of tree nodes"""
    def push(node): pass
    def pop(): pass
    def empty(): pass


class Strategy(object):
    """Represents overall algorithm strategy"""
    def __init__(self, upper_bound_eval_maker,
                       lower_bound_eval_maker,
                       node_container_maker):
        self.upper_bound_eval_maker = upper_bound_eval_maker
        self.lower_bound_eval_maker = lower_bound_eval_maker
        self.node_container_maker = node_container_maker

    def make_upper_bound_eval(self) -> BoundEval:
        return self.upper_bound_eval_maker()

    def make_lower_bound_eval(self) -> BoundEval:
        return self.lower_bound_eval_maker()

    def make_node_container(self) -> NodeContainer:
        return self.node_container_maker()


class Context(object):
    def __init__(self, T: np.array,
                       d: np.array,
                       s: Strategy):
        self.T = T
        self.d = d
        self.n = len(d)
        self.strategy = s
        self.container = s.make_node_container()
        self.lower_eval = s.make_lower_bound_eval()
        self.upper_eval = s.make_upper_bound_eval()

    def eval_lower(self, x, candidates):
        return self.lower_eval.eval(self.T, self.d, x, candidates)

    def eval_upper(self, x, candidates):
        return self.upper_eval.eval(self.T, self.d, x, candidates)

    def init_container(self):
        x = np.zeros(1, dtype=xint)
        u = np.full(len(self.d) + 1, False, dtype=np.bool)
        u[0] = True
        root = Node(self, NodeArg(x, u))
        self.container.push(root)
        return self.container, root


class NodeArg(object):
    def __init__(self, x: np.array, u: np.array):
        self.x = x
        self.u = u

    def candidates(self) -> np.array:
        return _candidates(self.x, self.u)

    def xu(self):
        return (self.x, self.u)

    def x_copy(self):
        x_copy = np.empty(len(self.x) + 1, dtype=xint)
        x_copy[:-1] = self.x
        return x_copy


class Node(object):
    """Represents single tree node"""
    def __init__(self, cntx, arg):
        self.arg = arg
        self._cntx = cntx
        self._eval_upper_cache = None
        self._eval_lower_cache = None
        self._candidates = arg.candidates()

    def lower(self):
        return self._eval_lower_cached()[0]

    def upper(self):
        return self._eval_upper_cached()[0]

    def branch(self):
        nodes = []
        x_copy = self.arg.x_copy()
        for cand in self._candidates:
            u_copy = self.arg.u.copy()
            u_copy[cand] = True
            x_copy[-1] = cand
            arg = NodeArg(x_copy.copy(), u_copy)
            node = Node(self._cntx, arg)
            nodes.append(node)
        return nodes

    def greedy(self):
        return self._eval_upper_cached()

    def is_root(self):
        return len(self.arg.x) == 1

    def _eval_upper(self):
        w, x = self._cntx.eval_upper(self.arg.x,
                                     self._candidates)
        return w, x

    def _eval_lower(self):
        return self._cntx.eval_lower(self.arg.x,
                                     self._candidates)

    def _eval_upper_cached(self):
        if not self._eval_upper_cache:
            self._eval_upper_cache = self._eval_upper()
        return self._eval_upper_cache

    def _eval_lower_cached(self):
        if not self._eval_lower_cache:
            self._eval_lower_cache = self._eval_lower()
        return self._eval_lower_cache

    def __repr__(self):
        x, l, u = self.arg.x, self.lower(), self.upper()
        return '{{ {}: {} {} }}'.format(str(x), l, u)


Result = namedtuple('Result', ['w', 'x', 'N',
                    'w_greedy', 'x_greedy'])

@numba.jit(nogil=True)
def minimize(matrix, constraints, strategy=None, verbose=False):
    s = _fix_strategy(strategy)
    T, d = _fix_inputs(matrix, constraints)

    cntx = Context(T, d, s)
    container, root = cntx.init_container()
    w_greedy, x_greedy = root.greedy()

    counter = 1
    best_known_node = root
    while not container.empty():
        node = container.pop()
        if node.lower() >= best_known_node.upper():
            continue
        # if node.upper() > best_known_node.upper():
        #     continue
        counter += 1

        added_children = 0
        for n in node.branch():
            if n.lower() < best_known_node.upper():
                if n.upper() < best_known_node.upper():
                    if verbose:
                        print(counter, "update", n)
                    best_known_node = n
                if n.lower() != n.upper():
                    added_children += 1
                    container.push(n)
        if verbose:
            print(counter, node, added_children)

    w, x = best_known_node.greedy()
    w = _evalute_criterion(T, d, x)
    return Result(w=w, x=x[1:], N=counter,
                  w_greedy=w_greedy,
                  x_greedy=x_greedy[1:])


class NodeDeque(NodeContainer):
    def __init__(self):
        super(NodeDeque, self).__init__()
        from collections import deque
        self.data = deque()

    def empty(self):
        return len(self.data) == 0

    def push(self, node):
        self.data.append(node)


class NodeQueue(NodeDeque):
    def pop(self):
        return self.data.popleft()


class NodeStack(NodeDeque):
    def pop(self):
        return self.data.pop()


class NodeHeap(NodeContainer):
    def __init__(self):
        super(NodeHeap, self).__init__()
        self.data = list()

    def empty(self):
        return len(self.data) == 0

    def pop(self):
        from heapq import heappop
        return heappop(self.data)

    def push(self, node):
        from heapq import heappush
        return heappush(self.data, node)


class NodeHeapUpper(NodeHeap):
    def __init__(self):
        super(NodeHeapUpper, self).__init__()
        Node.__lt__ = lambda x, y: x.upper() < y.upper()


class NodeHeapLower(NodeHeap):
    def __init__(self):
        super(NodeHeapLower, self).__init__()
        Node.__lt__ = lambda x, y: x.lower() < y.lower()


class NodeHeapUpperLower(NodeHeap):
    def __init__(self):
        super(NodeHeapUpperLower, self).__init__()
        Node.__lt__ = ( lambda x, y: x.upper() < y.upper() or
                        (x.upper() == y.upper() and x.lower() < y.lower()) )


class NodeHeapLowerUpper(NodeHeap):
    def __init__(self):
        super(NodeHeapLowerUpper, self).__init__()
        Node.__lt__ = ( lambda x, y: x.lower() < y.lower() or
                        (x.lower() == y.lower() and x.upper() < y.upper()) )


class CombinedBoundEval(BoundEval):
    def __init__(self, bound_evals: list):
        self.bounds = bound_evals

    def eval(self, T, d, x, candidates):
        wx_best = (wint_max, None)
        for bound in self.bounds:
            wx = bound.eval(T, d, x, candidates)
            if wx[0] < wx_best[0]:
                wx_best = wx
        return wx_best


class UpperBoundDefault(BoundEval):
    def eval(self, T, d, x, candidates):
        return _upper_bound_default(T, d, x, candidates)


class UpperBoundByDistance(BoundEval):
    def eval(self, T, d, x, candidates):
        return _upper_bound_by_distance(T, d, x, candidates)


class UpperBoundAneal(BoundEval):
    def eval(self, T, d, x, candidates):
        return _upper_bound_aneal(T, d, x, candidates)


class LowerBoundDefault(BoundEval):
    def eval(self, T, d, x, candidates):
        return _lower_bound_default(T, d, x, candidates)


def _fix_inputs(matrix, constraints):
    T = np.asarray(matrix).astype(xint)
    d = np.asarray(constraints).astype(xint)
    assert len(d.shape) == 1
    assert len(T.shape) == 2
    assert d.shape[0] > 0
    assert T.shape[0] == T.shape[1]
    assert T.shape[0] == d.shape[0] + 1
    assert np.all(T == T.T)
    return T, d

def _fix_strategy(strategy):
    if strategy is None:
        upper = lambda: CombinedBoundEval([ UpperBoundDefault(),
                                            UpperBoundByDistance() ])
        upper = lambda: UpperBoundByDistance()
        lower = lambda: LowerBoundDefault()
        container = NodeQueue()
        return _fix_strategy_tuple((upper, lower, container))
    elif isinstance(strategy, Strategy):
        return strategy
    elif isinstance(strategy, tuple):
        return _fix_strategy_tuple(strategy)
    else:
        raise(Exception('Cannot convert {} into strategy'.format(strategy)))

def _fix_strategy_tuple(strategy):
    upper_bound_maker = strategy[0]
    lower_bound_maker = strategy[1]
    container_maker = strategy[2]
    assert isinstance(upper_bound_maker(), BoundEval)
    assert isinstance(lower_bound_maker(), BoundEval)
    assert isinstance(container_maker(), NodeContainer)
    return Strategy(upper_bound_maker, lower_bound_maker, container_maker)

@numba.jit(nopython=True)
def _lower_bound_default(T, d, x, candidates):
    k = len(x) - 1
    m = len(candidates)
    z_sum = 0
    w_sum = 0
    for i in range(0, k):
        z_sum += T[x[i], x[i + 1]]
        w_sum += wint(z_sum > d[x[i + 1] - 1])
    w = w_sum
    for cand in candidates:
        z = z_sum + T[x[k], cand]
        w += wint(z > d[cand - 1])
    return w, None

@numba.jit(nopython=True)
def _upper_bound_default(T, d, x, candidates):
    k = len(x) - 1
    m = len(candidates)
    z_sum = wint(0)
    w = wint(0)
    for i in range(0, k):
        z_sum += T[x[i], x[i + 1]]
        w += wint(z_sum > d[x[i + 1] - 1])

    candidates_used = [False for i in range(0, m)]
    x_ext = np.empty(m + k + 1, dtype=xint)
    x_ext[:k + 1] = x
    for i in range(0, m):
        w_min = wint_max
        best_cand_j = -1

        for j in range(0, m):
            if candidates_used[j]: continue
            if best_cand_j < 0: best_cand_j = j
            cand = candidates[j]
            diff = d[cand - 1] - (z_sum + T[x_ext[k + i], cand])
            if diff >= 0 and w_min > diff:
                w_min = diff
                best_cand_j = j

        candidates_used[best_cand_j] = True
        best_cand = candidates[best_cand_j]
        x_ext[k + i + 1] = best_cand
        z_sum += T[x_ext[k + i], best_cand]
        w += wint(z_sum > d[best_cand - 1])
    return w, x_ext

@numba.jit(nopython=True)
def _upper_bound_by_distance(T, d, x_beg, candidates):
    x_end = _greedy_shortest_path(T, candidates, x_beg[-1])
    x = np.hstack((x_beg, x_end))
    w = _evalute_criterion(T, d, x)
    return w, x

@numba.jit(nopython=True)
def _greedy_shortest_path(T, candidates, x_0):
    m = len(candidates)
    x = np.empty(m, dtype=xint)
    x_prev = x_0
    candidates_used = [False for i in range(0, m)]
    for i in range(0, m):
        dist_min = wint_max
        best_cand_j = -1
        for j in range(0, m):
            if candidates_used[j]: continue
            if best_cand_j < 0: best_cand_j = j
            cand = candidates[j]
            dist = T[x_prev, cand]
            if dist_min > dist:
                dist_min = dist
                best_cand_j = j
        candidates_used[best_cand_j] = True
        best_cand = candidates[best_cand_j]
        x[i] = best_cand
        x_prev = best_cand
    return x

import math
import numpy.random as rng

@numba.jit(nopython=True)
def _upper_bound_aneal(T, d, x_beg, candidates):
    k = len(x_beg) - 1
    m = len(candidates)
    if m < 2:
        return _upper_bound_by_distance(T, d, x_beg, candidates)

    stopping_temp = 1e-8
    alpha = 0.995
    temp = np.sqrt(m)
    max_iters = 100

    z, w = _evalute_partial_criterion(T, d, x_beg)
    cur_fitness, cur_sollution_ = _upper_bound_by_distance(T, d, x_beg, candidates)
    cur_sollution = cur_sollution_[k + 1 :]
    best_fitness = cur_fitness
    best_sollution = cur_sollution

    iters = 0
    while temp >= stopping_temp and iters < max_iters:
        cands = cur_sollution.copy()
        l = rng.randint(2, m + 1)
        i = rng.randint(0, m - l + 1)
        cands[i : (i + l)] = cands[i : (i + l)][::-1]
        fitness = _evalute_criterion(T, d, np.hstack((x_beg, cands)))
        if fitness <= cur_fitness:
            cur_fitness, cur_sollution = fitness, cands
            if fitness < best_fitness:
                best_fitness, best_sollution = fitness, cands
        else:
            p_accept = math.exp(-abs(fitness - cur_fitness) / temp)
            if rng.random() < p_accept:
                cur_fitness, cur_sollution = fitness, cands
        temp *= alpha
        iters += 1

    x = np.hstack((x_beg, best_sollution))
    w = _evalute_criterion(T, d, x)
    return w, x

@numba.jit(nopython=True)
def _candidates(x, used):
    n = used.shape[0]
    candidates = []
    for i in range(0, n):
        if not used[i]:
            candidates.append(i)
    return np.array(candidates, dtype=xint)

@numba.jit(nopython=True)
def _evalute_criterion(T, d, x):
    n = d.shape[0]
    z = np.zeros(n + 1, dtype=xint)
    z_sum = 0
    for i in range(0, n):
        z_sum += T[x[i], x[i + 1]]
        z[x[i + 1]] = z_sum
    return np.sum(z[1:] > d)

@numba.jit(nopython=True)
def _evalute_partial_criterion(T, d, x, z=0, w=0):
    n = x.shape[0]
    for i in range(0, n - 1):
        z += T[x[i], x[i + 1]]
        w += wint(z > d[x[i + 1] - 1])
    return z, w
