import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple, Dict, Optional
try:
    import heapdict
except ImportError:
    class _SimpleHeapDict(dict):
        def popitem(self):
            key = min(self, key=lambda k: self[k])
            val = self[key]
            del self[key]
            return key, val
        def peekitem(self):
            key = min(self, key=lambda k: self[k])
            return key, self[key]
    class heapdict:
        heapdict = _SimpleHeapDict
from collections import deque


class _Node:
    __slots__ = ("state", "parent", "action", "g")
    def __init__(self, state: int, parent: Optional['_Node']=None,
                 action: Optional[int]=None, g: float=0.0) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g


def _solution(node: _Node) -> Tuple[List[int], float]:
    actions: List[int] = []
    cur = node
    while cur.parent is not None:
        actions.append(cur.action)
        cur = cur.parent
    actions.reverse()
    return actions, float(node.g)


def _haifa_h(env: HaifaEnv, state: int) -> float:
    if state is None:
        return float('inf')
    goals = env.get_goal_states()
    if not goals:
        return float('inf')
    r, c = env.to_row_col(state)
    best = min(abs(r - env.to_row_col(g)[0]) + abs(c - env.to_row_col(g)[1]) for g in goals)
    # in the supplied environment entering any P cell costs 100
    return float(min(best, 100.0))


class BFSGAgent():
    def __init__(self) -> None:
        pass

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        start = env.get_initial_state()
        root = _Node(start, None, None, 0.0)
        if env.is_final_state(start):
            return [], 0.0, 0

        open_q = deque([root])
        open_states = {start}
        closed = set()
        expanded = 0

        while open_q:
            node = open_q.popleft()
            open_states.remove(node.state)
            if node.state in closed:
                continue
            closed.add(node.state)
            expanded += 1

            for action in range(env.action_space.n):
                new_state, cost, terminated = env.succ(node.state)[action]
                if new_state is None or cost is None:
                    continue
                child = _Node(int(new_state), node, action, node.g + float(cost))
                if env.is_final_state(child.state):
                    actions, total_cost = _solution(child)
                    return actions, total_cost, expanded
                if child.state not in closed and child.state not in open_states:
                    open_q.append(child)
                    open_states.add(child.state)

        return [], float('inf'), expanded


class GreedyAgent():
    def __init__(self) -> None:
        pass

    def h(self, env: HaifaEnv, state: int) -> float:
        return _haifa_h(env, state)

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        start = env.get_initial_state()
        root = _Node(start, None, None, 0.0)
        open_h = heapdict.heapdict()
        nodes: Dict[int, _Node] = {start: root}
        open_h[start] = (self.h(env, start), start)
        closed = set()
        expanded = 0

        while open_h:
            state, _ = open_h.popitem()
            node = nodes[state]
            if state in closed:
                continue
            closed.add(state)
            if env.is_final_state(state):
                actions, total_cost = _solution(node)
                return actions, total_cost, expanded
            expanded += 1

            for action in range(env.action_space.n):
                new_state, cost, terminated = env.succ(state)[action]
                if new_state is None or cost is None:
                    continue
                new_state = int(new_state)
                if new_state not in closed and new_state not in open_h:
                    child = _Node(new_state, node, action, node.g + float(cost))
                    nodes[new_state] = child
                    open_h[new_state] = (self.h(env, new_state), new_state)

        return [], float('inf'), expanded


class AStarEpsilonAgent():
    def __init__(self):
        pass

    def h(self, env: HaifaEnv, state: int) -> float:
        return _haifa_h(env, state)

    def h_focal(self, state: int) -> float: # heuristic for focal list
        # the assignment requests using H_HAIFA as the focal tie breaker
        # the method signature has no env parameter, so the actual implementation computes h inside search
        return 0.0

    def search(self, env: HaifaEnv, epsilon: float = None):
        if epsilon is None:
            epsilon = 0.0
        start = env.get_initial_state()
        root = _Node(start, None, None, 0.0)
        open_h = heapdict.heapdict()
        nodes: Dict[int, _Node] = {start: root}
        best_g: Dict[int, float] = {start: 0.0}
        open_h[start] = (self.h(env, start), start)  # f=g+h, g(start)=0
        closed = set()
        expanded = 0

        while open_h:
            # build focal list
            # all nodes with f <= (1+epsilon)*f_min
            _, (f_min, _) = open_h.peekitem()
            threshold = (1.0 + float(epsilon)) * f_min
            focal = []
            for st, pr in open_h.items():
                f_val = pr[0]
                if f_val <= threshold:
                    focal.append(st)
            # choose by h_HAIFA tie breaker, then by state index for determinism
            state = min(focal, key=lambda st: (self.h(env, st), st))
            # remove selected node from OPEN
            del open_h[state]
            node = nodes[state]
            if state in closed:
                continue
            closed.add(state)
            if env.is_final_state(state):
                actions, total_cost = _solution(node)
                return actions, total_cost, expanded
            expanded += 1

            for action in range(env.action_space.n):
                new_state, cost, terminated = env.succ(state)[action]
                if new_state is None or cost is None:
                    continue
                new_state = int(new_state)
                new_g = node.g + float(cost)
                if new_state in closed:
                    # reopen if a better path was found (needed for admissible but inconsistent heuristics)
                    if new_g < best_g.get(new_state, float('inf')):
                        closed.remove(new_state)
                    else:
                        continue
                if new_g < best_g.get(new_state, float('inf')):
                    child = _Node(new_state, node, action, new_g)
                    nodes[new_state] = child
                    best_g[new_state] = new_g
                    open_h[new_state] = (new_g + self.h(env, new_state), new_state)

        return [], float('inf'), expanded


class AStarAgent():
    def __init__(self):
        self._astar_epsilon = AStarEpsilonAgent()

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        return self._astar_epsilon.search(env, epsilon=0.0)
