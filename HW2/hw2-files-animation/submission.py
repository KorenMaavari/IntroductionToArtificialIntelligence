from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, delivery_reward_multiplier
import random
import time


EXPECTIMAX_ACTION_WEIGHTS = {
    "move north": 4,
    "charge": 4,
}

TIE_BREAKING_ORDER = [
    "drop off",
    "pick up",
    "charge",
    "move north",
    "move east",
    "move south",
    "move west",
    "park",
]

class _SearchTimeout(Exception):
    pass


def _check_timeout(deadline):
    if deadline is not None and time.time() >= deadline:
        raise _SearchTimeout()


def _ordered_operators(operators):
    """return operators ordered according to the required deterministic tie breaker"""
    order = {op: i for i, op in enumerate(TIE_BREAKING_ORDER)}
    return sorted(operators, key=lambda op: order.get(op, len(order)))


def _other(robot_id):
    return (robot_id + 1) % 2


def _delivery_reward(package):
    return manhattan_distance(package.position, package.destination) * delivery_reward_multiplier


def _nearest_charge_distance(env: WarehouseEnv, position):
    if not env.charge_stations:
        return 0
    return min(manhattan_distance(position, station.position) for station in env.charge_stations)


def _available_packages(env: WarehouseEnv):
    return [p for p in env.packages[:2] if p.on_board]


def _robot_potential(env: WarehouseEnv, robot_id: int):
    """
    a utility estimate for one robot. the final heuristic is the difference
    between the two robots' potentials, so larger is better for robot_id
    """
    robot = env.get_robot(robot_id)

    # 1) real current score. this is the true objective at the end of the game
    value = 10.0 * robot.credit

    # 2) battery is valuable because it enables future deliveries, but it is
    # deliberately weighted lower than actual credit
    value += 0.4 * robot.battery

    if robot.package is not None:
        package = robot.package
        reward = _delivery_reward(package)
        dist_to_goal = manhattan_distance(robot.position, package.destination)

        # carrying a package is promising, especially if it is close to delivery
        value += 2.0 * reward
        value -= 1.5 * dist_to_goal

        # strongly prefer delivering immediately when possible
        if dist_to_goal == 0:
            value += 20.0

        needed_battery = dist_to_goal
    else:
        # 3) if not carrying anything, estimate the best package opportunity:
        # package value minus the distance to pick it up and deliver it
        best_package_value = 0.0
        best_required_battery = 0
        for package in _available_packages(env):
            dist_to_package = manhattan_distance(robot.position, package.position)
            dist_package_to_goal = manhattan_distance(package.position, package.destination)
            reward = _delivery_reward(package)
            route_len = dist_to_package + dist_package_to_goal
            candidate = 2.0 * reward - 1.2 * route_len
            if candidate > best_package_value:
                best_package_value = candidate
                best_required_battery = route_len
        value += best_package_value
        needed_battery = best_required_battery

    # 4) charging awareness. a robot with too little battery to complete a
    # useful route should prefer being close to a charging station. charging
    # is useful only if it has credit to convert into battery
    if robot.battery < needed_battery:
        value -= 4.0 * (needed_battery - robot.battery)
        value -= 2.0 * _nearest_charge_distance(env, robot.position)
        if env.get_charge_station_in(robot.position) is not None and robot.credit > 0:
            value += 8.0
    elif robot.battery <= 2:
        value -= 1.5 * _nearest_charge_distance(env, robot.position)

    return value


# section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    return _robot_potential(env, robot_id) - _robot_potential(env, _other(robot_id))


def _evaluate(env: WarehouseEnv, robot_id: int, heuristic_fn):
    return heuristic_fn(env, robot_id)


def _minimax_value(env: WarehouseEnv, current_robot: int, maximizing_robot: int,
                   depth: int, heuristic_fn, deadline=None):
    _check_timeout(deadline)
    if depth == 0 or env.done():
        return _evaluate(env, maximizing_robot, heuristic_fn)

    operators = _ordered_operators(env.get_legal_operators(current_robot))
    if not operators:
        return _evaluate(env, maximizing_robot, heuristic_fn)

    next_robot = _other(current_robot)
    if current_robot == maximizing_robot:
        best = -float("inf")
        for op in operators:
            child = env.clone()
            child.apply_operator(current_robot, op)
            best = max(best, _minimax_value(child, next_robot, maximizing_robot,
                                            depth - 1, heuristic_fn, deadline))
        return best
    else:
        best = float("inf")
        for op in operators:
            child = env.clone()
            child.apply_operator(current_robot, op)
            best = min(best, _minimax_value(child, next_robot, maximizing_robot,
                                            depth - 1, heuristic_fn, deadline))
        return best


def _minimax_decision_internal(env: WarehouseEnv, robot_id: int, depth: int,
                               heuristic_fn=None, deadline=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic
    depth = max(1, depth)
    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if not operators:
        return "park"

    best_op = operators[0]
    best_value = -float("inf")
    for op in operators:
        _check_timeout(deadline)
        child = env.clone()
        child.apply_operator(robot_id, op)
        value = _minimax_value(child, _other(robot_id), robot_id, depth - 1,
                               heuristic_fn, deadline)
        if value > best_value:
            best_value = value
            best_op = op
    return best_op


# section b : fixed-depth helper for deterministic grading
def minimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited minimax.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    return _minimax_decision_internal(env, robot_id, depth, heuristic_fn, None)


def _alphabeta_value(env: WarehouseEnv, current_robot: int, maximizing_robot: int,
                     depth: int, alpha: float, beta: float, heuristic_fn, deadline=None):
    _check_timeout(deadline)
    if depth == 0 or env.done():
        return _evaluate(env, maximizing_robot, heuristic_fn)

    operators = _ordered_operators(env.get_legal_operators(current_robot))
    if not operators:
        return _evaluate(env, maximizing_robot, heuristic_fn)

    next_robot = _other(current_robot)
    if current_robot == maximizing_robot:
        value = -float("inf")
        for op in operators:
            child = env.clone()
            child.apply_operator(current_robot, op)
            value = max(value, _alphabeta_value(child, next_robot, maximizing_robot,
                                                depth - 1, alpha, beta,
                                                heuristic_fn, deadline))
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value
    else:
        value = float("inf")
        for op in operators:
            child = env.clone()
            child.apply_operator(current_robot, op)
            value = min(value, _alphabeta_value(child, next_robot, maximizing_robot,
                                                depth - 1, alpha, beta,
                                                heuristic_fn, deadline))
            beta = min(beta, value)
            if value <= alpha:
                break
        return value


def _alphabeta_decision_internal(env: WarehouseEnv, robot_id: int, depth: int,
                                 heuristic_fn=None, deadline=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic
    depth = max(1, depth)
    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if not operators:
        return "park"

    best_op = operators[0]
    best_value = -float("inf")
    alpha = -float("inf")
    beta = float("inf")
    for op in operators:
        _check_timeout(deadline)
        child = env.clone()
        child.apply_operator(robot_id, op)
        value = _alphabeta_value(child, _other(robot_id), robot_id, depth - 1,
                                 alpha, beta, heuristic_fn, deadline)
        if value > best_value:
            best_value = value
            best_op = op
        alpha = max(alpha, best_value)
    return best_op


# section c : fixed-depth helper for deterministic grading
def alphabeta_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited alpha-beta pruning.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    return _alphabeta_decision_internal(env, robot_id, depth, heuristic_fn, None)


def _expectimax_value(env: WarehouseEnv, current_robot: int, maximizing_robot: int,
                      depth: int, heuristic_fn, deadline=None):
    _check_timeout(deadline)
    if depth == 0 or env.done():
        return _evaluate(env, maximizing_robot, heuristic_fn)

    operators = _ordered_operators(env.get_legal_operators(current_robot))
    if not operators:
        return _evaluate(env, maximizing_robot, heuristic_fn)

    next_robot = _other(current_robot)
    if current_robot == maximizing_robot:
        best = -float("inf")
        for op in operators:
            child = env.clone()
            child.apply_operator(current_robot, op)
            best = max(best, _expectimax_value(child, next_robot, maximizing_robot,
                                               depth - 1, heuristic_fn, deadline))
        return best
    else:
        weights = [EXPECTIMAX_ACTION_WEIGHTS.get(op, 1) for op in operators]
        total_weight = sum(weights)
        expected_value = 0.0
        for op, weight in zip(operators, weights):
            child = env.clone()
            child.apply_operator(current_robot, op)
            expected_value += (weight / total_weight) * _expectimax_value(
                child, next_robot, maximizing_robot, depth - 1, heuristic_fn, deadline
            )
        return expected_value


def _expectimax_decision_internal(env: WarehouseEnv, robot_id: int, depth: int,
                                  heuristic_fn=None, deadline=None):
    if heuristic_fn is None:
        heuristic_fn = smart_heuristic
    depth = max(1, depth)
    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if not operators:
        return "park"

    best_op = operators[0]
    best_value = -float("inf")
    for op in operators:
        _check_timeout(deadline)
        child = env.clone()
        child.apply_operator(robot_id, op)
        value = _expectimax_value(child, _other(robot_id), robot_id, depth - 1,
                                  heuristic_fn, deadline)
        if value > best_value:
            best_value = value
            best_op = op
    return best_op


# section d : fixed-depth helper for deterministic grading
def expectimax_decision(env: WarehouseEnv, robot_id: int, depth: int, heuristic_fn=None):
    """
    Return the selected legal operator using depth-limited expectimax.
    The opponent's legal actions are weighted by EXPECTIMAX_ACTION_WEIGHTS;
    every legal action not in the dictionary has weight 1.
    If heuristic_fn is None, use smart_heuristic.
    Ties must be broken according to TIE_BREAKING_ORDER.
    """
    return _expectimax_decision_internal(env, robot_id, depth, heuristic_fn, None)


def _fallback_action(env: WarehouseEnv, robot_id: int):
    """a safe action for very small time limits: one-ply greedy by the heuristic"""
    operators = _ordered_operators(env.get_legal_operators(robot_id))
    if not operators:
        return "park"

    best_op = operators[0]
    best_value = -float("inf")
    for op in operators:
        child = env.clone()
        child.apply_operator(robot_id, op)
        value = smart_heuristic(child, robot_id)
        if value > best_value:
            best_value = value
            best_op = op
    return best_op


def _iterative_deepening(env: WarehouseEnv, robot_id: int, time_limit: float, decision_function):
    start = time.time()
    # Leave a small safety margin so main.py will not reject the move for timeout.
    safety_margin = max(0.005, min(0.02, 0.10 * time_limit))
    deadline = start + max(0.0, time_limit - safety_margin)

    best_op = _fallback_action(env, robot_id)
    depth = 1
    while time.time() < deadline:
        try:
            candidate = decision_function(env, robot_id, depth, smart_heuristic, deadline)
            if candidate in env.get_legal_operators(robot_id):
                best_op = candidate
            depth += 1
        except _SearchTimeout:
            break
    return best_op


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # section b : 4
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return _iterative_deepening(env, agent_id, time_limit, _minimax_decision_internal)


class AgentAlphaBeta(Agent):
    # section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return _iterative_deepening(env, agent_id, time_limit, _alphabeta_decision_internal)


class AgentExpectimax(Agent):
    # section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        return _iterative_deepening(env, agent_id, time_limit, _expectimax_decision_internal)


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
