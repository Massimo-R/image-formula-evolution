"""
operators.py
Evolutionary operators: selection, crossover, and mutation
"""

import random
import numpy as np
from formula import ParametricFormula, ExpressionTree, create_random_tree


# ============================================================================
# SELECTION OPERATORS
# ============================================================================

def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Select individual using tournament selection

    Args:
        population: List of individuals (formulas)
        fitnesses: List of fitness values
        tournament_size: Number of individuals in tournament

    Returns:
        Selected individual
    """
    competitors = random.sample(list(zip(population, fitnesses)), tournament_size)
    winner = max(competitors, key=lambda x: x[1])
    return winner[0]


def roulette_wheel_selection(population, fitnesses):
    """
    Select individual using roulette wheel (fitness proportional) selection

    Args:
        population: List of individuals
        fitnesses: List of fitness values

    Returns:
        Selected individual
    """
    # Shift fitnesses to be non-negative
    min_fitness = min(fitnesses)
    adjusted_fitnesses = [f - min_fitness + 1 for f in fitnesses]

    total_fitness = sum(adjusted_fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0

    for individual, fitness in zip(population, adjusted_fitnesses):
        current += fitness
        if current >= pick:
            return individual

    return population[-1]


def rank_selection(population, fitnesses):
    """
    Select individual using rank-based selection

    Args:
        population: List of individuals
        fitnesses: List of fitness values

    Returns:
        Selected individual
    """
    # Sort by fitness and assign ranks
    sorted_pairs = sorted(zip(population, fitnesses), key=lambda x: x[1])
    ranks = list(range(1, len(population) + 1))

    total_rank = sum(ranks)
    pick = random.uniform(0, total_rank)
    current = 0

    for individual, rank in zip([p[0] for p in sorted_pairs], ranks):
        current += rank
        if current >= pick:
            return individual

    return sorted_pairs[-1][0]


# ============================================================================
# CROSSOVER OPERATORS - PARAMETRIC
# ============================================================================

def crossover_parametric_single_point(parent1: ParametricFormula,
                                      parent2: ParametricFormula) -> ParametricFormula:
    """
    Single-point crossover for parametric formulas

    Args:
        parent1: First parent formula
        parent2: Second parent formula

    Returns:
        Child formula
    """
    point = random.randint(1, len(parent1.params) - 1)
    child_params = np.concatenate([parent1.params[:point], parent2.params[point:]])
    return ParametricFormula(child_params)


def crossover_parametric_two_point(parent1: ParametricFormula,
                                   parent2: ParametricFormula) -> ParametricFormula:
    """
    Two-point crossover for parametric formulas

    Args:
        parent1: First parent formula
        parent2: Second parent formula

    Returns:
        Child formula
    """
    length = len(parent1.params)
    point1 = random.randint(1, length - 2)
    point2 = random.randint(point1 + 1, length - 1)

    child_params = np.concatenate([
        parent1.params[:point1],
        parent2.params[point1:point2],
        parent1.params[point2:]
    ])
    return ParametricFormula(child_params)


def crossover_parametric_uniform(parent1: ParametricFormula,
                                 parent2: ParametricFormula) -> ParametricFormula:
    """
    Uniform crossover for parametric formulas
    Each parameter randomly chosen from either parent

    Args:
        parent1: First parent formula
        parent2: Second parent formula

    Returns:
        Child formula
    """
    child_params = np.array([
        p1 if random.random() < 0.5 else p2
        for p1, p2 in zip(parent1.params, parent2.params)
    ])
    return ParametricFormula(child_params)


def crossover_parametric_blend(parent1: ParametricFormula,
                               parent2: ParametricFormula,
                               alpha=0.5) -> ParametricFormula:
    """
    Blend crossover (arithmetic crossover) for parametric formulas

    Args:
        parent1: First parent formula
        parent2: Second parent formula
        alpha: Blending coefficient (0-1)

    Returns:
        Child formula
    """
    child_params = alpha * parent1.params + (1 - alpha) * parent2.params
    return ParametricFormula(child_params)


# ============================================================================
# MUTATION OPERATORS - PARAMETRIC
# ============================================================================

def mutate_parametric_gaussian(formula: ParametricFormula,
                               mutation_rate=0.15,
                               sigma=2.0) -> ParametricFormula:
    """
    Gaussian mutation for parametric formulas

    Args:
        formula: Formula to mutate
        mutation_rate: Probability of mutating each parameter
        sigma: Standard deviation of Gaussian noise

    Returns:
        Mutated formula
    """
    mutated = formula.copy()
    for i in range(len(mutated.params)):
        if random.random() < mutation_rate:
            mutated.params[i] += np.random.normal(0, sigma)
    return mutated


def mutate_parametric_uniform(formula: ParametricFormula,
                              mutation_rate=0.15,
                              range_min=-1.0,
                              range_max=1.0) -> ParametricFormula:
    """
    Uniform mutation for parametric formulas

    Args:
        formula: Formula to mutate
        mutation_rate: Probability of mutating each parameter
        range_min: Minimum value to add
        range_max: Maximum value to add

    Returns:
        Mutated formula
    """
    mutated = formula.copy()
    for i in range(len(mutated.params)):
        if random.random() < mutation_rate:
            mutated.params[i] += random.uniform(range_min, range_max)
    return mutated


def mutate_parametric_reset(formula: ParametricFormula,
                            mutation_rate=0.05) -> ParametricFormula:
    """
    Reset mutation - randomly reinitialize parameters

    Args:
        formula: Formula to mutate
        mutation_rate: Probability of resetting each parameter

    Returns:
        Mutated formula
    """
    mutated = formula.copy()
    for i in range(len(mutated.params)):
        if random.random() < mutation_rate:
            mutated.params[i] = random.uniform(-10, 10)
    return mutated


# ============================================================================
# CROSSOVER OPERATORS - GP
# ============================================================================

def crossover_tree_subtree(parent1: ExpressionTree,
                           parent2: ExpressionTree) -> ExpressionTree:
    """
    Subtree crossover for expression trees

    Args:
        parent1: First parent tree
        parent2: Second parent tree

    Returns:
        Child tree
    """
    child = parent1.copy()

    # Get all nodes from both trees
    nodes1 = child.get_all_nodes()
    nodes2 = parent2.get_all_nodes()

    if len(nodes1) > 1 and nodes2:
        # Select random nodes
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)

        # Copy node2's subtree to replace node1
        replacement = node2.copy()

        # Replace node1 with replacement
        node1.type = replacement.type
        node1.value = replacement.value
        node1.left = replacement.left
        node1.right = replacement.right

    return child


def crossover_tree_one_point(parent1: ExpressionTree,
                             parent2: ExpressionTree) -> ExpressionTree:
    """
    One-point crossover at matching depth

    Args:
        parent1: First parent tree
        parent2: Second parent tree

    Returns:
        Child tree
    """
    # For simplicity, use subtree crossover
    # A true one-point would require matching tree structures
    return crossover_tree_subtree(parent1, parent2)


# ============================================================================
# MUTATION OPERATORS - GP
# ============================================================================

def mutate_tree_subtree(tree: ExpressionTree,
                        mutation_rate=0.1,
                        max_depth=3) -> ExpressionTree:
    """
    Subtree mutation - replace random subtree with new random tree

    Args:
        tree: Tree to mutate
        mutation_rate: Probability of mutation
        max_depth: Maximum depth of new subtree

    Returns:
        Mutated tree
    """
    if random.random() < mutation_rate:
        return create_random_tree(max_depth)

    mutated = tree.copy()
    nodes = mutated.get_all_nodes()

    if nodes and random.random() < 0.3:
        node = random.choice(nodes)
        replacement = create_random_tree(max_depth)

        node.type = replacement.type
        node.value = replacement.value
        node.left = replacement.left
        node.right = replacement.right

    return mutated


def mutate_tree_point(tree: ExpressionTree,
                      mutation_rate=0.15) -> ExpressionTree:
    """
    Point mutation - change values of nodes

    Args:
        tree: Tree to mutate
        mutation_rate: Probability of mutating each node

    Returns:
        Mutated tree
    """
    mutated = tree.copy()
    nodes = mutated.get_all_nodes()

    for node in nodes:
        if random.random() < mutation_rate:
            if node.type == 'const':
                # Mutate constant value
                node.value += random.uniform(-2, 2)
            elif node.type == 'var':
                # Swap variable
                node.value = 'n' if node.value == 'm' else 'm'
            elif node.type == 'op':
                # Change operator (keeping arity)
                if node.value in ['sin', 'cos', 'abs']:
                    node.value = random.choice(['sin', 'cos', 'abs'])
                else:
                    node.value = random.choice(['+', '-', '*', '/'])

    return mutated


def mutate_tree_hoist(tree: ExpressionTree) -> ExpressionTree:
    """
    Hoist mutation - replace tree with random subtree
    Reduces bloat

    Args:
        tree: Tree to mutate

    Returns:
        Mutated tree (hoisted subtree)
    """
    nodes = tree.get_all_nodes()

    if len(nodes) > 1:
        subtree = random.choice(nodes[1:])  # Don't select root
        return subtree.copy()

    return tree.copy()


def mutate_tree_shrink(tree: ExpressionTree) -> ExpressionTree:
    """
    Shrink mutation - replace random subtree with terminal

    Args:
        tree: Tree to mutate

    Returns:
        Mutated tree
    """
    mutated = tree.copy()
    nodes = mutated.get_all_nodes()

    if len(nodes) > 1:
        node = random.choice(nodes)

        # Replace with terminal
        if random.random() < 0.5:
            node.type = 'var'
            node.value = random.choice(['m', 'n'])
        else:
            node.type = 'const'
            node.value = random.uniform(-5, 5)

        node.left = None
        node.right = None

    return mutated