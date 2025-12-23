"""
formula.py
Formula representation classes for parametric and GP approaches
"""

import numpy as np
import random
import operator


class ParametricFormula:
    """Parametric formula with fixed structure but evolved parameters"""

    def __init__(self, params=None, num_params=12):
        if params is None:
            self.params = np.random.uniform(-10, 10, num_params)
        else:
            self.params = np.array(params)

    def evaluate(self, m: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Evaluate formula: a1*sin(a2*m + a3) + a4*cos(a5*n + a6) +
                         a7*m + a8*n + a9*sin(a10*m*n) + a11

        Args:
            m: Normalized m coordinates [0, 1]
            n: Normalized n coordinates [0, 1]

        Returns:
            Pixel values in range [0, 255]
        """
        p = self.params
        try:
            value = (p[0] * np.sin(p[1] * m + p[2]) +
                     p[3] * np.cos(p[4] * n + p[5]) +
                     p[6] * m + p[7] * n +
                     p[8] * np.sin(p[9] * m * n) +
                     p[10])

            # Normalize to [0, 255]
            value = ((value + 10) / 20) * 255
            value = np.clip(value, 0, 255)

            # Handle NaN/Inf
            value = np.nan_to_num(value, nan=128, posinf=255, neginf=0)

            return value.astype(np.uint8)
        except:
            return np.full_like(m, 128, dtype=np.uint8)

    def copy(self):
        """Create a deep copy of this formula"""
        return ParametricFormula(self.params.copy())

    def __str__(self):
        """Human-readable string representation"""
        p = self.params
        return (f"f(m,n) = {p[0]:.2f}*sin({p[1]:.2f}*m + {p[2]:.2f}) + "
                f"{p[3]:.2f}*cos({p[4]:.2f}*n + {p[5]:.2f}) + "
                f"{p[6]:.2f}*m + {p[7]:.2f}*n + "
                f"{p[8]:.2f}*sin({p[9]:.2f}*m*n) + {p[10]:.2f}")


class ExpressionTree:
    """Expression tree node for Genetic Programming"""

    # Available mathematical operations
    OPS = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': lambda x, y: x / (y + 0.001),  # Protected division
        'sin': np.sin,
        'cos': np.cos,
        'abs': np.abs,
    }

    def __init__(self, node_type, value, left=None, right=None):
        """
        Create an expression tree node

        Args:
            node_type: 'op', 'var', or 'const'
            value: operator name, variable name, or constant value
            left: left child node
            right: right child node
        """
        self.type = node_type
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, m: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Recursively evaluate tree at given coordinates

        Args:
            m: Normalized m coordinates [0, 1]
            n: Normalized n coordinates [0, 1]

        Returns:
            Pixel values in range [0, 255]
        """
        try:
            if self.type == 'const':
                return np.full_like(m, self.value, dtype=float)
            elif self.type == 'var':
                return m if self.value == 'm' else n
            else:  # operator
                left_val = self.left.evaluate(m, n) if self.left else 0

                # Unary operators
                if self.value in ['sin', 'cos', 'abs']:
                    result = self.OPS[self.value](left_val)
                else:
                    # Binary operators
                    right_val = self.right.evaluate(m, n) if self.right else 0
                    result = self.OPS[self.value](left_val, right_val)

                # Normalize to [0, 255]
                result = ((result + 5) / 10) * 255
                result = np.clip(result, 0, 255)
                result = np.nan_to_num(result, nan=128, posinf=255, neginf=0)

                return result.astype(np.uint8)
        except:
            return np.full_like(m, 128, dtype=np.uint8)

    def copy(self):
        """Create a deep copy of this tree"""
        return ExpressionTree(
            self.type,
            self.value,
            self.left.copy() if self.left else None,
            self.right.copy() if self.right else None
        )

    def depth(self):
        """Calculate maximum depth of tree"""
        if not self.left and not self.right:
            return 1
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def size(self):
        """Calculate total number of nodes in tree"""
        count = 1
        if self.left:
            count += self.left.size()
        if self.right:
            count += self.right.size()
        return count

    def get_all_nodes(self):
        """Get list of all nodes in tree"""
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_all_nodes())
        if self.right:
            nodes.extend(self.right.get_all_nodes())
        return nodes

    def __str__(self):
        """Human-readable string representation"""
        if self.type == 'const':
            return f"{self.value:.2f}"
        elif self.type == 'var':
            return self.value
        elif self.value in ['sin', 'cos', 'abs']:
            return f"{self.value}({self.left})"
        else:
            return f"({self.left} {self.value} {self.right})"


def create_random_tree(max_depth=4, current_depth=0):
    """
    Generate a random expression tree

    Args:
        max_depth: Maximum depth of tree
        current_depth: Current depth in recursion

    Returns:
        Random ExpressionTree
    """
    # Terminal nodes at max depth or probabilistically
    if current_depth >= max_depth or (current_depth > 1 and random.random() < 0.3):
        # Create terminal: variable or constant
        if random.random() < 0.5:
            return ExpressionTree('var', random.choice(['m', 'n']))
        else:
            return ExpressionTree('const', random.uniform(-5, 5))

    # Operator node
    ops = ['+', '-', '*', '/', 'sin', 'cos']
    op = random.choice(ops)

    # Unary operators need only left child
    if op in ['sin', 'cos', 'abs']:
        return ExpressionTree('op', op,
                              create_random_tree(max_depth, current_depth + 1))
    else:
        # Binary operators need both children
        return ExpressionTree('op', op,
                              create_random_tree(max_depth, current_depth + 1),
                              create_random_tree(max_depth, current_depth + 1))