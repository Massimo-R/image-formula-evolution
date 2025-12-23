"""
fitness.py
Fitness evaluation functions and image quality metrics
"""

import numpy as np
from formula import ParametricFormula, ExpressionTree


def calculate_fitness(formula, target_image: np.ndarray,
                      coords_m: np.ndarray, coords_n: np.ndarray,
                      sample_rate: float = 0.2) -> float:
    """
    Calculate fitness as similarity percentage (0-100%)
    Higher is better

    Args:
        formula: ParametricFormula or ExpressionTree
        target_image: Target image array (grayscale)
        coords_m: Normalized m coordinates
        coords_n: Normalized n coordinates
        sample_rate: Fraction of pixels to sample (0-1)

    Returns:
        Similarity score as percentage (0-100)
    """
    h, w = target_image.shape[:2]

    # Sample pixels for computational efficiency
    mask = np.random.random((h, w)) < sample_rate
    sampled_m = coords_m[mask]
    sampled_n = coords_n[mask]
    sampled_target = target_image[mask]

    # Generate values from formula
    if isinstance(formula, ParametricFormula):
        generated = formula.evaluate(sampled_m, sampled_n)
    else:  # ExpressionTree
        generated = formula.evaluate(sampled_m, sampled_n)

    # Calculate MSE
    mse = np.mean((sampled_target.astype(float) - generated.astype(float)) ** 2)

    # Convert to similarity percentage
    max_mse = 255 ** 2  # Maximum possible MSE for 8-bit images
    similarity = 100 * (1 - min(mse / max_mse, 1))

    return similarity


def calculate_mse(formula, target_image: np.ndarray,
                  coords_m: np.ndarray, coords_n: np.ndarray) -> float:
    """
    Calculate Mean Squared Error on full image

    Args:
        formula: ParametricFormula or ExpressionTree
        target_image: Target image array (grayscale)
        coords_m: Normalized m coordinates
        coords_n: Normalized n coordinates

    Returns:
        MSE value (lower is better)
    """
    if isinstance(formula, ParametricFormula):
        generated = formula.evaluate(coords_m, coords_n)
    else:
        generated = formula.evaluate(coords_m, coords_n)

    mse = np.mean((target_image.astype(float) - generated.astype(float)) ** 2)
    return mse


def calculate_psnr(formula, target_image: np.ndarray,
                   coords_m: np.ndarray, coords_n: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio

    Args:
        formula: ParametricFormula or ExpressionTree
        target_image: Target image array (grayscale)
        coords_m: Normalized m coordinates
        coords_n: Normalized n coordinates

    Returns:
        PSNR in dB (higher is better)
    """
    mse = calculate_mse(formula, target_image, coords_m, coords_n)

    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


def calculate_mae(formula, target_image: np.ndarray,
                  coords_m: np.ndarray, coords_n: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error

    Args:
        formula: ParametricFormula or ExpressionTree
        target_image: Target image array (grayscale)
        coords_m: Normalized m coordinates
        coords_n: Normalized n coordinates

    Returns:
        MAE value (lower is better)
    """
    if isinstance(formula, ParametricFormula):
        generated = formula.evaluate(coords_m, coords_n)
    else:
        generated = formula.evaluate(coords_m, coords_n)

    mae = np.mean(np.abs(target_image.astype(float) - generated.astype(float)))
    return mae


def calculate_complexity_penalty(formula, penalty_weight=0.01) -> float:
    """
    Calculate complexity penalty for formula
    Used in multi-objective optimization

    Args:
        formula: ParametricFormula or ExpressionTree
        penalty_weight: Weight for complexity penalty

    Returns:
        Complexity penalty value
    """
    if isinstance(formula, ParametricFormula):
        # Penalize large parameter values
        complexity = np.sum(np.abs(formula.params))
    else:  # ExpressionTree
        # Penalize tree size and depth
        complexity = formula.size() + formula.depth() * 2

    return penalty_weight * complexity


def calculate_fitness_with_complexity(formula, target_image: np.ndarray,
                                      coords_m: np.ndarray, coords_n: np.ndarray,
                                      complexity_weight=0.01,
                                      sample_rate=0.2) -> float:
    """
    Calculate fitness with complexity penalty
    Balances accuracy vs formula simplicity

    Args:
        formula: ParametricFormula or ExpressionTree
        target_image: Target image array
        coords_m: Normalized m coordinates
        coords_n: Normalized n coordinates
        complexity_weight: Weight for complexity penalty
        sample_rate: Fraction of pixels to sample

    Returns:
        Combined fitness score
    """
    accuracy_fitness = calculate_fitness(formula, target_image,
                                         coords_m, coords_n, sample_rate)
    complexity_penalty = calculate_complexity_penalty(formula, complexity_weight)

    return accuracy_fitness - complexity_penalty


def evaluate_all_metrics(formula, target_image: np.ndarray,
                         coords_m: np.ndarray, coords_n: np.ndarray) -> dict:
    """
    Evaluate all quality metrics for a formula

    Args:
        formula: ParametricFormula or ExpressionTree
        target_image: Target image array
        coords_m: Normalized m coordinates
        coords_n: Normalized n coordinates

    Returns:
        Dictionary with all metric values
    """
    metrics = {
        'similarity': calculate_fitness(formula, target_image, coords_m, coords_n, sample_rate=1.0),
        'mse': calculate_mse(formula, target_image, coords_m, coords_n),
        'psnr': calculate_psnr(formula, target_image, coords_m, coords_n),
        'mae': calculate_mae(formula, target_image, coords_m, coords_n),
    }

    if isinstance(formula, ExpressionTree):
        metrics['tree_size'] = formula.size()
        metrics['tree_depth'] = formula.depth()
    else:
        metrics['param_count'] = len(formula.params)

    return metrics