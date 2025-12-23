"""
visualize.py
Visualization and plotting utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def visualize_results(ga, save_path: Optional[str] = None, show: bool = True):
    """
    Visualize evolution results in a comprehensive plot

    Args:
        ga: GeneticAlgorithm instance after evolution
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Target image
    axes[0, 0].imshow(ga.target_image, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Target Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Evolved image
    best_image = ga.get_best_image()
    axes[0, 1].imshow(best_image, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Evolved Image\nSimilarity: {ga.best_fitness:.2f}%',
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Difference map
    diff = np.abs(ga.target_image.astype(float) - best_image.astype(float))
    im = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=255)
    axes[1, 0].set_title('Absolute Difference', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Fitness curve
    axes[1, 1].plot(ga.fitness_history, linewidth=2, label='Best Fitness', color='blue')
    if ga.avg_fitness_history:
        axes[1, 1].plot(ga.avg_fitness_history, linewidth=2,
                        label='Average Fitness', color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Generation', fontsize=12)
    axes[1, 1].set_ylabel('Similarity (%)', fontsize=12)
    axes[1, 1].set_title('Evolution Progress', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to {save_path}")

    if show:
        try:
            plt.show()
        except:
            print("Display not supported, saving instead.")
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.close()


def plot_fitness_curve(ga, save_path: Optional[str] = None, show: bool = True):
    """
    Plot fitness evolution curve

    Args:
        ga: GeneticAlgorithm instance
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))

    plt.plot(ga.fitness_history, linewidth=2, label='Best Fitness', color='blue')

    if ga.avg_fitness_history:
        plt.plot(ga.avg_fitness_history, linewidth=2,
                 label='Average Fitness', color='orange', alpha=0.7)

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Similarity (%)', fontsize=12)
    plt.title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Fitness curve saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(target_images, evolved_images, titles,
                    save_path: Optional[str] = None, show: bool = True):
    """
    Plot comparison of multiple target and evolved images

    Args:
        target_images: List of target images
        evolved_images: List of evolved images
        titles: List of titles for each pair
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    n = len(target_images)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Target
        axes[0, i].imshow(target_images[i], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'Target: {titles[i]}', fontsize=12)
        axes[0, i].axis('off')

        # Evolved
        axes[1, i].imshow(evolved_images[i], cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f'Evolved: {titles[i]}', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_image_evolution_grid(ga_list, generation_snapshots,
                              save_path: Optional[str] = None, show: bool = True):
    """
    Plot grid showing evolution progress at different generations

    Args:
        ga_list: List of GeneticAlgorithm instances at different generations
        generation_snapshots: List of generation numbers
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    n = len(ga_list)
    fig, axes = plt.subplots(1, n + 1, figsize=(4*(n+1), 4))

    # Target
    axes[0].imshow(ga_list[0].target_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Target', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Evolution snapshots
    for i, (ga, gen) in enumerate(zip(ga_list, generation_snapshots)):
        img = ga.get_best_image()
        axes[i+1].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i+1].set_title(f'Gen {gen}\n{ga.best_fitness:.1f}%', fontsize=12)
        axes[i+1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Evolution grid saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(metrics_dict, save_path: Optional[str] = None, show: bool = True):
    """
    Plot comparison of different metrics

    Args:
        metrics_dict: Dictionary with keys as labels and values as metric dictionaries
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    labels = list(metrics_dict.keys())

    # Similarity
    similarities = [metrics_dict[label]['similarity'] for label in labels]
    axes[0, 0].bar(labels, similarities, color='skyblue')
    axes[0, 0].set_ylabel('Similarity (%)', fontsize=12)
    axes[0, 0].set_title('Similarity Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # MSE
    mses = [metrics_dict[label]['mse'] for label in labels]
    axes[0, 1].bar(labels, mses, color='salmon')
    axes[0, 1].set_ylabel('MSE', fontsize=12)
    axes[0, 1].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # PSNR
    psnrs = [metrics_dict[label]['psnr'] for label in labels]
    axes[1, 0].bar(labels, psnrs, color='lightgreen')
    axes[1, 0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1, 0].set_title('Peak Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # MAE
    maes = [metrics_dict[label]['mae'] for label in labels]
    axes[1, 1].bar(labels, maes, color='plum')
    axes[1, 1].set_ylabel('MAE', fontsize=12)
    axes[1, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def show_image(image: np.ndarray, title: str = "Image", cmap: str = 'gray'):
    """
    Quick display of a single image

    Args:
        image: Image array
        title: Image title
        cmap: Colormap to use
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap, vmin=0, vmax=255)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_animation(ga, interval: int = 100, save_path: Optional[str] = None):
    """
    Create animated visualization of evolution (requires running evolution with snapshots)

    Args:
        ga: GeneticAlgorithm instance
        interval: Milliseconds between frames
        save_path: Optional path to save animation (as GIF or MP4)
    """
    # Note: This requires storing snapshots during evolution
    # For now, this is a placeholder for future enhancement
    print("Animation feature requires storing evolution snapshots.")
    print("Implement snapshot storage in GA.run() for this feature.")


def plot_parameter_evolution(ga_snapshots, save_path: Optional[str] = None, show: bool = True):
    """
    Plot how parameters evolve over time (for parametric approach)

    Args:
        ga_snapshots: List of GA instances at different generations
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    from formula import ParametricFormula

    if not ga_snapshots or not isinstance(ga_snapshots[0].best_formula, ParametricFormula):
        print("This function only works with parametric formulas")
        return

    num_params = len(ga_snapshots[0].best_formula.params)
    generations = [ga.generation for ga in ga_snapshots]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for param_idx in range(min(num_params, 12)):
        param_values = [ga.best_formula.params[param_idx] for ga in ga_snapshots]
        axes[param_idx].plot(generations, param_values, linewidth=2, marker='o')
        axes[param_idx].set_xlabel('Generation')
        axes[param_idx].set_ylabel(f'Parameter {param_idx}')
        axes[param_idx].set_title(f'a_{param_idx} Evolution')
        axes[param_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Parameter evolution saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()