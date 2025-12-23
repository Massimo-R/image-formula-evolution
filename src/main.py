"""
main.py
Main script to run image formula evolution experiments
"""

import argparse
from ga import GeneticAlgorithm
from images import (
    create_gradient_image,
    create_circle_image,
    create_checkerboard_image,
    create_stripes_image,
    save_image
)
from visualize import visualize_results, plot_fitness_curve
from fitness import evaluate_all_metrics


def run_experiment(target_type='gradient',
                   approach='parametric',
                   image_size=50,
                   population_size=100,
                   max_generations=300,
                   mutation_rate=0.15,
                   crossover_rate=0.8):
    """
    Run a single evolution experiment

    Args:
        target_type: Type of target image ('gradient', 'circle', 'checkerboard', 'stripes')
        approach: Evolution approach ('parametric' or 'gp')
        image_size: Size of target image
        population_size: GA population size
        max_generations: Number of generations to evolve
        mutation_rate: Mutation probability
        crossover_rate: Crossover probability
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {target_type.upper()} - {approach.upper()}")
    print(f"{'='*70}\n")

    # Create target image
    print("Creating target image...")
    if target_type == 'gradient':
        target = create_gradient_image(image_size, 'horizontal')
    elif target_type == 'circle':
        target = create_circle_image(image_size)
    elif target_type == 'checkerboard':
        target = create_checkerboard_image(image_size)
    elif target_type == 'stripes':
        target = create_stripes_image(image_size)
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    # Initialize GA
    ga = GeneticAlgorithm(
        target_image=target,
        approach=approach,
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )

    # Run evolution
    ga.run(verbose=True, log_interval=50)

    # Evaluate final metrics
    print("\nEvaluating final metrics...")
    metrics = evaluate_all_metrics(
        ga.best_formula,
        ga.target_image,
        ga.coords_m,
        ga.coords_n
    )

    print(f"\nFinal Metrics:")
    print(f"  Similarity: {metrics['similarity']:.2f}%")
    print(f"  MSE:        {metrics['mse']:.2f}")
    print(f"  PSNR:       {metrics['psnr']:.2f} dB")
    print(f"  MAE:        {metrics['mae']:.2f}")

    # Save results
    result_prefix = f"{target_type}_{approach}_{image_size}"

    print(f"\nSaving results...")
    save_image(ga.get_best_image(), f"{result_prefix}_evolved.png")
    save_image(target, f"{result_prefix}_target.png")

    visualize_results(ga, save_path=f"{result_prefix}_results.png", show=False)
    plot_fitness_curve(ga, save_path=f"{result_prefix}_fitness.png", show=False)

    print(f"\nResults saved with prefix: {result_prefix}")
    print(f"{'='*70}\n")

    return ga, metrics


def run_comparison_experiments():
    """
    Run multiple experiments comparing different approaches and targets
    """
    print("\n" + "="*70)
    print("RUNNING COMPARISON EXPERIMENTS")
    print("="*70 + "\n")

    experiments = [
        ('gradient', 'parametric', 50, 100, 300),
        ('circle', 'parametric', 50, 100, 300),
        ('checkerboard', 'parametric', 50, 100, 300),
        ('gradient', 'gp', 50, 100, 200),
    ]

    results = []

    for target_type, approach, size, pop, gens in experiments:
        ga, metrics = run_experiment(
            target_type=target_type,
            approach=approach,
            image_size=size,
            population_size=pop,
            max_generations=gens
        )
        results.append((target_type, approach, ga, metrics))

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Target':<15} {'Approach':<12} {'Similarity':<12} {'MSE':<10} {'PSNR':<10}")
    print("-"*70)

    for target_type, approach, ga, metrics in results:
        print(f"{target_type:<15} {approach:<12} "
              f"{metrics['similarity']:>10.2f}% "
              f"{metrics['mse']:>9.2f} "
              f"{metrics['psnr']:>9.2f}")

    print("="*70 + "\n")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Image Formula Evolution using GA/GP'
    )

    parser.add_argument('--target', type=str, default='gradient',
                        choices=['gradient', 'circle', 'checkerboard', 'stripes'],
                        help='Type of target image')

    parser.add_argument('--approach', type=str, default='parametric',
                        choices=['parametric', 'gp'],
                        help='Evolution approach')

    parser.add_argument('--size', type=int, default=50,
                        help='Image size (width and height)')

    parser.add_argument('--population', type=int, default=100,
                        help='Population size')

    parser.add_argument('--generations', type=int, default=300,
                        help='Maximum number of generations')

    parser.add_argument('--mutation-rate', type=float, default=0.15,
                        help='Mutation rate')

    parser.add_argument('--crossover-rate', type=float, default=0.8,
                        help='Crossover rate')

    parser.add_argument('--compare', action='store_true',
                        help='Run comparison experiments instead')

    args = parser.parse_args()

    if args.compare:
        run_comparison_experiments()
    else:
        run_experiment(
            target_type=args.target,
            approach=args.approach,
            image_size=args.size,
            population_size=args.population,
            max_generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate
        )


if __name__ == "__main__":
    # Quick test run if no arguments provided
    import sys

    if len(sys.argv) == 1:
        print("Running quick test with default parameters...")
        print("Use --help to see available options\n")

        run_experiment(
            target_type='gradient',
            approach='parametric',
            image_size=50,
            population_size=100,
            max_generations=300
        )
    else:
        main()