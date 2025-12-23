"""
ga.py
Main Genetic Algorithm / Genetic Programming engine
"""

import numpy as np
import random
from typing import List, Callable, Optional
from formula import ParametricFormula, create_random_tree
from fitness import calculate_fitness, calculate_mse
from operators import (
    tournament_selection,
    crossover_parametric_single_point,
    mutate_parametric_gaussian,
    crossover_tree_subtree,
    mutate_tree_subtree
)
from images import save_image
import os


class GeneticAlgorithm:
    """Main GA/GP evolution engine"""

    def __init__(self,
                 target_image: np.ndarray,
                 approach: str = 'parametric',
                 population_size: int = 100,
                 max_generations: int = 500,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 elitism_count: int = 1,
                 tournament_size: int = 3,
                 sample_rate: float = 0.2):

        self.target_image = target_image
        self.approach = approach
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.sample_rate = sample_rate

        # Normalize coordinates to [0, 1]
        h, w = target_image.shape[:2]
        m_coords = np.linspace(0, 1, w)
        n_coords = np.linspace(0, 1, h)
        self.coords_m, self.coords_n = np.meshgrid(m_coords, n_coords)

        # Evolution state
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.avg_fitness_history = []
        self.best_formula = None
        self.best_fitness = 0

        # NEW: evolution snapshots storage
        self.snapshots = []   # List[(generation, image, fitness)]

        self._initialize_population()

    def _initialize_population(self):
        """Create initial random population"""
        print(f"Initializing population of {self.population_size} individuals...")

        if self.approach == 'parametric':
            self.population = [ParametricFormula()
                               for _ in range(self.population_size)]
        else:
            self.population = [create_random_tree()
                               for _ in range(self.population_size)]

        print(f"Population initialized with {self.approach} approach.")

    def _evaluate_population(self) -> List[float]:
        fitnesses = []
        for individual in self.population:
            fitness = calculate_fitness(
                individual,
                self.target_image,
                self.coords_m,
                self.coords_n,
                self.sample_rate
            )
            fitnesses.append(fitness)
        return fitnesses

    def _select_parents(self, fitnesses: List[float]):
        parent1 = tournament_selection(self.population, fitnesses, self.tournament_size)
        parent2 = tournament_selection(self.population, fitnesses, self.tournament_size)
        return parent1, parent2

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            if self.approach == 'parametric':
                return crossover_parametric_single_point(parent1, parent2)
            else:
                return crossover_tree_subtree(parent1, parent2)
        return parent1.copy()

    def _mutate(self, individual):
        if self.approach == 'parametric':
            return mutate_parametric_gaussian(individual, self.mutation_rate)
        return mutate_tree_subtree(individual, self.mutation_rate)

    def evolve_generation(self):
        fitnesses = self._evaluate_population()
        best_idx = np.argmax(fitnesses)
        avg_fitness = np.mean(fitnesses)

        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_formula = self.population[best_idx].copy()

        self.fitness_history.append(self.best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Elitism
        new_population = []
        elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())

        # Offspring generation
        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents(fitnesses)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def run(self,
            callback: Optional[Callable] = None,
            verbose: bool = True,
            log_interval: int = 50):

        if verbose:
            print("\n" + "="*60)
            print("Starting Evolution")
            print("="*60)

        for gen in range(self.max_generations):
            self.evolve_generation()

            # NEW: snapshot every 25 generations
            if gen % 25 == 0 or gen == self.max_generations - 1:
                snapshot_image = self.get_best_image()
                self.snapshots.append((gen, snapshot_image, self.best_fitness))

            # Logging
            if verbose and gen % log_interval == 0:
                mse = calculate_mse(
                    self.best_formula,
                    self.target_image,
                    self.coords_m,
                    self.coords_n
                )
                print(f"Gen {gen:4d} | Best: {self.best_fitness:6.2f}% | "
                      f"Avg: {self.avg_fitness_history[-1]:6.2f}% | MSE: {mse:8.2f}")

            if callback:
                callback(self)

        if verbose:
            print("\nEvolution Complete!")
            print(f"Final Similarity: {self.best_fitness:.2f}%")
            print("Best Formula:", self.best_formula)

    def get_best_image(self) -> np.ndarray:
        if self.best_formula is None:
            raise ValueError("Run evolution first.")
        return self.best_formula.evaluate(self.coords_m, self.coords_n)

    def save_snapshots(self, folder="snapshots"):
        """Save snapshot images to disk."""
        os.makedirs(folder, exist_ok=True)
        for gen, img, fit in self.snapshots:
            path = f"{folder}/gen_{gen:04d}_{fit:.1f}.png"
            save_image(img, path)

    def get_statistics(self) -> dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
            "fitness_history": self.fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
            "final_mse": calculate_mse(
                self.best_formula,
                self.target_image,
                self.coords_m,
                self.coords_n
            ),
        }

    def reset(self):
        self.generation = 0
        self.fitness_history = []
        self.avg_fitness_history = []
        self.best_formula = None
        self.best_fitness = 0
        self.snapshots = []
        self._initialize_population()