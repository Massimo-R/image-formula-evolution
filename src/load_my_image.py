from ga import GeneticAlgorithm
from images import load_image, save_image
from visualize import visualize_results

# Load your image
target = load_image('names-of-shapes.png', size=200)

# Optional: Save preprocessed version to see what it looks like
save_image(target, 'preprocessed_target.png')

# Run evolution
ga = GeneticAlgorithm(
    target_image=target,
    approach='parametric',
    population_size=150,
    max_generations=500
)

ga.run()
visualize_results(ga, save_path="custom_results.png", show=False)