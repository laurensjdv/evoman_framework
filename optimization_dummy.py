###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os


def run(enemy=2):
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    experiment_name = "optimization_test"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[enemy],
        playermode="ai",
        player_controller=player_controller(
            n_hidden_neurons
        ),  # you  can insert your own controller here
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
    )

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # start writing your own code from here

    env.state_to_log()  # checks environment state

    run_mode = "train"

    dom_u = 1
    dom_l = -1
    npop = 100
    gens = 30
    mutation = 0.2
    last_best = 0
    n_offspring = 50

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(env, pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    print(
        "\n GENERATION "
        + str(ini_g)
        + " "
        + str(round(fit_pop[best], 6))
        + " "
        + str(round(mean, 6))
        + " "
        + str(round(std, 6))
    )

    last_sol = fit_pop[best]
    notimproved = 0

    results = {
        "mean": [np.mean(fit_pop)],
        "best": [np.max(fit_pop)],
        "std": [np.std(fit_pop)],
    }

    for i in range(ini_g + 1, gens):
        parents = parent_selection(pop, fit_pop, n_offspring)
        offspring = crossover(parents)
        offspring = mutate(offspring, dom_l, dom_u, mutation)

        offspring_fit = evaluate(env, offspring)

        pop = np.vstack((pop, offspring))
        fit_pop = np.concatenate((fit_pop, offspring_fit))

        pop, fit_pop = survivor_selection(pop, fit_pop, npop)

        print(f"Gen {i} - Best: {np.max (fit_pop)} - Mean: {np.mean(fit_pop)}")

        results["mean"].append(np.mean(fit_pop))
        results["best"].append(np.max(fit_pop))
        results["std"].append(np.std(fit_pop))

    return results


def parent_selection(population, fitness_values, n_parents):
    # Parent selection based on rank based selection

    rankings = np.argsort(np.argsort(fitness_values))

    probability = (1 - rankings / len(population)) / np.sum(
        1 - rankings / len(population)
    )

    parent_indices = np.random.choice(
        np.arange(0, population.shape[0]), size=(n_parents, 2), p=probability
    )

    selected_parents = population[parent_indices]

    return selected_parents


def crossover(parents):
    # crossover based on random indices
    parentsA, parentsB = np.hsplit(parents, 2)

    random = np.random.uniform(size=parentsA.shape)

    offspring = np.where(random >= 0.5, parentsA, parentsB)
    return np.squeeze(offspring, 1)


def mutate(pop, minim, maxim, sigma):
    # creating of random mutation based on normal distribution of same size as population
    mutation = np.random.normal(0, sigma, size=pop.shape)
    # new population is created of both org_pop and newpop combined
    newpop = np.add(pop, mutation)
    # newpop values are not supposed to be outside of minim-maxim range:
    newpop = np.clip(newpop, minim, maxim)
    return newpop


def survivor_selection(population, fitness, total_survivors):
    # Input pop here is total population + offspring here
    # After calculating the fitness of the offspring before,
    # concatenate the offspring fitness to the initial population fitness --> total fitness
    # And determine how many survivors we want into the next generation

    sort_fitness = np.argsort(fitness * -1)
    top_survivors_index = sort_fitness[:total_survivors]
    survivors = population[top_survivors_index]

    return survivors, fitness[top_survivors_index]


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


if __name__ == "__main__":
    results = run()
