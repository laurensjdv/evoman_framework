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




def main():
    # choose this for not using visuals and thus making experiments faster
    headless = False
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=True)


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here

    env.state_to_log() # checks environment state

    run_mode = 'train'

    dom_u = 1
    dom_l = -1
    npop = 100
    gens = 30
    mutation = 0.2
    last_best = 0

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    print(pop.shape)
    fit_pop = evaluate(env, pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

    print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))

    last_sol = fit_pop[best]
    notimproved = 0

    # for i in range(ini_g+1, gens):
    #     parents = parent_selection(pop, fit_pop, 2)
    #     offspring = crossover()
    


    


def parent_selection(pop, pop_fit, n_parents, smoothing = 1):
    fitness  = pop_fit + smoothing - np.min(pop_fit)

    # Fitness proportional selection probability
    fps = fitness / np.sum (fitness)
    
    # make a random selection of indices
    parent_indices = np.random.choice (np.arange(0,pop.shape[0]), (n_parents,2), p=fps)
    return pop [parent_indices]

def crossover(parents):
    parentsA, parentsB = np.hsplit (parents,2)
    roll = np.random.uniform (size = parentsA.shape)
    offspring = parentsA * (roll >= 0.5) + parentsB * (roll < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return np.squeeze(offspring,1)

def mutate(pop,min_value,max_value, sigma):
    mutation = np.random.normal(0, sigma, size=pop.shape)
    new_pop = pop + mutation
    new_pop[new_pop>max_value] = max_value
    new_pop[new_pop<min_value] = min_value
    return new_pop

def survivor_selection(pop, pop_fit, n_pop):
    best_fit_indices = np.argsort(pop_fit * -1) # -1 since we are maximizing
    survivor_indices = best_fit_indices [:n_pop]
    return pop [survivor_indices], pop_fit[survivor_indices]



# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
        return np.array(list(map(lambda y: simulation(env,y), x)))



if __name__ == '__main__':
    main()