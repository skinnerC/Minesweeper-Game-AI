import random
import math
import pygame

import game
import config

from ANN import ANN

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy

def runGA(visable=True):
    # Read your ANN structure from "config.py":
    num_inputs = config.nnet['n_inputs']
    num_hidden_nodes = config.nnet['n_h_neurons']
    num_outputs = config.nnet['n_outputs']

    my_game = game.Game()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Prepare your individuals below.
    # Let's assume that you have a one-hidden layer neural network with 2 hidden nodes:
    # You would need to define a list of floating numbers of size: 16 (10+6)
    toolbox.register("attr_real", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, n=((num_inputs+1)*num_hidden_nodes)+((num_hidden_nodes+1)*num_outputs))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=config.game['n_agents'])

    # Fitness Evaluation:
    def evalANN(individual):
        return my_game.get_ind_fitness(individual),
        # comma at the end is necessarys since DEAP stores fitness values as a tuple

    toolbox.register("evaluate", evalANN)

    # Define your selection, crossover and mutation operators below:
    toolbox.register("mate", tools.cxBlend, alpha=0.05)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("average", numpy.mean)
    stats.register("standard dev", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()

    # Define EA parameters: n_gen, pop_size, prob_xover, prob_mut:
    # You can define them in the "config.py" file too.
    CXPB, MUTPB, NGEN = 0.5, 0.2, 300

    pop = toolbox.population()

    # Create initial population (each individual represents an agent or ANN):
    for ind in pop:
        # ind (individual) corresponds to the list of weights
        # ANN class is initialized with ANN parameters and the list of weights
        ann = ANN(num_inputs, num_hidden_nodes, num_outputs, ind)
        my_game.add_agent(ann)

    # Let's evaluate the fitness of each individual.
    # First, simulation should be run!
    my_game.game_loop(visable) # Set it to "False" for headless mode;
    #recommended for training, otherwise learning process will be very slow!
        
    # Let's collect the fitness values from the simulation using
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=my_game.generation, **record)

    for g in range(1, NGEN):
        my_game.generation += 1
        my_game.reset()#
        
        # Start creating the children (or offspring)
            
        # First, Apply selection:
        offspring = toolbox.select(pop, len(pop))
            
        # Apply variations (xover and mutation), Ex: algorithms.varAnd(?, ?, ?, ?)
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

        # Repeat the process of fitness evaluation below. You need to put the recently
        # created offspring-ANN's into the game (Line 55-69) and extract their fitness values:
        for ind in offspring:
            # ind (individual) corresponds to the list of weights
            # ANN class is initialized with ANN parameters and the list of weights
            ann = ANN(num_inputs, num_hidden_nodes, num_outputs, ind)
            my_game.add_agent(ann)

        # Let's evaluate the fitness of each individual.
        # First, simulation should be run!
        my_game.game_loop(visable) # Set it to "False" for headless mode;
        #recommended for training, otherwise learning process will be very slow!
        
        # Let's collect the fitness values from the simulation using
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # One way of implementing elitism is to combine parents and children to give them equal chance to compete:
        # For example: pop[:] = pop + offspring
        # Otherwise you can select the parents of the generation from the offspring population only: pop[:] = offspring
        offspring.remove(tools.selRoulette(offspring, 1)[0])
        pop[:] = offspring + tools.selBest(pop, 1)

        record = stats.compile(pop)
        logbook.record(gen=my_game.generation, **record)
        # This is the end of the "for" loop (end of generations!)

    logbook.header = "gen", "average", "standard dev", "max", "min"
    return [logbook, tools.selBest(pop, 1)]
    ##print "Training is over"    
    ###raw_input("Training is over!")
    ##while True:
    ##    my_game.game_loop(True)
    ##
    ##    
    pygame.quit()

def runGame(num_inputs, num_hidden_nodes, num_outputs, weights):
##    num_inputs = config.nnet['n_inputs']
##    num_hidden_nodes = config.nnet['n_h_neurons']
##    num_outputs = config.nnet['n_outputs']

    my_game = game.Game()
    for i in range(config.game['n_agents']):
        ann = ANN(num_inputs, num_hidden_nodes, num_outputs, weights)
        my_game.add_agent(ann)

    while True:
        my_game.game_loop(True)
        
    pygame.quit()

if __name__ == '__main__':
##    f = open("results.txt", "w")
##    for i in range(1,11):
##        config.nnet['n_h_neurons'] = i
##        f.write("{0} Hidden Neurons\n".format(i))
##        results = runGA()
##        f.write(str(results[1]))
##        f.write("\n")
##        f.write(str(results[0]))
##        f.write("\n\n")
##    f.close()
    #config.nnet['n_h_neurons'] = 4
    #runGA()
    weights4 = [1.1331724393742246, 0.37224833425661313, 0.6246465943644157, 0.6427665562659374, -0.22932494908600254, 0.3504085161755975, 0.1445675256926817, 0.9280206545739222, 0.15481308538989424, 0.6349608860704619, 0.6844657437657429, 0.300862828209683, 0.1948862825517816, 0.3487161381822424, 0.8247271738876739, 0.526904986046438, 0.9439522689686565, 0.3246673221631321, 0.5720346244169912, 0.8403163994516007, 0.9188187953509087, 0.4891582848545075, 0.5950852564441498, 0.825960515604923, 0.17824047635879595, 0.601636768168944, 0.19679507746791525, 0.944727951875811, 0.8219959368917009, 0.17945051351207075]
    runGame(4, 4, 2, weights4)
    
    
