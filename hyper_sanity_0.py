from deap import base, creator, tools, algorithms
import random
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import SAC
from algos.sac import core
from algos.common import replay_buffer
import numpy
from scoop import futures


def rand_individual():
    depth = random.randint(1, 10)
    return {
        "batch_size": 2**random.randint(1, 12),
        "gamma": random.random(),
        "polyak": random.random(),
        "lr": random.uniform(0.0, 0.1),
        "start_steps": random.randint(1, 100000),
        # "update_every": 2**random.randint(1, 12),
        "num_updates": 2**random.randint(0, 12),
        "nn_width": [2**random.randint(0, 12) for _ in range(depth)],
        "nn_depth": depth,
        # "conv_sizes":
        # [random.randint(1, 128)] + random.sample(range(1, 11), 3),
        # "feature_dim": 2**random.randint(1, 12)
    }


def rand_ind_ctor(ctor):
    return ctor(rand_individual())


creator.create("Fitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", dict, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("individual", rand_ind_ctor, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval(individual):
    env = gym.make('Pendulum-v0')

    ac_kwargs = dict(hidden_sizes=[
        individual["nn_width"][i] for i in range(individual["nn_depth"])
    ],
                     activation=nn.ReLU)
    rb_kwargs = dict(size=1000000)

    logger_kwargs = dict(output_dir='data/hyper_sanity_0', exp_name='hyper_sanity_0')

    model = SAC(env=env,
                actor_critic=core.MLPActorCritic,
                ac_kwargs=ac_kwargs,
                replay_buffer=replay_buffer.ReplayBuffer,
                rb_kwargs=rb_kwargs,
                max_ep_len=1000,
                batch_size=individual["batch_size"],
                gamma=individual["gamma"],
                lr=individual["lr"],
                polyak=individual["polyak"],
                start_steps=individual["start_steps"],
                update_after=individual["batch_size"],
                update_every=1,
                num_updates=individual["num_updates"],
                logger_kwargs=logger_kwargs)

    ret = 0.0

    try:
        ret = model.train(steps_per_epoch=1000, epochs=10)
    except:
        pass

    return ret,


def cx(ind1, ind2):
    # swap random values
    for k in ind1:
        if random.random() < 0.5:
            ind1[k], ind2[k] = ind2[k], ind1[k]

    return ind1, ind2


def mut(individual):
    mut_n = random.randint(1, 3)
    mut_keys = random.sample(individual.keys(), mut_n)
    rand_ind = rand_individual()

    for k in mut_keys:
        individual[k] = rand_ind[k]

    return individual,


toolbox.register("evaluate", eval)
toolbox.register("mate", cx)
toolbox.register("mutate", mut)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)


def main():
    NGEN = 10
    MU = 10
    LAMBDA = 20
    CXPB = 0.7
    MUTPB = 0.2

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop,
                              toolbox,
                              MU,
                              LAMBDA,
                              CXPB,
                              MUTPB,
                              NGEN,
                              stats,
                              halloffame=hof)

    return pop, stats, hof


if __name__ == "__main__":
    main()
