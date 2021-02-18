from deap import base, creator, tools, algorithms
import random
import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import SAC
from algos.sac import core_cam
from algos.common import replay_buffer_cam
import numpy
from gym.wrappers.time_limit import TimeLimit
from utils.wrappers import TorchifyWrapper
import pickle


def rand_individual():
    depth = random.randint(1, 10)
    return {
        "batch_size": 2**random.randint(1, 12),
        "gamma": random.random(),
        "polyak": random.random(),
        "lr": random.uniform(0.0, 0.1),
        "start_steps": random.randint(1, 100000),
        "num_updates": 2**random.randint(0, 12),
        "nn_width": 2**random.randint(0, 12),
        "nn_depth": random.randint(1, 10),
        "conv_depth": random.randint(1, 10),
        "conv_channels": 2**random.randint(0, 8),
        "conv_kernel": random.randint(1, 32),
        "conv_stride": random.randint(1, 32),
        "feature_dim": 2**random.randint(1, 12)
    }


def rand_ind_ctor(ctor):
    return ctor(rand_individual())


creator.create("Fitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", dict, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("individual", rand_ind_ctor, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def eval(individual):
    env = TorchifyWrapper(
        TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
                  max_episode_steps=100))

    n_channel = 1
    conv_lrs = []

    for _ in range(individual["conv_depth"]):
        conv_lrs.append(
            (n_channel, individual["conv_channels"], individual["conv_kernel"],
             individual["conv_stride"], 0))
        n_channel = individual["conv_channels"]

    ac_kwargs = dict(hidden_sizes=[
        individual["nn_width"] for _ in range(individual["nn_depth"])
    ],
                     activation=nn.ReLU,
                     conv_sizes=conv_lrs,
                     feature_dim=individual["feature_dim"])
    rb_kwargs = dict(size=1000)

    logger_kwargs = dict(output_dir='data/hyper_train_0',
                         exp_name='hyper_train_0')

    ret = 0.0

    try:
        model = SAC(env=env,
                    actor_critic=core_cam.MLPActorCritic,
                    ac_kwargs=ac_kwargs,
                    replay_buffer=replay_buffer_cam.ReplayBuffer,
                    rb_kwargs=rb_kwargs,
                    max_ep_len=100,
                    batch_size=individual["batch_size"],
                    gamma=individual["gamma"],
                    lr=individual["lr"],
                    polyak=individual["polyak"],
                    start_steps=individual["start_steps"],
                    update_after=individual["batch_size"],
                    update_every=1,
                    num_updates=individual["num_updates"],
                    logger_kwargs=logger_kwargs)

        ret = model.train(steps_per_epoch=10000,
                          epochs=500,
                          stop_return=0.8,
                          abort_after_epoch=50,
                          abort_return_threshold=0.2)
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


def main():
    NGEN = 20
    MU = 10
    LAMBDA = 20
    CXPB = 0.5
    MUTPB = 0.5

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = algorithms.eaMuPlusLambda(pop,
                              toolbox,
                              MU,
                              LAMBDA,
                              CXPB,
                              MUTPB,
                              NGEN,
                              stats,
                              halloffame=hof,
                              verbose=True)

    with open("data/hyper_train_0_logbook.pickle", "wb") as output_file:
        pickle.dump(logbook, output_file)

    return pop, stats, hof


if __name__ == "__main__":
    pop, stats, hof = main()
