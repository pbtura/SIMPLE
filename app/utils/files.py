

import os
import sys
import random
import csv
import time
import numpy as np
import app.config as Config

from mpi4py import MPI

from shutil import rmtree
from stable_baselines3.ppo import PPO

from app.utils.register import get_network_arch


from stable_baselines3.common import logger

def write_results(players, game, games, episode_length):
    
    out = {'game': game
    , 'games': games
    , 'episode_length': episode_length
    , 'p1': players[0].name
    , 'p2': players[1].name
    , 'p1_points': players[0].points
    , 'p2_points': np.sum([x.points for x in players[1:]])
    }

    if not os.path.exists(Config.RESULTSPATH):
        with open(Config.RESULTSPATH,'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out.keys())
            writer.writeheader()

    with open(Config.RESULTSPATH,'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out.keys())
        writer.writerow(out)


def load_model(env, name) -> PPO:

    filename = os.path.join(Config.MODELDIR, env.name, name)
    if os.path.exists(filename):
        Config.logger.info(f'Loading {name}')
        cont = True
        while cont:
            try:
                ppo_model = PPO.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)
    
    elif name == 'base.zip':
        cont = True
        while cont:
            try:
                
                rank = MPI.COMM_WORLD.Get_rank()
                if rank == 0:
                    ppo_model = PPO(get_network_arch(env.name), env=env, device="cuda")
                    Config.logger.info(f'Saving base.zip PPO model...')
                    ppo_model.save(os.path.join(Config.MODELDIR, env.name, 'base.zip'))
                else:

                    ppo_model = PPO.load(os.path.join(Config.MODELDIR, env.name, 'base.zip'), env=env)

                cont = False
            except IOError as e:
                sys.exit(f'Check zoo/{env.name}/ exists and read/write permission granted to user')
            except Exception as e:
                Config.logger.error(e)
                time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model


def load_all_models(env):
    modellist = [f for f in os.listdir(os.path.join(Config.MODELDIR, env.name)) if f.startswith("_model")]
    modellist.sort()
    models = [load_model(env, 'base.zip')]
    for model_name in modellist:
        models.append(load_model(env, name = model_name))
    return models


def get_best_model_name(env_name):
    modellist = [f for f in os.listdir(os.path.join(Config.MODELDIR, env_name)) if f.startswith("_model")]
    
    if len(modellist)==0:
        filename = None
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename

def get_model_stats(filename):
    if filename is None:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        stats = filename.split('_')
        generation = int(stats[2])
        best_rules_based = float(stats[3])
        best_reward = float(stats[4])
        timesteps = int(stats[5])
    return generation, timesteps, best_rules_based, best_reward


def reset_logs(model_dir):
    try:
        filelist = [ f for f in os.listdir(Config.LOGDIR) if f not in ['.gitignore']]
        for f in filelist:
            if os.path.isfile(f):  
                os.remove(os.path.join(Config.LOGDIR, f))

        for i in range(100):
            if os.path.exists(os.path.join(Config.LOGDIR, f'tb_{i}')):
                rmtree(os.path.join(Config.LOGDIR, f'tb_{i}'))
        
        open(os.path.join(Config.LOGDIR, 'log.txt'), 'a').close()
    
        
    except Exception as e :
        print(e)
        print('Reset logs failed')

def reset_models(model_dir):
    try:
        filelist = [ f for f in os.listdir(model_dir) if f not in ['.gitignore']]
        for f in filelist:
            os.remove(os.path.join(model_dir , f))
    except Exception as e :
        print(e)
        print('Reset models failed')