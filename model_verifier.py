
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import wandb

import gymnasium as gym
from a2c import *
from finder_gym import *
from alg_parameters import *
from map_tester import *

best_actor2 = "./models/actor_prototype2.2scratch-20240218-120124"
best_actor1 = "./models/actor_prototype2.2scratch-20240219-103110"
actor_file_name = "./models/actor_prototype2.5-20240223-030259"

# environment hyperparams
n_env = 1
n_updates = 1000
n_steps_per_update = 2080
randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
actor_lr = 0.001
critic_lr = 0.005

def create_mask(env):
    return np.equal(env, 2)
    #return (env == 2)

def prune_nodes(env):

    world = env.world
    x_len = len(world)
    y_len = len(world[0])

    new_world = world.copy()

    #prune nodes with priority on nodes adjacent to multiple nodes
    for i in range(3,0,-1):
        for index, value in np.ndenumerate(world):
            x = index[0]
            y = index[1]
            #find nodes to prune
            if value == 1:
                count = 0
                if(x > 0 and world[x-1][y] == 1):
                    count += 1
                if(x < x_len-1 and world[x+1][y] == 1):
                    count += 1
                if(y > 0 and world[x][y-1] == 1):
                    count += 1
                if(y < y_len-1 and world[x][y+1] == 1):
                    count += 1

                if(x > 0 and y > 0 and world[x-1][y-1] == 1):
                    count += 0.5
                if(x < x_len-1 and y > 0 and world[x+1][y-1] == 1):
                    count += 1
                if(x > 0 and y < y_len-1 and world[x-1][y+1] == 1):
                    count += 1
                if(x < x_len-1 and y < y_len-1 and world[x+1][y+1] == 1):
                    count += 1

                if (count >= i):
                    world[x][y] = 2
                    env.world = world
                    env.render()
            
            #for first loop only
            #only copy obstacles for the new world
            if i == 3:
                if world[x][y] != -1:
                    new_world[x][y] = 0
            
    env.world = world
    env.render()

    #raycast again on new world to check visibility on pruned nodes
    for index, value in np.ndenumerate(world):
        x = index[0]
        y = index[1]

        if world[x][y] == 1:
            _, new_world, visible_nodes_count = ray_cast(new_world, (x,y), fog_dist, True)

        env.world = new_world
        env.render()
        
    pygame.time.wait(2000)
    #check and eliminate any blindspots by adding extra node
    for index, value in np.ndenumerate(new_world):
        x = index[0]
        y = index[1]

        if new_world[x][y] == 0:
            result = -1

            dx = -1
            dy = -1
            state = 0

            while(result != 1):
                if(state == 0):
                    if(x+dx > 0 and x+dx < len(new_world)-1):
                        result, new_world, visible_nodes = ray_cast(new_world, (x+dx,y), fog_dist, False)
                    state = 1
                
                elif(state == 1):
                    if(y+dy > 0 and y+dy < len(new_world[0])-1):
                        result, new_world, visible_nodes = ray_cast(new_world, (x,y+dy), fog_dist, False)
                    state = 2
                
                elif(state == 2):
                    if(x+dx > 0 and x+dx < len(new_world)-1 and y+dy > 0 and y+dy < len(new_world[0])-1):
                        result, new_world, visible_nodes = ray_cast(new_world, (x+dx,y+dy), fog_dist, False)

                    #increment dx and dy (while alternating signs)
                    if(dx < 0):
                        dx = -dx
                    else:
                        dx = -dx - 1
                    if(dy < 0):
                        dy = -dy
                    else:
                        dy = -dy - 1
                
                    state = 0
                
                env.world = new_world
                env.render()
            

env = nodeFindEnv()

obs_shape = env.observation_space.shape
action_shape = env.action_space.n

# set the device
use_cuda = True
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_env)

agent.actor.load_state_dict(torch.load(actor_file_name))

while True:
    states, info = env.reset()
    terminated = False
    while not terminated:

        # select an action A_{t} using S_{t} as input for the agent
        actions, action_log_probs, state_value_preds, entropy = agent.select_action(
            states, mask=create_mask(states)
        )

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        states, rewards, terminated, truncated, infos = env.step(
            actions.cpu().numpy()
        )

        pygame.event.get()
        env.render()

    #Pruning function
    prune_nodes(env)

    print("complete")
    input("Press Enter to continue...")


