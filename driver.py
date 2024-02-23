from __future__ import annotations

import os

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

# Note: the actor has a slower learning rate so that the value targets become
# more stationary and are theirfore easier to estimate for the critic


# #TODO: GENERATE WORLD
# # prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1],
# #                             self.PROB[1])  # sample a value from triangular distribution
# # size = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]],
# #                         p=[.5, .25, .25])  # sample a value according to the given probability
# prob = 0.4
# size = 20
# # prob = self.PROB
# # size = self.SIZE  # fixed world0 size and obstacle density for evaluation
# # here is the map without any agents nor goals
# world = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
# #for PRIMAL1 map
# # world = random_generator(SIZE_O=self.SIZE, PROB_O=self.PROB)
# world = padding_world(world)

# #add starting node
# for index, x in np.ndenumerate(world):
#     if x == 0:
#         start_pos = index
#         world[index[0]][index[1]] = 1
#         break

# environment setup

def create_mask(env):
    return np.equal(env, 2)
    #return (env == 2)


env = nodeFindEnv()


obs_shape = env.observation_space.shape
action_shape = env.action_space.n

# set the device
use_cuda = True
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# init the agent
agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_env)

# #call to reload saved actor critic data
agent.actor.load_state_dict(torch.load("./models/actor_prototype2.5-20240223-030259"))
agent.critic.load_state_dict(torch.load("./models/critic_prototype2.5-20240223-030259"))

# create a wrapper environment to save episode returns and episode lengths
env_wrapper = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_env * n_updates)

wandb_id = wandb.util.generate_id()
wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
            name=RecordingParameters.EXPERIMENT_NAME,
            entity=RecordingParameters.ENTITY,
            notes=RecordingParameters.EXPERIMENT_NOTE,
            config=all_args,
            id=wandb_id,
            resume='allow')
print('id is:{}'.format(wandb_id))
print('Launching wandb...\n')

while True:

    # use tqdm to get a progress bar for training

    image_Saved = False
    for sample_phase in tqdm(range(n_updates)):
        # we don't have to reset the env, they just continue playing
        # until the episode is over and then reset automatically

        critic_losses = []
        actor_losses = []
        entropies = []

        # reset lists that collect experiences of an episode (sample phase)
        num_invalid_move = 0
        ep_length = []
        ep_value_preds = torch.zeros(n_steps_per_update, n_env, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_env, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, n_env, device=device)
        masks = torch.zeros(n_steps_per_update, n_env, device=device)

        # at the start of training reset all env to get an initial state
        if sample_phase == 0:
            states, info = env_wrapper.reset()

        image_Saved = False
        # play n steps in our parallel environments to collect data
        for step in range(n_steps_per_update):
            # select an action A_{t} using S_{t} as input for the agent
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states, mask=create_mask(states)
            )

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            states, rewards, terminated, truncated, infos = env_wrapper.step(
                actions.cpu().numpy()
            )

            #count number of invalid moves
            if(infos['isValid'] != 1):
                num_invalid_move += 1


            if("episode" in infos):
                ep_length.append(infos["episode"]["l"])

            pygame.event.get()
            env.render()

            if(terminated):

                #save image once every training cycle (2080 steps)
                if((not image_Saved)and step > n_steps_per_update/2):
                    image_path = "/home/marmot/patrick/ALPHA/node_placer/complete_map/"
                    image_name = RecordingParameters.EXPERIMENT_NAME + "/" + time.strftime("%Y%m%d-%H%M%S") + ".jpeg"
                    pygame.image.save(env.screen, image_path + image_name)

                    #comment below to always save at end of every ep
                    image_Saved = True 
                env_wrapper.reset()

            # #save image aevery 200 steps
            # if(step % 200 == 0 and step > 0):
            #     image_Saved = False

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, device=device)
            ep_action_log_probs[step] = action_log_probs

            # add a mask (for the return calculation later);
            # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            masks[step] = torch.tensor([not terminated])

            #pygame.time.wait(2)

        # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        #gradient norm
        gradient_norm = nn.utils.clip_grad_norm_(agent.actor.parameters(), TrainingParameters.MAX_GRAD_NORM)

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())


        wandb.log({'Perf_random_eval/Num_Invalid_Moves': num_invalid_move})
        wandb.log({'Perf_random_eval/Episode_length': np.mean(np.array(ep_length))})
        wandb.log({'Perf_random_eval/Rewards': torch.mean(ep_rewards)})
        # print(actor_losses)
        # print(critic_losses)
        wandb.log({'Perf_random_eval/Actor Loss': actor_losses[0]})
        wandb.log({'Perf_random_eval/Critic Loss': critic_losses[0]})
        wandb.log({'Perf_random_eval/Actor Entropy': entropies[0]})
        wandb.log({'Perf_random_eval/Actor_Grad': gradient_norm})

        # wandb.log({'Perf_random_eval/Num_block': performance_dict['per_block']}, step=step)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    critic_file_name = "./models/critic_" + RecordingParameters.EXPERIMENT_NAME + "-" + timestr
    actor_file_name = "./models/actor_" + RecordingParameters.EXPERIMENT_NAME + "-" + timestr


    torch.save(agent.critic.state_dict(), critic_file_name)
    torch.save(agent.actor.state_dict(), actor_file_name)
        # pygame.event.get()
        # env.render()
        # pygame.time.wait(200)

    ####recording line

    pygame.event.get()
    # action = env.action_space.sample()  # Random action selection
    # print('Reward:', rewards)
    # print('critic loss:', critic_loss)

    pygame.time.wait(200)

""" plot the results """

# %matplotlib inline

# rolling_length = 20
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
# fig.suptitle(
#     f"Training plots for {agent.__class__.__name__} in the LunarLander-v2 environment \n \
#             (n_env={n_env}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
# )

# # episode return
# axs[0][0].set_title("Episode Returns")
# episode_returns_moving_average = (
#     np.convolve(
#         np.array(env_wrapper.return_queue).flatten(),
#         np.ones(rolling_length),
#         mode="valid",
#     )
#     / rolling_length
# )
# axs[0][0].plot(
#     np.arange(len(episode_returns_moving_average)) / n_env,
#     episode_returns_moving_average,
# )
# axs[0][0].set_xlabel("Number of episodes")

# # entropy
# axs[1][0].set_title("Entropy")
# entropy_moving_average = (
#     np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
#     / rolling_length
# )
# axs[1][0].plot(entropy_moving_average)
# axs[1][0].set_xlabel("Number of updates")


# # critic loss
# axs[0][1].set_title("Critic Loss")
# critic_losses_moving_average = (
#     np.convolve(
#         np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
#     )
#     / rolling_length
# )
# axs[0][1].plot(critic_losses_moving_average)
# axs[0][1].set_xlabel("Number of updates")


# # actor loss
# axs[1][1].set_title("Actor Loss")
# actor_losses_moving_average = (
#     np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
#     / rolling_length
# )
# axs[1][1].plot(actor_losses_moving_average)
# axs[1][1].set_xlabel("Number of updates")

# plt.tight_layout()
# plt.show()


