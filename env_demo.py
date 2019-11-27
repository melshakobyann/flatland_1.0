import numpy as np
from numpy import inf
import time
import math
import torch
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool
from baselines.utils.observation_utils import norm_obs_clip, split_tree

NUMBER_OF_AGENTS = 2
env = RailEnv(
            width=20,
            height=20,
            rail_generator=complex_rail_generator(
                                    nr_start_goal=9,
                                    nr_extra=1,
                                    min_dist=9,
                                    max_dist=99999,
                                    seed=0),
            schedule_generator=complex_schedule_generator(),
            obs_builder_object=GlobalObsForRailEnv(),
            number_of_agents=NUMBER_OF_AGENTS)

env_renderer = RenderTool(env)

def my_controller():
    """
    You are supposed to write this controller
    """
    _action = {}
    for _idx in range(NUMBER_OF_AGENTS):
        _action[_idx] = np.random.randint(0, 5)
    return _action

done = False
iteration = 0
# while not done:
for i in range(5):

    iteration += 1
    _action = my_controller()
    # print(_action)
    obs, all_rewards, done, _ = env.step(_action)
    new_obs1 = torch.tensor(obs[0][0])
    new_obs2 = torch.tensor(obs[0][1])
    new_obs3 = torch.tensor(obs[0][2])

    # new_obs1 = new_obs1.view(-1, env.height, env.width)
    # new_obs2 = new_obs2.view(-1, env.height, env.width)
    # new_obs3 = new_obs3.view(-1, env.height, env.width)

    # stacked = torch.stack([new_obs1, new_obs2, new_obs3])

    print(new_obs2.view(-1, 20, 20)[3])
    # rail_data, distance_data, agent_data = split_tree(tree=np.array(obs[0]),
    #                                                       num_features_per_node=8,
    #                                                       current_depth=0)

    # tmp_obs = np.array(obs[0])
    # tmp_obs[tmp_obs==-np.inf] = -10
    # tmp_obs[tmp_obs==np.inf] = -10

    # print("Rewards: {}, [done={}], obs_len={}, rail_data={}, dist_data={}, agent_data={}, changed_obs={}, split_tree_obs_len={}".format(all_rewards, done, len(obs[0]), len(rail_data), len(distance_data), len(agent_data), tmp_obs, len(rail_data)+len(distance_data)+len(agent_data)))
    
    # uncomment this in evaluation
    env_renderer.render_env(show=True, frames=False, show_observations=False)
    
    done = done['__all__']
    if iteration >= 50:
        done = True

    time.sleep(0.3)