from main import state_space, action_space, hidden_size, agent_num, num_epochs, gamma, learning_rate, clip_epsilon, td_lambda, c1, c2, update_timestep, random_seed, episode_num, env_width, env_height, render_sleep_time
import PPO
from torch import load
import numpy as np
from numpy import inf
import time
import math
from flatland_model import Memory
from flatland.envs.rail_generators import complex_rail_generator, sparse_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator # Round 1
from flatland.envs.schedule_generators import sparse_schedule_generator # Round 2
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool



def learning_schedule(timestep, round, update_time, agent_num, episode_num, env_width, env_height):




	

	if round == 1:

		rail_generator=complex_rail_generator(
										nr_start_goal=9,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator=complex_schedule_generator()
	else:

		rail_generator=sparse_rail_generator(
										nr_start_goal=9,
										nr_extra=1,
										min_dist=9,
										max_dist=99999,
										seed=0)

		schedule_generator= sparse_schedule_generator()

	env = RailEnv(
	        width=env_width,
	        height=env_height,
	        rail_generator=rail_generator,
	        schedule_generator=schedule_generator,
	        number_of_agents=agent_num)


	return rail_generator, schedule_generator, env





def main():

	model = flatlandPPO(state_space=state_space,
				action_space=action_space, 
				hidden_size=hidden_size, 
				agent_num=agent_num, 
				num_epochs=num_epochs, 
				gamma=gamma, 
				learning_rate=learning_rate, 
				clip_epsilon=clip_epsilon, 
				td_lambda=td_lambda, 
				c1=c1, 
				c2=c2)


	model.model.load_state_dict(load('/home/vache/ML_projects/rl/flatland/saved_model/PPO_flatland.pth'))

	############################################################
	# Env stuff #
	rail_generator=complex_rail_generator(
											nr_start_goal=9,
											nr_extra=1,
											min_dist=9,
											max_dist=99999,
											seed=0)
	schedule_generator=complex_schedule_generator()
	env = RailEnv(
	        width=env_width,
	        height=env_height,
	        rail_generator=rail_generator,
	        schedule_generator=schedule_generator,
	        number_of_agents=agent_num)
	env_renderer = RenderTool(env)

	schedule_update_time = 200
	############################################################



	timesteps = 0


	rewards = {}
	for i in range(agent_num):
		rewards[i] = -1

	for n in range(episode_num):
		all_done = False

		states_dict = env.reset()
		states = list(states_dict.values())
		states = ppo.obs_into_right_shape(states)
		print('episode', n)

		if render:
			env_renderer.reset()

		while not all_done:
		# for a in range(50):
			timesteps += 1
			prev_rewards = rewards
			actions, values = ppo.step(obs=states, memory=memory, model=ppo.model_old, agent_num=agent_num)

			

			ppo.add_rewards_to_list(rewards)
			ppo.add_values_to_list(values.detach().numpy())

			states, rewards, done, _ = env.step(actions)
			states = ppo.obs_into_right_shape(list(states.values()))
			all_done = done['__all__']
			if render:
				env_renderer.render_env(show=True, frames=False, show_observations=False)
				time.sleep(render_sleep_time)

			if update_timestep - timesteps == 0:

				ppo.train(memory)
				memory.clear_memory()
				timesteps = 0



		save(ppo.model.state_dict(), '/home/vache/ML_projects/rl/flatland/saved_model/PPO_flatland.pth')





if __name__ == '__main__':
    main()


