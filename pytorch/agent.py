import os
import json
import time
from itertools import count
import numpy as np
from pytorch.memory import ReplayMemory, Transition
from pytorch.config import Config
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

is_notebook = 'inline' in matplotlib.get_backend()

if is_notebook:
    from IPython import display
    from tqdm.notebook import trange
else:
    from tqdm import trange


class QAgent():
    def __init__(self, env, config, model_class):
        self.env = env
        self.model_dir = Path('./models')
        self.model_class = model_class
        self.config_file = config
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.episode_durations = []
        self.config = Config(self.config_file)
        self.memory = None
        self.rng = np.random.default_rng(42)
        self.episode_durations = []
        self.reward_in_episode = []
        self.epsilon_vec = []
        self.loss = None
        self.last_step = 0
        self.last_episode = 0
        self.id = int(time.time())

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_optimizer(self):
        try:
            if self.config.optimizer.lower() == "adam":
                return optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
            elif self.config.optimizer.lower() == "adadelta":
                return optim.Adadelta(self.model.parameters(), lr=self.config.training.learning_rate)
            elif self.config.optimizer.lower() == "rms":
                return optim.RMSprop(self.model.parameters(), lr=self.config.training.learning_rate)
            else:
                return optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)
        except AttributeError:
            return optim.Adam(self.model.parameters(), lr=self.config.training.learning_rate)

    def compile(self):
        n_actions = self.env.action_space.n

        self.model = self.model_class(n_actions).to(self.device)
        self.target_model = self.model_class(n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = self._get_optimizer()
        
    def _get_epsilon(self, episode):
        epsilon = self.config.epsilon.min_epsilon + \
                          (self.config.epsilon.max_epsilon - self.config.epsilon.min_epsilon) * \
                              np.exp(-episode / self.config.epsilon.decay_epsilon)
        return epsilon

    def _get_action_for_state(self, state):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            predicted = self.model(torch.tensor([state], device=self.device))
            action = predicted.max(1)[1]
        return action.item()
        
    def _choose_action(self, state, epsilon):
        if self.rng.uniform() < epsilon:
            # Explore
            action = self.env.action_space.sample()
        else:
            # Exploit
            action = self._get_action_for_state(state)
        return action
    
    def _adjust_learning_rate(self, episode):
        delta = self.config.training.learning_rate - self.config.optimizer.lr_min
        base = self.config.optimizer.lr_min
        rate = self.config.optimizer.lr_decay
        lr = base + delta * np.exp(-episode / rate)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _train_model(self):
        if len(self.memory) < self.config.training.batch_size:
            return
        transitions = self.memory.sample(self.config.training.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # >>> zip(*[('a', 1), ('b', 2), ('c', 3)]) === zip(('a', 1), ('b', 2), ('c', 3))
        # [('a', 'b', 'c'), (1, 2, 3)]
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Compute predicted Q values
        predicted_q_value = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute the expected Q values
        next_state_values= self.target_model(next_state_batch).max(1)[0]
        expected_q_values = (~done_batch * next_state_values * self.config.rl.gamma) + reward_batch

        # Compute loss
        loss = self.loss(predicted_q_value, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def _remember(self, state, action, next_state, reward, done):
        self.memory.push(torch.tensor([state], device=self.device),
                        torch.tensor([action], device=self.device, dtype=torch.long),
                        torch.tensor([next_state], device=self.device),
                        torch.tensor([reward], device=self.device),
                        torch.tensor([done], device=self.device, dtype=torch.bool))

    def _get_loss(self):
        try:
            if self.config.training.loss.lower() == "huber":
                return F.smooth_l1_loss
            elif self.config.training.loss.lower() == "mse":
                return F.mse_loss
            else:
                return F.smooth_l1_loss
        except AttributeError:
            return F.smooth_l1_loss

    def fit(self):
        try:
            self.config = Config(self.config_file)            
            self.loss = self._get_loss()
            self.memory = ReplayMemory(self.config.rl.max_queue_length)

            self.episode_durations = []
            self.reward_in_episode = []
            self.epsilon_vec = []
            reward_in_episode = 0
            epsilon = 1

            progress_bar = trange(0,
                                  self.config.training.num_episodes, 
                                  initial=self.last_episode,
                                  total=self.config.training.num_episodes)

            for i_episode in progress_bar:
                # Initialize the environment and state
                state = self.env.reset()
                if i_episode >= self.config.training.warmup_episode:
                    epsilon = self._get_epsilon(i_episode - self.config.training.warmup_episode)

                for step in count():
                    # Select and perform an action
                    action = self._choose_action(state, epsilon)
                    next_state, reward, done, _ = self.env.step(action)

                    # Store the transition in memory
                    self._remember(state, action, next_state, reward, done)

                    # Perform one step of the optimization (on the target network)
                    if i_episode >= self.config.training.warmup_episode:
                        self._train_model()
                        self._adjust_learning_rate(i_episode - self.config.training.warmup_episode + 1)
                        done = (step == self.config.rl.max_steps_per_episode - 1) or done
                    else:
                        done = (step == 5 * self.config.rl.max_steps_per_episode - 1) or done

                    # Move to the next state
                    state = next_state
                    reward_in_episode += reward

                    if done:
                        self.episode_durations.append(step + 1)
                        self.reward_in_episode.append(reward_in_episode)
                        self.epsilon_vec.append(epsilon)
                        reward_in_episode = 0
                        N = min(10, len(self.episode_durations))
                        progress_bar.set_postfix({
                            "reward": np.mean(self.reward_in_episode[-N:]),
                            "steps": np.mean(self.episode_durations[-N:]),
                            "epsilon": epsilon
                            })
                        self.plot_durations()
                        break

                # Update the target network, copying all weights and biases in DQN
                if i_episode % self.config.rl.target_model_update_episodes == 0:
                    self._update_target()

                if i_episode % self.config.training.save_freq == 0:
                    self.save()

                self.last_episode = i_episode

        except KeyboardInterrupt:
            self.plot_durations()
            print("Training has been interrupted")

    def fit2(self):
        try:
            self.config = Config(self.config_file)
            self.memory = ReplayMemory(self.config.rl.max_queue_length)

            self.episode_durations = []
            self.reward_in_episode = []
            self.epsilon_vec = []
            current_episode = 0
            reward_in_episode = 0
            steps_in_episode = 0

            progress_bar = trange(0,
                                  self.config.training.train_steps, 
                                  initial=self.last_step,
                                  total=self.config.training.train_steps)
            
            # Initialize the environment and state
            state = self.env.reset()
            # Do not use model for warmup
            epsilon = 1

            for step in progress_bar:
                # Select and perform an action
                action = self._choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                # Store the transition in memory
                self._remember(state, action, next_state, reward, done)

                if current_episode >= self.config.training.warmup_episode:
                    epsilon = self._get_epsilon(current_episode - self.config.training.warmup_episode)
                    self._train_model()
                    done = steps_in_episode == self.config.rl.max_steps_per_episode or done
                else:
                    done = steps_in_episode == 5 * self.config.rl.max_steps_per_episode or done

                # Move to the next state
                state = next_state
                reward_in_episode += reward
                steps_in_episode += 1

                if done:
                    state = self.env.reset()
                    current_episode += 1
                    self.reward_in_episode.append(reward_in_episode)
                    self.episode_durations.append(steps_in_episode)
                    self.epsilon_vec.append(epsilon)
                    steps_in_episode = 0
                    reward_in_episode = 0
                    N = min(10, len(self.episode_durations))
                    progress_bar.set_postfix({
                        "episode": current_episode,
                        "reward": np.mean(self.reward_in_episode[-N:]),
                        "steps": np.mean(self.episode_durations[-N:]),
                        "epsilon": epsilon
                        })
                    self.plot_durations()

                # Update the target network, copying all weights and biases in DQN
                if current_episode % self.config.rl.target_model_update_episodes == 0:
                    self._update_target()

                self.last_step = step

        except KeyboardInterrupt:
            self.plot_durations()
            print("Training has been interrupted")

    @staticmethod
    def _moving_average(x, periods=5):
        if len(x) < periods:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        res = (cumsum[periods:] - cumsum[:-periods]) / periods
        return np.hstack([x[:periods-1], res])

    def plot_durations(self):
        lines = []
        fig = plt.figure(1, figsize=(15, 7))
        plt.clf()
        ax1 = fig.add_subplot(111)

        plt.title('Training...')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration & Rewards')
        ax1.set_ylim(-2 * self.config.rl.max_steps_per_episode, self.config.rl.max_steps_per_episode + 10)
        ax1.plot(self.episode_durations, color="C1", alpha=0.2)
        ax1.plot(self.reward_in_episode, color="C2", alpha=0.2)
        mean_steps = self._moving_average(self.episode_durations, periods=5)
        mean_reward = self._moving_average(self.reward_in_episode, periods=5)
        lines.append(ax1.plot(mean_steps, label="steps", color="C1")[0])
        lines.append(ax1.plot(mean_reward, label="rewards", color="C2")[0])
        

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon')
        lines.append(ax2.plot(self.epsilon_vec, label="epsilon", color="C3")[0])
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc=3)

        if is_notebook:
            display.clear_output(wait=True)
        else:
            plt.show()
        
        plt.pause(0.001)

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "reward_in_episode": self.reward_in_episode,
            "episode_durations": self.episode_durations,
            "epsilon_vec": self.epsilon_vec,
            "config": self.config
            }, f"{self.model_dir}/pytorch_{self.id}.pt")

    def play(self, verbose:bool=False, sleep:float=0.2, max_steps:int=100):
        # Play an episode
        try:
            actions_str = ["South", "North", "East", "West", "Pickup", "Dropoff"]

            iteration = 0
            state = self.env.reset()  # reset environment to a new, random state
            self.env.render()
            if verbose:
                print(f"Iter: {iteration} - Action: *** - Reward ***")
            time.sleep(sleep)
            done = False

            while not done:
                action = self._get_action_for_state(state)
                iteration += 1
                state, reward, done, info = self.env.step(action)
                display.clear_output(wait=True)
                self.env.render()
                if verbose:
                    print(f"Iter: {iteration} - Action: {action}({actions_str[action]}) - Reward {reward}")
                time.sleep(sleep)
                if iteration == max_steps:
                    print("cannot converge :(")
                    break
        except KeyboardInterrupt:
            pass
            
    def evaluate(self, max_steps:int=100):
        try:
            total_steps, total_penalties = 0, 0
            episodes = 100

            for episode in trange(episodes):
                state = self.env.reset()  # reset environment to a new, random state
                nb_steps, penalties, reward = 0, 0, 0

                done = False

                while not done:
                    action = self._get_action_for_state(state)
                    state, reward, done, info = self.env.step(action)

                    if reward == -10:
                        penalties += 1

                    nb_steps += 1
                    if nb_steps == max_steps:
                        done = True

                total_penalties += penalties
                total_steps += nb_steps

            print(f"Results after {episodes} episodes:")
            print(f"Average timesteps per episode: {total_steps / episodes}")
            print(f"Average penalties per episode: {total_penalties / episodes}")    
        except KeyboardInterrupt:
            pass
