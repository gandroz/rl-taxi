import os
import json
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)
from collections import deque
from tqdm.notebook import trange, tqdm
from config import Config
from memory import Memory
from callback import LogTensorBoard, ExponentialDecay
from IPython.display import clear_output
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError


def clone_model(model):
    clone = Sequential.from_config(model.get_config())
    clone.set_weights(model.get_weights())
    return clone

# https://towardsdatascience.com/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b
# https://towardsdatascience.com/reinforcement-learning-explained-visually-part-6-policy-gradients-step-by-step-f9f448e73754


class QAgent():
    def __init__(self, env=None, config:str=None, seed:int=None, model:tf.keras.Model=None):
        assert env is not None, "A GYM environment must be provided"
        assert config is not None, "A config filename must be provided"
        assert model is not None, "A keras model must be provided"
        self.env = env
        self.config_file = config
        self.config = Config(self.config_file)
        self.model_name = f'train_{int(time.time())}'
        self.log_dir = ""
        self.model = model
        self.target_model = None
        self.tensorboard = None
        self.rng = np.random.default_rng(seed)
        self.memory = None
        self.last_step = 0
        self.current_episode = 0
    
    def compile(self, optimizer=None, loss=Huber()):
        # lr_schedule = ExponentialDecay(
        #                         initial_learning_rate=self.config.learning_rate,
        #                         decay_steps=self.config.lr_decay_steps,
        #                         decay_rate=self.config.lr_decay,
        #                         lr_min=self.config.lr_min)
        # optimizer = Adam(learning_rate=lr_schedule)
        if optimizer is None:
            optimizer = Adam(learning_rate=self.config.learning_rate)
        
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer='sgd', loss='mse')
        
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])
    
    def adjust_lr(self, lr=None):
        assert lr is not None
        K.set_value(self.model.optimizer.learning_rate, lr)

    def load_model(self, filepath):
        self.model.load_weights(filepath)
        self.target_model.set_weights(self.model.get_weights())

    def save(self, filepath=""):
        self.save_model(filepath)
        self.save_checkpoint(filepath)

    def save_model(self, filepath=""):
        filepath = os.path.join(filepath, "models")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.model.save_weights(os.path.join(filepath, self.model_name + ".h5"), overwrite=True)

    def save_checkpoint(self, filepath=""):
        filepath = os.path.join(filepath, "models")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        # save memory, current step
        data = {
                "memory": self.memory.json(), 
                "last_step": self.last_step,
                "current_episode": self.current_episode,
                "model_name": self.model_name
                }
        with open(os.path.join(filepath, self.model_name + "_checkpoint.json"), "w") as jsonfile:
            json.dump(data, jsonfile)

    def load_checkpoint(self, filename):
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        self.last_step = data['last_step']
        self.current_episode = data['current_episode']
        self.model_name = data['model_name']
        if self.memory is None:
            self.config = Config(self.config_file)
            self.memory = Memory(max_len=self.config.max_queue_length)
        self.memory.load(data['memory'])

    def _encode_state(self, state):
        return state
        
    def _train_model(self, step):
        if self.memory.length < self.config.batch_size:
            return
        
        mini_batch = self.memory.sample(self.config.batch_size)
        
        current_states = self._encode_state(mini_batch.states)
        next_states = self._encode_state(mini_batch.new_states)
        
        # current Q values for each action
        q_values = self.model.predict_on_batch(current_states)
        
        # identify the best action to take and get the corresponding target Q value
        target_q_values = self.target_model.predict_on_batch(next_states)
        q_batch = np.max(target_q_values, axis=1).flatten()
        
        indices = (np.arange(self.config.batch_size), mini_batch.actions)
        q_values[indices] = mini_batch.rewards + (1 - mini_batch.done) * self.config.discount_factor * q_batch
        
        # As the model will predict `q_values`, only the Q value for the proper action (given by indices)
        # differ and count for the loss computation.
        self.tensorboard.on_step_begin()
        metrics = self.model.train_on_batch(current_states.astype(np.float32), q_values.astype(np.float32), return_dict=True)
        self.tensorboard.on_step_end(step=step, logs=metrics)
        
    def _get_epsilon(self, episode):
        epsilon = self.config.min_epsilon + \
                          (self.config.max_epsilon - self.config.min_epsilon) * np.exp(-self.config.decay_epsilon * episode)
        return epsilon
    
    def _remember(self, state, action, reward, new_state, done):
        self.memory.append(state, action, reward, new_state, done)
            
    def _get_action_for_state(self, state):
        state_decoded = self._encode_state(state)
        predicted = self.model.predict_on_batch(np.array([state_decoded]))
        action = np.argmax(predicted[0])
        return action
        
    def _choose_action(self, state, epsilon):
        if self.rng.uniform() < epsilon:
            # Explore
            action = self.env.action_space.sample()
        else:
            # Exploit
            action = self._get_action_for_state(state)
        return action

    def fit(self):
        
        try:
            self.config = Config(self.config_file)
            if self.tensorboard is None:
                self.log_dir = os.path.join(self.config.log_dir, self.model_name)        
                self.tensorboard = LogTensorBoard(log_dir=self.log_dir)
            self.tensorboard.set_model(self.model)

            if self.memory is None:
                self.memory = Memory(max_len=self.config.max_queue_length)

            state = self.env.reset()
            done = False            
            epsilon = self._get_epsilon(self.current_episode)
            steps_in_episode = 0
            reward_queue = deque(maxlen=10)
            reward_in_episode = 0
            
            pbar = trange(self.last_step, self.config.train_steps, initial=self.last_step, total=self.config.train_steps)
            for step in pbar: 
                steps_in_episode += 1
                self.last_step = step

                # Greedy exploration strategy
                action = self._choose_action(state, epsilon)
                new_state, reward, done, info = self.env.step(action)
                self._remember(state, action, reward, new_state, done)
                reward_in_episode += reward

                if steps_in_episode == self.config.max_steps_per_episode:
                    done = True

                # Train with the Bellman equation
                if step > self.config.warmup_steps:
                    self._train_model(step)

                state = new_state
                
                if done:
                    steps_in_episode = 0
                    state = self.env.reset()
                    done = False
                    self.current_episode += 1
                    reward_queue.append(reward_in_episode)
                    reward_in_episode = 0
                    epsilon = self._get_epsilon(self.current_episode)
                    pbar.set_postfix({"reward": np.mean(reward_queue)})
                
                if step % self.config.target_model_update == 0:
                    self.target_model.set_weights(self.model.get_weights())
            
            self.last_step += 1
            
        except KeyboardInterrupt:
            print("Training has been interrupted")
            
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
                clear_output(wait=True)
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
