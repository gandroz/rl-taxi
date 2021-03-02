import os
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.regularizers import l1_l2, l1, l2
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError


def clone_model(model):
    clone = Sequential.from_config(model.get_config())
    clone.set_weights(model.get_weights())
    return clone


class QAgent():
    def __init__(self, env=None, config:str=None, seed:int=None, model:tensorflow.keras.Model=None):
        assert env is not None, "A GYM environment must be provided"
        assert config is not None, "A config filename must be provided"
        assert model is not None, "A keras model must be provided"
        self.env = env
        self.config = Config(config)
        self.model = model
        self.model = None
        self.target_model = None
        self.tensorboard = LogTensorBoard(log_dir=os.path.join(self.config.log_dir, f'train_{int(time.time())}'))
        self.rng = np.random.default_rng(seed)
        self.memory = Memory(max_len=self.config.max_queue_length)
        self.initial_step = 0
    
    def compile(self):        
        lr_schedule = ExponentialDecay(
                                initial_learning_rate=self.config.learning_rate,
                                decay_steps=self.config.lr_decay_steps,
                                decay_rate=self.config.lr_decay,
                                lr_min=self.config.lr_min)
        optimizer = Adam(learning_rate=lr_schedule)
        loss = Huber()
        
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer='sgd', loss='mse')
        
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])

        self.tensorboard.set_model(self.model)

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
        
        # identity the best action to take and get the corresponding target Q value
        target_q_values = self.target_model.predict_on_batch(next_states)
        q_batch = np.max(target_q_values, axis=1).flatten()
        
        indices = (np.arange(self.config.batch_size), mini_batch.actions)
        q_values[indices] = mini_batch.rewards + (1 - mini_batch.done) * self.config.discount_factor * q_batch
        
        # As the model will predict `q_values`, only the Q value for the proper action (given by indices)
        # differ and count for the loss computation.)
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
            state = self.env.reset()
            done = False
            episode = 0
            epsilon = self._get_epsilon(episode)
            steps_in_episode = 0
            
            pbar = trange(self.initial_step, self.config.train_steps, initial=self.initial_step, total=self.config.train_steps)
            for step in pbar: 
                # Explore/Exploit using the Epsilon Greedy Exploration Strategy
                action = self._choose_action(state, epsilon)
                new_state, reward, done, info = self.env.step(action)
                
                steps_in_episode += 1
                if steps_in_episode == self.config.max_steps_per_episode:
                    done = True
                self._remember(state, action, reward, new_state, done)

                # Update the Main Network using the Bellman Equation
                if step > self.config.warmup_steps:
                    self._train_model(step)

                state = new_state
                
                if done:
                    steps_in_episode = 0
                    state = self.env.reset()
                    done = False
                    episode += 1
                    epsilon = self._get_epsilon(episode)
                
                if step % self.config.target_model_update == 0:
                    self.target_model.set_weights(self.model.get_weights())

        except KeyboardInterrupt:
            print("Training has been interrupted")
            self.initial_step = step
            
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
