import gym
from pytorch.model import DQN
from pytorch.agent import QAgent


env = gym.make("Taxi-v3").env
config = "pytorch/config_pytorch.yaml"
agent = QAgent(env=env, config=config, model_class=DQN)

agent.compile()
agent.fit()
