import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
from tensorflow.keras import layers
from gym.wrappers.time_limit import TimeLimit


class CriticModel:
    # https://keras.io/examples/rl/actor_critic_cartpole/
    def __init__(self, num_inputs: int, num_actions: int):
        num_hidden = 128

        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(num_hidden, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)
        self._num_actions = num_actions
        self._model = keras.Model(inputs=inputs, outputs=[action, critic])

        self._optimizer = keras.optimizers.Adam(learning_rate=0.01)
        self._huber_loss = keras.losses.Huber()
        self._eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        self._episode_count = 0
        self._total_reward: float = 0.0
        self._episode_reward: float = 0.0
        self._rewards_history: List[float] = []
        self._critic_history: List[float] = []
        self._log_action_history: List[float] = []

    def predict(self, inputs: np.ndarray):
        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.expand_dims(inputs, 0)
        actions, critic_value = self._model(inputs)
        action_num = np.random.choice(self._num_actions, p=np.squeeze(actions))

        self._critic_history.append(critic_value[0, 0])
        self._log_action_history.append(tf.math.log(actions[0, action_num]))

        return action_num

    def set_reward(self, value: float):
        self._rewards_history.append(value)
        self._episode_reward += value

    def end_episode(self, tape) -> Tuple[int, float]:
        self._episode_count += 1;
        self._total_reward = 0.05 * self._episode_reward + (1 - 0.05) * self._total_reward

        gamma = 0.99  # Discount factor for past rewards
        sum_rewards: List[float] = []
        discounted_sum = 0.0
        for r in self._rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            sum_rewards.insert(0, discounted_sum)

        # Normalize
        sum_rewards = np.array(sum_rewards)
        sum_rewards = ((sum_rewards - np.mean(sum_rewards)) / (np.std(sum_rewards) + self._eps)).tolist()

        # Calculating loss values to update network
        actor_losses = []
        critic_losses = []
        history = zip(self._log_action_history, self._critic_history, sum_rewards)
        for log_action, critic_value, sum_reward in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_action` and ended up recieving a total reward = `sum_reward`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = sum_reward - critic_value
            actor_losses.append(-log_action * diff)

            # The critic must be updated so that it predicts a better estimate of the future rewards.
            critic_losses.append(
                self._huber_loss(tf.expand_dims(critic_value, 0), tf.expand_dims(sum_reward, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        self._episode_reward = 0.0
        self._rewards_history.clear()
        self._critic_history.clear()
        self._log_action_history.clear()

        return (self._episode_count, self._total_reward)


env: TimeLimit = gym.make("CartPole-v0")
action_space: gym.spaces.discrete.Discrete = env.action_space
observation_space: gym.spaces.box.Box = env.observation_space
env.seed(42)

model = CriticModel(observation_space.shape[0], action_space.n)

while True:
    with tf.GradientTape() as tape:
        state = env.reset()
        for _ in range(10000):
            action_num = model.predict(state)
            state, reward, done, _ = env.step(action_num)
            model.set_reward(reward)

            if done:
                break

        episode_count, total_reward = model.end_episode(tape)

    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(total_reward, episode_count))

    if total_reward > 195:
        print("Solved at episode {}!".format(episode_count))
        state = env.reset()
        for _ in range(10000):
            env.render()
            action_num = model.predict(state)
            state, reward, done, _ = env.step(action_num)
            if done:
                break

        break
