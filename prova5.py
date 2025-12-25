import os
import argparse
import random
from collections import deque

import gym
import numpy as np
from PIL import Image

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Layer, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


# =========================
# CONFIG
# =========================
ENV_NAME = "SpaceInvaders-v0"
SEED = 42

INPUT_SHAPE = (84, 84)     # (H, W)
WINDOW_LENGTH = 4          # stack in keras-rl memory/model
STAGE_STEPS_DEFAULT = 3_000_000  # recomendado para acercarte a min(last100)>20
WEIGHTS_PATH = "dqn_spaceinvaders_v0_weights.h5f"
LOG_PATH = "training_log.json"


# =========================
# WRAPPERS (compatibles con gym clasico y keras-rl2)
# keras-rl2 espera:
#   reset() -> obs
#   step()  -> obs, reward, done, info
# =========================
class MaxAndSkipEnv(gym.Wrapper):
    """Frame-skip + max-pooling de los últimos 2 frames."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs  # keras-rl2 espera solo obs

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            terminated = terminated or done
            if done:
                break

        max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[-1])
        done = terminated
        return max_frame, total_reward, done, info  # keras-rl2 API


class FireResetEnv(gym.Wrapper):
    """Asegura que el juego arranque (SpaceInvaders suele requerir FIRE)."""
    def reset(self, **kwargs):
        obs = self.env.reset()

        # Acción 1 suele ser FIRE en SpaceInvaders ALE
        obs, _, done, info = self.env.step(1)

        if done:
            obs = self.env.reset()
        return obs


# =========================
# PROCESSOR (reward clipping + preprocessing)
# =========================
class AtariProcessor(Processor):
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert("L")   # grayscale 84x84
        return np.array(img).astype("uint8")

    def process_state_batch(self, batch):
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


# =========================
# DUELING HEAD
# =========================
class DuelingCombine(Layer):
    def __init__(self, dueling_type="avg", **kwargs):
        super().__init__(**kwargs)
        self.dueling_type = dueling_type

    def call(self, inputs):
        value, advantage = inputs
        if self.dueling_type == "avg":
            advantage = advantage - K.mean(advantage, axis=1, keepdims=True)
        return value + advantage

    def get_config(self):
        config = super().get_config()
        config.update({"dueling_type": self.dueling_type})
        return config


def build_model(nb_actions: int) -> Model:
    # keras-rl2 entrega (window_length, H, W)
    inputs = Input(shape=(WINDOW_LENGTH,) + INPUT_SHAPE, name="input_frames")
    x = Permute((2, 3, 1), name="permute_to_channels_last")(inputs)  # (H, W, window)

    x = Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(x)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)

    value = Dense(1, activation="linear", name="V")(x)
    advantage = Dense(nb_actions, activation="linear", name="A")(x)
    q_values = DuelingCombine(dueling_type="avg", name="Q")([value, advantage])

    return Model(inputs=inputs, outputs=q_values, name="DuelingDQN")


def make_env(seed: int):
    env = gym.make(ENV_NAME)

    # wrappers "tipo Atari"
    env = MaxAndSkipEnv(env, skip=4)
    env = FireResetEnv(env)

    env.seed(seed)
    _ = env.reset()
    return env


def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def build_agent(env, steps_anneal: int, eps_min: float, memory_limit: int):
    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    memory = SequentialMemory(limit=memory_limit, window_length=WINDOW_LENGTH)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=eps_min,
        value_test=0.01,
        nb_steps=steps_anneal,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=50_000,
        gamma=0.99,
        target_model_update=10_000,
        train_interval=4,
        enable_double_dqn=True,
        enable_dueling_network=False,  # tu modelo ya es dueling
        custom_model_objects={"DuelingCombine": DuelingCombine},
    )

    dqn.compile(Adam(learning_rate=2.5e-4), metrics=["mae"])
    return dqn


def train(dqn, env, nb_steps: int):
    if os.path.exists(WEIGHTS_PATH + ".index"):
        print(f"Cargando pesos: {WEIGHTS_PATH}")
        dqn.load_weights(WEIGHTS_PATH)
    else:
        print("No se encontraron pesos previos, entrenando desde cero.")

    dqn.fit(
        env,
        nb_steps=nb_steps,
        log_interval=10_000,
        visualize=False,
        verbose=2,
        callbacks=[
            FileLogger(LOG_PATH, interval=100),
            ModelIntervalCheckpoint(WEIGHTS_PATH, interval=100_000),
        ],
    )

    dqn.save_weights(WEIGHTS_PATH, overwrite=True)
    print(f"Pesos guardados en: {WEIGHTS_PATH}")


def evaluate(dqn, env, nb_episodes: int = 110):
    print(f"\nIniciando evaluación ({nb_episodes} episodios)...")
    history = dqn.test(env, nb_episodes=nb_episodes, visualize=False, verbose=0)

    rewards = history.history["episode_reward"]
    last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
    min_score = float(np.min(last_100))
    mean_score = float(np.mean(last_100))

    print(f"Min reward (últimos {len(last_100)}): {min_score}")
    print(f"Mean reward (últimos {len(last_100)}): {mean_score}")

    if len(last_100) >= 100 and min_score > 20:
        print("ESTADO: REQUISITO CUMPLIDO (min(last100) > 20)")
    else:
        print("ESTADO: REQUISITO NO CUMPLIDO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Entrenar el agente")
    parser.add_argument("--test", action="store_true", help="Evaluar el agente")
    parser.add_argument("--steps", type=int, default=STAGE_STEPS_DEFAULT, help="Pasos de entrenamiento")
    parser.add_argument("--anneal", type=int, default=1_000_000, help="Pasos para bajar epsilon")
    parser.add_argument("--epsmin", type=float, default=0.1, help="Epsilon mínimo durante entrenamiento")
    parser.add_argument("--memory", type=int, default=1_000_000, help="Tamaño del replay buffer")
    parser.add_argument("--episodes", type=int, default=110, help="Episodios de test")
    args = parser.parse_args()

    # Si no pones flags, hace train+test por defecto
    do_train = args.train or (not args.train and not args.test)
    do_test = args.test or (not args.train and not args.test)

    set_seeds(SEED)

    env = make_env(SEED)
    dqn = build_agent(env, steps_anneal=args.anneal, eps_min=args.epsmin, memory_limit=args.memory)

    if do_train:
        train(dqn, env, nb_steps=args.steps)

    if do_test:
        # Asegura que cargue pesos si solo test
        if os.path.exists(WEIGHTS_PATH + ".index"):
            print(f"Cargando pesos para evaluación: {WEIGHTS_PATH}")
            dqn.load_weights(WEIGHTS_PATH)
        else:
            print(f"No se encontraron pesos en {WEIGHTS_PATH}, saliendo...")
            return
        evaluate(dqn, env, nb_episodes=args.episodes)


if __name__ == "__main__":
    main()
