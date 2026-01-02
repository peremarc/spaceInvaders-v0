import os
import argparse
import random
from collections import deque

import gym
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Layer, Permute
from tensorflow.keras.models import Model, Sequential
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
STAGE_STEPS_DEFAULT = 3_000_000  # para acercarnos al objetivo min(last100)>20
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
    """Asegura que el juego arranque (SpaceInvaders puede requerir FIRE para empezar)."""
    def reset(self, **kwargs):
        obs = self.env.reset()

        # Acción 1 suele ser FIRE en SpaceInvaders ALE
        obs, _, done, info = self.env.step(1)

        if done:
            obs = self.env.reset()
        return obs


class NoopResetEnv(gym.Wrapper):
    """
    Ejecuta un número aleatorio de acciones NOOP tras reset.
    """
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # NOOP en Atari

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # Número aleatorio de NOOPs
        n_noops = np.random.randint(1, self.noop_max + 1)

        for _ in range(n_noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)

        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Hace 'done=True' cuando se pierde una vida (pero no es game over).
    Acelera aprendizaje en Atari (solo recomendable en TRAIN).
    """
    def __init__(self, env, lives_key="ale.lives"):
        super().__init__(env)
        self.lives_key = lives_key
        self.lives = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.lives = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        lives = info.get(self.lives_key, None)
        if lives is not None:
            if self.lives == 0:
                self.lives = lives

            # Si pierdes una vida pero NO es game over, cortamos el episodio
            if (lives < self.lives) and (lives > 0):
                done = True

            self.lives = lives

        return obs, reward, done, info


# =========================
# PROCESSOR (reward clipping + preprocessing)
# =========================
class AtariProcessor(Processor):
    def process_observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, INPUT_SHAPE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def process_state_batch(self, batch):
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nb_actions, activation="linear"))
    return model


def make_env(seed: int, training: bool):
    env = gym.make(ENV_NAME)

    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)  

    # Episodic life SOLO en training
    if training:
        env = EpisodicLifeEnv(env, lives_key="ale.lives")

    env = MaxAndSkipEnv(env, skip=4)

    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    except TypeError:
        env.seed(seed)
        env.reset()

    return env



def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def build_agent(env):
    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=0.02, # 0.1, 1.0
        value_min=0.005, # 0.02, 0.1
        value_test=0, 
        nb_steps=300_000, # 500_000, 1000000
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=20_000, # 100_000
        gamma=0.99,
        target_model_update=5e-3, # usamos Polyak averaging
        batch_size=32,
        train_interval=4,
        enable_double_dqn=True,
        enable_dueling_network=True,
        dueling_type="avg",
        delta_clip=1.0,
    )

    dqn.compile(Adam(learning_rate=5e-5, clipnorm=10.0), metrics=["mae"]) #1e-4
    return dqn


def train(dqn, env):
    if os.path.exists(WEIGHTS_PATH + ".index"):
        print(f"Cargando pesos: {WEIGHTS_PATH}")
        dqn.load_weights(WEIGHTS_PATH)
    else:
        print("No se encontraron pesos previos, entrenando desde cero.")

    dqn.fit(
        env,
        nb_steps=STAGE_STEPS_DEFAULT,
        log_interval=10_000,
        visualize=False,
        verbose=2,
        callbacks=[
            FileLogger(LOG_PATH, interval=100),
            ModelIntervalCheckpoint(WEIGHTS_PATH, interval=50_000),
        ],
    )

    dqn.save_weights(WEIGHTS_PATH, overwrite=True)
    print(f"Pesos guardados en: {WEIGHTS_PATH}")


def evaluate(dqn, env):
    print("\nIniciando evaluación (110 episodios)...")
    history = dqn.test(env, nb_episodes=110, visualize=False, verbose=0)

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
    args = parser.parse_args()

    # Si no se ponen flags, hace train+test por defecto
    do_train = args.train or (not args.train and not args.test)
    do_test = args.test or (not args.train and not args.test)

    set_seeds(SEED)

    env_train = make_env(SEED, training=True)
    env_test  = make_env(SEED, training=False)
    
    dqn = build_agent(env_train)
    # Confirmar que el modelo es dueling
    print("Dueling activado:", dqn.enable_dueling_network)
    print("Dueling type:", getattr(dqn, "dueling_type", "no attribute"))

    if do_train:
        train(dqn, env_train)

    if do_test:
        # Asegura que cargue pesos si solo test
        if os.path.exists(WEIGHTS_PATH + ".index"):
            print(f"Cargando pesos para evaluación: {WEIGHTS_PATH}")
            dqn.load_weights(WEIGHTS_PATH)
        else:
            print(f"No se encontraron pesos en {WEIGHTS_PATH}, saliendo...")
            return
        evaluate(dqn, env_test)


if __name__ == "__main__":
    main()
