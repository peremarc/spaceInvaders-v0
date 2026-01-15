#==================================================================================================
#              08MIAR 2025 Grupo 21 - Fichero de Funciones Básicas Proyecto Programación RL
#==================================================================================================
#----------------------------- 1. Importación Librerias y funciones -------------------------------
import os
import glob
import json
import sys
import random
import gym
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from collections import deque
from pathlib import Path 

#-------------------------------------- 2. Configuración Básica -----------------------------------
ENV_NAME = "SpaceInvaders-v0"
INPUT_SHAPE = (84, 84)     # (H, W)
WINDOW_LENGTH = 4          # stack in keras-rl memory/model

IM_SHAPE = (INPUT_SHAPE[0], INPUT_SHAPE[1], WINDOW_LENGTH)

#-------------------------------------- 3. Wrappers Gym Clasico -----------------------------------
# WRAPPERS (compatibles con gym clasico y keras-rl2)
# keras-rl2 espera:
#   reset() -> obs
#   step()  -> obs, reward, done, info

# MaxAndSkipEnv: aplica dos transformaciones clave:
# **Frame skipping**, ejecutando la misma acción durante varios pasos consecutivos.
# **Max-pooling** sobre los dos últimos frames observados.
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

# FireResetEnv: tras cada `reset` ejecuta automáticamente la acción necesaria para iniciar la partida.
class FireResetEnv(gym.Wrapper):
    """Asegura que el juego arranque (SpaceInvaders puede requerir FIRE para empezar)."""
    def reset(self, **kwargs):
        obs = self.env.reset()

        # Acción 1 suele ser FIRE en SpaceInvaders ALE
        obs, _, done, info = self.env.step(1)

        if done:
            obs = self.env.reset()
        return obs
    
# NoopResetEnv: introduce una secuencia aleatoria de acciones nulas (`NOOP`) al inicio de cada episodio,
#               con un número máximo predefinido.
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
    
# EpisodicLifeEnv: redefine el final de un episodio cada vez que el agente pierde una vida, 
#                  sin reiniciar completamente el entorno.
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


#--------------------------- 4. PROCESSOR (reward clipping + preprocessing) -----------------------
class AtariProcessor(Processor):
    def process_observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, INPUT_SHAPE, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def process_state_batch(self, batch):
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


#-------------------------------- 5. Funciones de monitorización ----------------------------------
# Función para evaluación resultado del agente.
def agent_eval(hist, Nepi=20, Rmin=20):
    """
    Evalúa el rendimiento del agente de acuerdo al enunciado del problema a partir de la historia de test.
    
    Parámetros:
    - hist: objeto devuelto por dqn.test()
    - Nepi: número de episodios consecutivos a comprobar (int)
    - Rmin: recompensa mínima exigida en cada episodio (int)
    
    Imprime:
    - número total de episodios ejecutados
    - media de recompensas
    - si ha superado el reto de N episodios consecutivos con reward >= Rmin
    """  
    # Extraemos puntuaciones del historial de episodios
    score = hist.get('episode_reward', [])
    
    # Calculamos numero total de episodios ejecutados. Si no hay ninguno nos salimos.
    total_Nepi = len(score)
    
    if total_Nepi <= 0:
        print("\nNo se detectan episodios en la historia del agente. No se puede evaluar su rendimiento")
        return

    print(f"\nEl numero total de episodios ejecutados es: {total_Nepi}")
    
    # Media total de recompensas
    score_avg = np.mean(score)
    score_max = np.max(score)
    score_min = np.min(score)
    print(f"Resultados Test (reward): Media: {score_avg:.2f} | Máximo: {score_max:.2f} | Mínimo: {score_min:.2f}")

    # Revisamos performance del agente
    Cmin = 0       # total de episodios con reward >= Rmin 
    Cmin_cont = 0  # contador consecutivo
    
    if total_Nepi >= Nepi:
        for val in score:
            if val >= Rmin:
                Cmin += 1
                Cmin_cont += 1
            else:
                Cmin_cont = 0
                
        if Cmin_cont == Nepi:
            print(f"Objetivo alcanzado: Durante {Nepi} episodios consecutivos se ha superado o igualado los {Rmin} puntos.")
        elif Cmin_cont > Nepi:
            print(f"Objetivo superado: Durante {Cmin_cont} episodios consecutivos se ha superado o igualado los {Rmin} puntos.")
        else:
            print(f"Objetivo no alcanzado: Se han obtenido {Cmin} de {Nepi} episodios por encima de la puntuación. Es necesario seguir entrenando el agente.")

    else:
        print(f"Objetivo no alcanzado: No se ha evaluado el agente para el numero minimo de {Nepi} episodios requeridos.")   
        print(f"Se han obtenido {Cmin} de {total_Nepi} episodios por encima de la puntuación minima.") 
        
    # Sacamos grafica del test
    print(f"\n") 
    plt.figure(figsize=(10,5))

    # Eje X empezando en 1
    x_vals = range(1, len(score) + 1)

    plt.bar(x_vals, score, label="Score por episodio", alpha=0.7, color="blue")

    plt.axhline(y=Rmin, color='red', linestyle='--', linewidth=2,
                label=f"Límite Rmin = {Rmin}")

    plt.xlabel("Episodio")
    plt.ylabel("Score")
    plt.title("Score del agente durante el test")
    plt.grid(True)

    # Forzar que el eje X empiece en 1 
    plt.xlim(1, len(score))

    ax = plt.gca()
    ticks = ax.get_xticks()

    # Filtrar ticks no deseados (como el 0)
    ticks = [t for t in ticks if t >= 1]

    # Asegurar que el 1 aparece
    if 1 not in ticks:
        ticks = [1] + ticks

    ax.set_xticks(ticks)
    plt.legend()
    plt.show()

    
# Funcion para graficar evolución del reward obtenido por episodio
def plot_rewards(log_path, alpha=0.03):
    """
    Funcion que grafica las recompensa por episodio y la media móvil. Ambas en la misma figura.
    Parametros:
     - log_path: path del fichero log de entrenamiento
     - alpha: factor de suavizado de la EMA (0.02–0.05 recomendado)
    Retorno:
     - Grafica.
    """
    # Cargar JSON
    with open(log_path, "r") as f:
        data = json.load(f)

    # Extraer recompensas
    rewards = np.array(data["episode_reward"])
    n = len(rewards)

    # EMA (Exponential Moving Average)
    ema = np.zeros_like(rewards, dtype=float)
    ema[0] = rewards[0]
    for i in range(1, n): 
        ema[i] = alpha * rewards[i] + (1 - alpha) * ema[i - 1]
    
    # Crear figura
    plt.figure(figsize=(20, 5))

    # Curva original
    plt.plot(rewards, label="Recompensa por episodio", alpha=0.35, color="blue")

    # EMA 
    plt.plot(ema, label=f"EMA (α={alpha})", linewidth=2, color="green")
    
    # Ajustar eje Y
    ymin = rewards.min()
    ymax = rewards.max()
    margen = (ymax - ymin) * 0.01
    plt.ylim(ymin - margen, ymax + margen)
    
    # --- Ajuste dinámico del eje X ---
    if n < 2500:
        step = 100
    elif n < 5000:
        step = 200
    elif n < 7500:
        step = 500
    else:
        step = 1000

    plt.xlim(0, n - 1) 
    ticks = list(np.arange(0, n, step))
    plt.xticks(ticks)

    # Etiquetas y estilo
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title(f"Recompensa por episodio + Media móvil (Total episodios: {n})")
    plt.grid(True)
    plt.legend()
    plt.show()

# Funcion para graficar evolucion del reward obtenido por episodio
def merge_json_logs(json_paths, output_path):
    """
    Une varios ficheros JSON generados por Keras-RL en uno solo.
    Cada JSON debe contener la clave 'episode_reward'.
    """
    merged_rewards = []

    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
            rewards = data.get("episode_reward", [])
            merged_rewards.extend(rewards)

    # Guardamos el JSON combinado
    with open(output_path, "w") as f:
        json.dump({"episode_reward": merged_rewards}, f)

    print(f"JSON combinado guardado en: {output_path}")
    print(f"Total episodios combinados: {len(merged_rewards)}")
    
# Funciones para mostrar en celda los datos del entrenamiento realizado.
# Limpiar Valores
def clean_values(values):
    """Devuelve solo números válidos, filtrando None, '', [], nan."""
    cleaned = []
    for v in values:
        if isinstance(v, (int, float)) and not np.isnan(v):
            cleaned.append(v)
    return cleaned

# Calcula media de una lista eliminando errores y NaN
def safe_mean(values):
    """Media segura: si no hay valores válidos → 0."""
    vals = clean_values(values)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

# Muestra los resultados del training accediendo al fichero log .json
def ShowLastTraining(log_path="training_log.json", steps_por_intervalo=100000):
    # Leer JSON
    with open(log_path, "r") as f:
        print(f"Mostramos los datos del entrenamiento realizado:\n")
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Warning: El fichero JSON está vacío o corrupto.")
            return

    # Comprobar si faltan campos obligatorios o están vacíos
    campos_obligatorios = ["episode", "episode_reward", "nb_steps", "nb_episode_steps", "duration"]

    for campo in campos_obligatorios:
        if campo not in data or not data[campo]:
            print(f"Warning: El fichero JSON no contiene datos suficientes ({campo} está vacío).")
            return

    # Extraer campos obligatorios
    episodes       = data["episode"]
    rewards        = data["episode_reward"]
    nb_steps       = data["nb_steps"]
    nb_ep_steps    = data["nb_episode_steps"]
    duraciones     = data["duration"]

    # Campos opcionales
    optional_fields = {
        "loss":      data.get("loss"),
        "mae":       data.get("mae"),
        "mean_q":    data.get("mean_q"),
        "mean_eps":  data.get("mean_eps"),
        "ale.lives": data.get("ale.lives"),
    }

    # Filtrar solo los que existen realmente
    optional_fields = {k: v for k, v in optional_fields.items() if v is not None}

    # Si nb_steps está vacío, no hay nada que mostrar
    if not nb_steps:
        print("Warning: No hay pasos registrados en el fichero JSON.")
        return

    max_steps = nb_steps[-1]
    num_intervalos = max_steps // steps_por_intervalo

    if num_intervalos == 0:
        print("Warning: No hay suficientes pasos para mostrar intervalos.")
        return

    # Recorrer intervalos
    for k in range(1, num_intervalos + 1):
        inicio = (k - 1) * steps_por_intervalo
        fin    = k * steps_por_intervalo

        idx = [i for i, s in enumerate(nb_steps) if inicio < s <= fin]
        if not idx:
            continue

        r_int      = [rewards[i]     for i in idx]
        dur_int    = [duraciones[i]  for i in idx]
        steps_ep   = [nb_ep_steps[i] for i in idx]

        r_mean  = safe_mean(r_int)
        r_min   = float(np.min(r_int))
        r_max   = float(np.max(r_int))
        dur_tot = float(np.sum(dur_int))
        ms_mean = (dur_tot * 1000.0) / steps_por_intervalo

        episodios_intervalo = len(idx)

        linea_final = (
            f"{episodios_intervalo} episodes - "
            f"episode_reward: {r_mean:.3f} [{r_min:.3f}, {r_max:.3f}]"
        )

        for campo, valores in optional_fields.items():
            valores_intervalo = [valores[i] for i in idx]
            media = safe_mean(valores_intervalo)

            if campo == "ale.lives":
                vivos = clean_values(valores_intervalo)
                ultimo = vivos[-1] if len(vivos) > 0 else 0.0
                linea_final += f" - ale.lives: {ultimo:.3f}"
            else:
                linea_final += f" - {campo}: {media:.3f}"

        print(f"Interval {k} ({fin} steps performed)")
        print(
            f"{steps_por_intervalo}/{steps_por_intervalo} [==============================] "
            f"- {int(dur_tot):d}s {ms_mean:.0f}ms/step - reward: {r_mean:.4f}"
        )
        print(linea_final)
        print("-" * 90)

#---------------------------------- 6. Funciones de training & test -------------------------------
# Crea el entrono de simulacion para training y/o test
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

# Funcion para definir la semilla
def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Funcion para entrenar al agente
def TrainAgent(agent, env, tr_steps, n_sol=1, n_train=1):
    
    # Crear carpeta de la solución 
    folder = Path(f"Solucion{n_sol}") 
    folder.mkdir(exist_ok=True)

    # Ficheros finales
    env_name = ENV_NAME
    w_filename = folder / f'dqn{n_sol}_{env_name}_weights_final_{n_train}.h5f'
    
    checkpoint_filename = folder / f'dqn{n_sol}_{env_name}_weights_{{step}}.h5f'
    log_filename = folder / f'dqn{n_sol}_{env_name}_log_{n_train}.json'
    
    # Callbacks para guardar pesos y logs periódicamente
    callback_1 = ModelIntervalCheckpoint(str(checkpoint_filename), interval=100000)
    callback_2 = FileLogger(str(log_filename), interval=1)

    # Combinamos todos los callbacks
    callbacks = [callback_1, callback_2]

    # Cargar pesos si existe el fichero de training y mostrar el log de resultados pasados
    if glob.glob(str(w_filename) + "*"):
        print(f"Cargando pesos: {w_filename}")
        agent.load_weights(w_filename)
        
        if glob.glob(str(log_filename) + "*"):
            ShowLastTraining(log_filename, 100000)
    else:
        # Entrenar agente desde 0 y guardar pesos al finalizar.
        print("No se encontraron pesos previos, entrenando desde cero...")
        agent.fit(env, nb_steps=tr_steps, visualize=False, verbose=1, callbacks=callbacks, log_interval=100000)
        agent.save_weights(w_filename, overwrite=True)
        print(f"\nPesos guardados en: {w_filename}")

    # Graficamos resultados entrenamiento
    plot_rewards(log_filename, 0.01)

# Funcion para testear al agente
def TestAgent(agent, env, n_sol=1, n_train=1, n_test=100, Nepi_min=100, reward_min=20, visualize=False):
    """
    Evalúa el rendimiento del agente de acuerdo al enunciado del problema a partir de la historia de test.
    
    Parámetros:
    - agent: objeto con el agente definido
    - env: objeto con el entorno definido.
    - n_sol: numero de experimento ejecutado (int).
    - n_train: numero de training dentro del experimento (int).
    - n_test: numero de tests a ejecutar (int)
    - Nepi_min: número de episodios consecutivos a comprobar (int)
    - reward_min: recompensa mínima exigida en cada episodio (int)
    - visualize: =True ver pantalla juego.
    """

    env_name = ENV_NAME
    folder = Path(f"Solucion{n_sol}") 
    w_filename = folder / f'dqn{n_sol}_{env_name}_weights_final_{n_train}.h5f'
        
    if glob.glob(str(w_filename) + "*"):
        print(f"Cargando pesos para test: {w_filename}")
        agent.load_weights(w_filename)

    agent.policy = EpsGreedyQPolicy(eps=0.0)

    print(f"Evaluando performance del agente durante {n_test} episodios...")

    history = {"episode_reward": []}

    for i in range(1, n_test + 1):
        sys.stdout.write(f"\r  Episodio {i}/{n_test}")
        sys.stdout.flush()
        result = agent.test(env, nb_episodes=1, visualize=visualize, verbose=0)
        reward = result.history["episode_reward"][0]
        history["episode_reward"].append(reward)

    print("\nTest finalizado.\n")
    print("Graficamos resultados test:")
    agent_eval(history, Nepi_min, reward_min)