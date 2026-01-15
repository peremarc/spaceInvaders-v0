#!/usr/bin/env python
# coding: utf-8
#==================================================================================================
#              08MIAR 2025 Grupo 21 - Fichero de Funciones PyTorch Proyecto Programación RL
#==================================================================================================
#----------------------------- 1. Importación Librerias y funciones -------------------------------
import os
import json
import time
import math
import random
import numpy as np
import torch
import copy
import cv2
from collections import deque
from IPython.display import clear_output
from env_functions import plot_rewards
import matplotlib.pyplot as plt

#-------------------------------------- 2. Configuración Básica -----------------------------------
# Funciones Adicionales para Pytorch   
# Buffer memoria PER
class ReplayBufferPER(object):
    def __init__(self, capacity, prob_alpha=0.6):
        """
        capacity: tamaño máximo del buffer
        prob_alpha: controla cuánta prioridad se aplica (0 = uniforme, 1 = totalmente prioritizado)
        """
        self.capacity   = capacity
        self.prob_alpha = prob_alpha

        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    # Insertar nueva transición en el buffer.
    def push(self, state, action, reward, next_state, done):
        """
        Guarda una transición en el buffer.
        state y next_state deben ser uint8 con shape (C,H,W)
        """
        # ELIMINA los np.expand_dims, guarda el array tal cual (C, H, W)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        # Manejo de prioridad inicial
        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    # Devolver batch de muestras usando PER
    def sample(self, batch_size, beta=0.4):
        """
        Devuelve un batch PER:
        - states, actions, rewards, next_states, dones
        - indices del buffer
        - importance sampling weights
        """
        current_size = len(self.buffer)
        prios = self.priorities[:current_size]

        # Probabilidades proporcional a la prioridad
        probs = prios ** self.prob_alpha
        s = probs.sum()
        
        # Prteccion contra NaNs o sumas cero
        if s == 0 or np.isnan(s):
            probs =  np.ones_like(probs) / len(probs)
        else:
            probs /= s

        # Selección de índices según probs
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Extraer muestras
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convertir a arrays numpy
        # states      = np.vstack(states)      # (B, C, H, W)
        # next_states = np.vstack(next_states)
        states      = np.stack(states)      # (B, C, H, W)
        next_states = np.stack(next_states)

        actions     = np.array(actions, dtype=np.int64)
        rewards     = np.array(rewards, dtype=np.float32)
        dones       = np.array(dones, dtype=np.uint8)

        # Importance Sampling Weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    # Actualiza las prioridades de las transiciones muestreadas
    def update_priorities(self, indices, priorities):
        """
        Actualiza prioridades después de calcular TD-error.
        """        
        max_prio = 1e4
        
        for idx, prio in zip(indices, priorities):
            # if np.isnan(prio) or prio <= 0:
            #     prio = 1e-6
            # self.priorities[idx] = prio
            
            prio = float(prio)
            # Si es NaN, asignamos prioridad mínima
            if np.isnan(prio) or prio <= 0:
                prio = 1e-6

            # Clipping para evitar explosiones numéricas
            if prio > max_prio:
                prio = max_prio

            self.priorities[idx] = prio

    # Devuelve el número actual de transiciones almacenadas en el buffer
    def __len__(self):
        return len(self.buffer)

# Time difference loss calculation para buffer PER.
# Actualiza la red principal con nuevos pesos para un batch de transiciones.
def compute_td_loss_PER(c_model, t_model, optimizer, replay_buffer, device, batch_size=32, gamma=0.99, beta=0.4, clipnorm=0, huberloss= False):

    # Sample con PER
    state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size, beta)

    # Convertir a tensores
    state      = torch.from_numpy(state).to(device)
    next_state = torch.from_numpy(next_state).to(device)
    action     = torch.from_numpy(action).long().to(device)
    reward     = torch.from_numpy(reward).float().to(device)
    done       = torch.from_numpy(done).float().to(device)
    weights    = torch.from_numpy(weights).float().to(device)

    # Normalización
    state = state.float() / 255.0
    next_state = next_state.float() / 255.0

    # Q(s,a)
    q_values = c_model(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    mean_q_batch = q_values.mean().item()

    # Q_target(s',a')
    with torch.no_grad():
        next_q_values = t_model(next_state)
        next_q_value = next_q_values.max(1)[0]

    # Bellman target
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # Calculamos valor Q esperado (Bellman)
    td_error = q_value - expected_q_value.detach()

    # Loss para PER (huber o MSE)
    if huberloss:
        huber = torch.nn.functional.smooth_l1_loss(q_value, expected_q_value, reduction='none')
        loss = (weights * huber).mean()
    else:
        loss = (weights * td_error.pow(2)).mean()

    # MAE del batch
    mae_batch = torch.abs(td_error).mean().item()

    # Limpiamos gradientes anteriores y hacemos calculo gradientes nuevos mediante loss.backward
    optimizer.zero_grad()
    loss.backward()

    # Actualizamos pesos con ClipNorm
    if clipnorm > 0:
        torch.nn.utils.clip_grad_norm_(c_model.parameters(), max_norm=clipnorm)
        
    optimizer.step()

    # Devolver prioridades nuevas (abs del TD-error)
    new_priorities = td_error.abs().detach().cpu().numpy() + 1e-6
    
    # Actualiza prioridades del buffer
    replay_buffer.update_priorities(indices, new_priorities)
    
    # Compcta datos para salida
    td_data = [loss.item(), mean_q_batch, mae_batch]

    # Devolvemos la datos salida calculados
    return td_data


# Mi agente DQN. Calcula la accion en cada Step.
class DQNag:
    # Inicializamos el agente
    def __init__(self, c_model, n_actions, env, eps_start=1.0, eps_final=0.01, eps_decay=30000, exp_decay= False,
                       nb_steps_warmup=50000, train_interval=4, target_model_update=10000, batch_size=32, gamma=0.99, 
                       soft_update=False, tau=0.005):
        # Cargamos device, modelo y numero acicones
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_model = c_model.to(self.device)
        self.n_actions = n_actions      

        # Parámetros de epsilon
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.exp_decay = exp_decay

        # Parámetros de entrenamiento 
        self.env = env
        self.nb_steps_warmup = nb_steps_warmup 
        self.train_interval = train_interval 
        self.target_model_update = target_model_update 
        self.batch_size = batch_size 
        self.gamma = gamma
        self.soft_update = soft_update 
        self.tau = tau

        # Valor actual de epsilon (inicialmente el máximo) 
        self.epsilon = eps_start
    
    # Calculamos acción según metodo €-greedy
    def act(self, state, n_step):
        # Calculo epsilon con decaimiento exponencial o lineal.
        if self.eps_decay > 0:
            if self.exp_decay:
                self.epsilon = self.eps_final + (self.eps_start - self.eps_final) * math.exp(-1.0 * n_step / self.eps_decay)
            else:
                decay_rate = (self.eps_start - self.eps_final) / self.eps_decay
                self.epsilon = max(self.eps_final, self.eps_start - decay_rate * n_step)
        else:
            self.epsilon = self.eps_final
        
        # Politica €-greedy. Muestreamos y si el valor < epsilon --> seleccionamos una accion aleatoria
        if random.random() < self.epsilon:               
            act = random.randrange(self.n_actions)
               
        # Politica €-greedy. Muestreamos y si el valor >= epsilon --> seleccionar el maximo q-value del modelo 
        else:
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
                                                                             # Añadimos 1 dimension para eu la CNN lo pueda procesar
                                                                             # (C, H, W) → (1, C, H, W)          
            with torch.no_grad():                   # Evitamos que PyTorch calcule gradientes en este punto
                q_value = self.c_model(state_t)     # cargamos estado en la red y obtenemos las 6 opciones Q-values
                act = q_value.argmax(dim=1).item()  # Seleccionamos el indice con el valor Q maximo como accion.

        # Devuelve el numero de acción a ejecutar (int).
        return act
    
# Funcion actualizacion de la red target. Copia todos los pesos de la red principal a la target.   
def update_target(c_model, t_model, tau=0.005, soft_update=False):
    with torch.no_grad(): # Evita el rastreo de gradientes innecesario
        # Soft update
        if soft_update and (0.0 <= tau <= 1.0):
            for target_param, param in zip(t_model.parameters(), c_model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Hard update
        else:
            t_model.load_state_dict(c_model.state_dict())

# Funcion para gestionar logs en PyTorch
class LogTorch:
    def __init__(self, interval=10000, tr_steps=10000):
        # Entradas
        self.steps_per_interval = max(1,min(interval, tr_steps))
        self.tr_steps = tr_steps
         
        # Parametros
        self.bar_length = 20 
        self.n_steps_print = 200

        # variables internas       
        self.n_steps_int, self.n_int_actual, self.n_episodes_int = 0,0,0
        self.last_start_time = time.time()
        
        # Acumuladores
        self.reward_acc, self.eps_acc, self.mae_acc = 0,0,0
        self.loss_acc, self.qvalue_acc = 0,0
        
        # Estadisticas por episodio
        self.episode_reward, self.reward_previo, self.reward_epi_acc = 0,0,0
        self.reward_N, self.eps_mean, self.mae_mean, self.loss_mean, self.qvalue_mean, = 0,0,0,0,0
        self.loss, self.q_batch, self.mae_batch = 0,0,0
               
        # Diccionario para logs
        self.dict_datos_intervalo = {    
            "reward_mean": 0.0,
            "reward_min": 100.0,
            "reward_max": 0.0,
            "reward_N": 0.0,
            "eps_mean": 0.0,
            "mae_mean": 0.0,
            "loss_mean": 0.0,
            "qvalue_mean": 0.0,
            "n_episodes": 0,
            "n_steps": 0.0,
            "duration": 0.0,
            "ms_step": 0.0
        }

        # Buffers por episodio
        self.all_rewards = []
        
        # Buffer por intervalo
        self.num_intervalos = tr_steps//self.steps_per_interval
        self.datos_intervalo = [
            copy.deepcopy(self.dict_datos_intervalo) for _ in range(self.num_intervalos)
        ]
        
        # Buffer calculos ciclicos
        self.datos_actuales = copy.deepcopy(self.dict_datos_intervalo)

    @staticmethod 
    def to_python(obj): 
        if hasattr(obj, "item"): 
            return obj.item() 
        return obj
        
    # Funcion que calcula los datos a mostrar durante el entrenamiento
    def update_interval(self, n_step, TD_data, done, reward, epsilon):
        # Extramemos TD
        loss, q_batch, mae_batch = TD_data       

        # Calculo valores acumulados en el intervalo
        self.n_steps_int += 1 
        self.reward_acc += reward       
        self.eps_acc += epsilon
        self.mae_acc += mae_batch
        self.loss_acc += loss
        self.qvalue_acc += q_batch
  
        # Calculo valores ciclicos
        if self.n_steps_int > 0:
            self.datos_actuales["reward_N"] = self.reward_acc/self.n_steps_int
            self.datos_actuales["eps_mean"] = self.eps_acc/self.n_steps_int
            self.datos_actuales["mae_mean"] = self.mae_acc/self.n_steps_int
            self.datos_actuales["loss_mean"] = self.loss_acc/self.n_steps_int
            self.datos_actuales["qvalue_mean"] = self.qvalue_acc/self.n_steps_int

        self.datos_actuales["duration"] = time.time() - self.last_start_time  
        self.datos_actuales["ms_step"] = 0
        self.datos_actuales["n_steps"] = self.n_steps_int

        # Calculo valores por episodio
        if done: 
            # Valores reward
            self.episode_reward = self.reward_acc - self.reward_previo
            self.reward_previo = self.reward_acc
            self.reward_epi_acc += self.episode_reward
            self.datos_actuales["reward_max"]= max(self.datos_actuales["reward_max"], self.episode_reward)

            if self.n_episodes_int > 0:
                self.datos_actuales["reward_mean"] = self.reward_epi_acc / self.n_episodes_int
                self.datos_actuales["reward_min"]= min(self.datos_actuales["reward_min"], self.episode_reward)
            else:
                self.datos_actuales["reward_mean"] = self.reward_epi_acc
                self.datos_actuales["reward_min"]= 100

            # Incrementamos numero episodios en este intervalo y lo guardamos
            self.n_episodes_int += 1
            self.datos_actuales["n_episodes"] = self.n_episodes_int
            
            # Guardamos reward episodio en buffer global
            self.all_rewards.append(self.episode_reward)
    
        # Inicio nuevo intervalo
        if n_step % self.steps_per_interval == 0:
            # Capturamos instante tiempo
            start_time = time.time()

            # Guardar copia del intervalo
            self.datos_intervalo[self.n_int_actual] = copy.deepcopy(self.datos_actuales)
           
            # Ajustar valores finales del intervalo
            dur = start_time - self.last_start_time 
            self.datos_intervalo[self.n_int_actual]["duration"] =  dur
            self.datos_intervalo[self.n_int_actual]["ms_step"] = 1000*(dur/self.steps_per_interval)
            self.datos_actuales["n_steps"] = self.steps_per_interval

            # Guardamos tiempo para siguiente intervalo
            self.last_start_time = start_time
            
            # Reseteamos contadores acumulados
            self.n_episodes_int = 0
            self.n_steps_int = 0
            self.reward_acc, self.eps_acc, self.mae_acc, self.loss_acc, self.qvalue_acc = 0,0,0,0,0

            # Reseteamos calculos
            self.episode_reward, self.reward_previo, self.reward_epi_acc = 0,0,0
            self.datos_actuales["reward_max"]= 0
            self.datos_actuales["reward_min"]= 100
            self.datos_actuales["reward_mean"]= 0
            
        
        # Imprimir datos en pantalla
        self.print_data(n_step, self.datos_intervalo[self.n_int_actual], self.n_int_actual, True)
                   
        # Imprimimos resumen final si es el ultimo step.
        if (n_step == self.tr_steps) :
            self.print_final(self.datos_intervalo)

        # Incrementar contador intervalos
        if (n_step % self.steps_per_interval == 0) and (n_step < self.tr_steps):
            self.n_int_actual += 1

    # Funcion que inprime en pantalla datos en tiempo real
    def print_data(self, n_step, int_data, n_int_actual, TrainingOn= False):
        
        # Imprimimos primera linea cuando se incrementa el intervalo
        if (n_step % self.steps_per_interval == 1) or not TrainingOn:
            print(f"Interval {n_int_actual+1} ({self.steps_per_interval} steps performed)")   

        # Imprimimos progreso dentro del intervalo - cada X steps
        if (n_step % self.n_steps_print == 0) and TrainingOn:   
            steps_left = self.steps_per_interval - self.datos_actuales["n_steps"]
            ratio_eta = (self.datos_actuales["duration"] / self.steps_per_interval) if self.steps_per_interval > 0 else 0
            ratio_steps = (self.datos_actuales["n_steps"] / self.steps_per_interval) if self.steps_per_interval > 0 else 0
            eta = ratio_eta * steps_left
            
            filled = int(self.bar_length * ratio_steps)
            bar = "=" * filled + "." * (self.bar_length - filled)

            act_rewardN = self.datos_actuales["reward_N"]
            
            # Línea de progreso (siempre en la misma línea)
            print(
                f"\r{self.n_steps_int}/{self.steps_per_interval} [{bar}] - ETA: {eta:.1f}s - reward: {act_rewardN:.3f}",
                end=""
            )

        # Imprimir en pantalla datos actuales
        if (n_step % self.steps_per_interval == 0) and TrainingOn or not TrainingOn:                    
            # Fin del intervalo - imprimimos segunda linea con resultados
            
            intervale_duration = int_data["duration"]
            intervale_ms_step = int_data["ms_step"]
            mean_reward_N = int_data["reward_N"]
            mean_reward = int_data["reward_mean"]
            min_reward = int_data["reward_min"]
            max_reward = int_data["reward_max"]
            mean_epi_loss = int_data["loss_mean"]
            mean_epi_mae = int_data["mae_mean"]
            mean_epi_q = int_data["qvalue_mean"]
            mean_epi_eps = int_data["eps_mean"]
            n_episodes = int_data["n_episodes"]
            
            print(
                f"\r{self.steps_per_interval}/{self.steps_per_interval} "
                f"[{'=' * self.bar_length}] - {intervale_duration:.0f}s "
                f"{intervale_ms_step:.1f}ms/step - reward: {mean_reward_N:.4f}"
            )
            print(
                f"{n_episodes} episodes - "
                f"episode_reward: {mean_reward:.3f} [{min_reward:.3f}, {max_reward:.3f}] - "
                f"loss: {mean_epi_loss:.3f} - mae: {mean_epi_mae:.3f} - "
                f"mean_q: {mean_epi_q:.3f} - mean_eps: {mean_epi_eps:.3f}"
            )
            print("-" * 90)
      
    # Funcion que imprime resumen final
    def print_final(self, datos_intervalo):
        
        # Inicializar variable funcion
        tr_total_t, total_episodes = 0,0 
        
        # Calculamos datos totales intervalos
        for tr_data in datos_intervalo:
            tr_total_t += np.sum(tr_data["duration"])
            total_episodes += np.sum(tr_data["n_episodes"])
        
        # Media final de ms/step
        ms_per_step_total = ((tr_total_t / self.tr_steps) * 1000) if self.tr_steps > 0 else 0
        
        print(
            f"\n==== TRAINING COMPLETADO ==== | "
            f"Total steps: {self.tr_steps} | "
            f"Episodes: {total_episodes} | "
            f"Total time: {tr_total_t:.1f}s {ms_per_step_total:.0f}ms/step | "
            f"=====\n"
        )

    # Funcion que carga los datos del fichero log de un entrenamiento anterior
    def load_log(self, log_filename):
        # Chequeamos si existe el fichero. Devovemos False sino lo hay-      
        if not os.path.exists(log_filename): 
            return False
    
        try:
            # abrir fichero logs
            with open(log_filename, "r") as f:
                log_data = json.load(f)

            print(f"Fichero Logs '{log_filename}' cargado correctamente.\n") 
                   
            # Imprimir datos en pantalla
            for ni, data_interval in enumerate(log_data['intervals']):
                self.print_data(-1, data_interval, ni, False)
            self.print_final(log_data['intervals'])

            return True
        
        # Si hay error lo mostramos y devolvemos False
        except json.JSONDecodeError: 
            print(f"Error: el fichero '{log_filename}' está corrupto o no es JSON válido.") 
            return False

    # Funcion que salva los datos del entrenamiento en un fichero .json
    def save_log(self, log_filename):
        logdata= {
            "intervals": self.datos_intervalo,
            "episode_reward": self.all_rewards
        }

        # Salvamos fichero logs
        with open(log_filename, "w") as f:
            json.dump(logdata, f, indent=4, default=self.to_python)

        # Chequeamos si existe el fichero. Devovemos False sino lo hay-      
        print(f"\nFichero datos entrenamiento {log_filename} guardado correctamente.")

## Funciones para pre-procesado
# Pasar de rgb a nivel de gris y re-escalado
def rgb2gray_and_resize(screen, height, width):
    # Convertir RGB → gris
    gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    # Redimensionar a (height, width)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    return resized

# Concatenar secuencias (window_lenght). Devuelve un ndarray.
def update_frame_sequence(state, obs, n_frames=4, width=84, height=84):
    # Preprocesado: gris + resize
    obs = rgb2gray_and_resize(obs, height, width).astype(np.uint8)

    # Añadir dimensión canal → (1, H, W)
    obs = obs.reshape(1, height, width)

    if state is None:
        # Repetimos el primer frame n_frames veces
        state = np.repeat(obs, n_frames, axis=0)
    else:
        # Desplazamos y añadimos el nuevo frame
        state = np.concatenate([state[1:], obs], axis=0)

    return state
