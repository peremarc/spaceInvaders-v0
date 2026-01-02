## Instalación del entorno (Linux)
- Ejecuta `./install_keras_rl2.sh --accept-license` para crear `.venv-rl2`, instalar dependencias fijadas (gym 0.17.3, TF 2.5.3, keras-rl2 1.0.5, atari-py) y descargar/importar ROMs con AutoROM (acepta la licencia automáticamente).
- Tras instalar, activa el entorno: `source .venv-rl2/bin/activate`. Para salir, `deactivate`.

## Entrenar y evaluar el agente en Linux (`prova5a.py`)
- Activa el venv (Linux: `source .venv-rl2/bin/activate`).
- Comando por defecto (entrena + evalúa): `python prova5a.py`. Guarda pesos en `dqn_spaceinvaders_v0_weights.h5f` y logs en `training_log.json`.
- Solo entrenamiento: `python prova5a.py --train.
- Solo test (requiere pesos existentes): `python prova5a.py --test.
- WINDOWS: no funciona el dueling de keras-rl2. Hay que desactivarlo -> enable_dueling_network=False. Hay que implementar el dueling "a mano".
