# NOTAS DE USO

## Instalación del entorno (Linux)
- Ejecuta `./install_keras_rl2.sh --accept-license` para crear `.venv-rl2`, instalar dependencias fijadas (gym 0.17.3, TF 2.5.3, keras-rl2 1.0.5, atari-py) y descargar/importar ROMs con AutoROM (acepta la licencia automáticamente).
- Si ya tienes los ROMs, usa `./install_keras_rl2.sh --rom-path /ruta/a/roms` (omites AutoROM).
- Requisitos: Python 3.8 de 64 bits disponible como `python3.8`; el script recrea el venv si existe.
- Tras instalar, activa el entorno: `source .venv-rl2/bin/activate`. Para salir, `deactivate`.
- Si quieres repetir la instalación limpia, borra `.venv-rl2` o deja que el script lo regenere.

## Entrenar y evaluar el agente (`prova5.py`)
- Activa el venv (`source .venv-rl2/bin/activate`) y asegúrate de tener los ROMs de Atari importados (`python -m atari_py.import_roms <dir>` si falta).
- Comando por defecto (entrena + evalúa): `python prova5.py`. Guarda pesos en `dqn_spaceinvaders_v0_weights.h5f` y logs en `training_log.json`.
- Solo entrenamiento: `python prova5.py --train --steps 3000000 --anneal 200000 --epsmin 0.02 --memory 1000000` (ajusta flags si quieres).
- Solo test (requiere pesos existentes): `python prova5.py --test --episodes 110`.
- Durante test imprime recompensa mínima y media de los últimos 100 episodios y comprueba si `min(last100) > 20`.
- Config clave: `ENV_NAME="SpaceInvaders-v0"`, red dueling DQN con frame stacking (4) y clip de recompensas; semilla fija 42 para reproducibilidad.***
