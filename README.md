# NOTAS DE USO

## Instalación del entorno (Linux)
- Ejecuta `./install_keras_rl2.sh --accept-license` para crear `.venv-rl2`, instalar dependencias fijadas (gym 0.17.3, TF 2.5.3, keras-rl2 1.0.5, atari-py) y descargar/importar ROMs con AutoROM (acepta la licencia automáticamente).
- Tras instalar, activa el entorno: `source .venv-rl2/bin/activate`. Para salir, `deactivate`.

## Instalación del entorno (Windows, PowerShell)
- Abre PowerShell en la carpeta del proyecto y ejecuta `.\install_keras_rl2.ps1 -AcceptLicense` para crear `.venv-rl2`, instalar dependencias fijadas y descargar/importar ROMs con AutoROM (acepta la licencia automáticamente).
- Tras instalar, activa el entorno: `.\.venv-rl2\Scripts\Activate.ps1`. Para salir, `deactivate`.

## Entrenar y evaluar el agente (`prova5a.py`)
- Activa el venv (Linux: `source .venv-rl2/bin/activate`; Windows: `.\.venv-rl2\Scripts\Activate.ps1`).
- Comando por defecto (entrena + evalúa): `python prova5a.py`. Guarda pesos en `dqn_spaceinvaders_v0_weights.h5f` y logs en `training_log.json`.
- Solo entrenamiento: `python prova5a.py --train.
- Solo test (requiere pesos existentes): `python prova5a.py --test.
