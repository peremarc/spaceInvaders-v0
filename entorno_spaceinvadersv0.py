#!/usr/bin/env python
# coding: utf-8
#==================================================================================================
#   08MIAR 2025 Grupo 21 - Fichero de instalacion y preparación entorno Proyecto Programación RL
#==================================================================================================
# Las prácticas han sido preparadas para poder realizarse en el entorno de trabajo de Google Colab. 
# Sin embargo, esta plataforma presenta ciertas incompatibilidades a la hora de visualizar la 
# renderización en gym. Por ello,para obtener estas visualizaciones,se deberá trasladar el entorno 
# de trabajo a local. 
# Por ello, el presente dosier presenta instrucciones para poder trabajar en ambos entornos. 
# Siga los siguientes pasos para un correcto funcionamiento:
#  1.**LOCAL:** Preparar el enviroment, siguiendo las intrucciones detalladas en la sección 
#               *1.1.Preparar enviroment*.
#  2.**AMBOS:** Modificar las variables "mount" y "drive_mount" a la carpeta de trabajo en drive en 
#               el caso de estar en Colab, y ejecturar la celda *1.2.Localizar entorno de trabajo*.
#  3.**COLAB:** Se deberá ejecutar las celdas correspondientes al montaje de la carpeta de trabajo  
#               en Drive.Esta corresponde a la sección *1.3.Montar carpeta de datos local*.
#  4.**AMBOS:** Instalar las librerías necesarias, siguiendo la sección *1.4.Instalar librerías 
#               necesarias*.
#----------------------------- 0. Importación Librerias y funciones -------------------------------
import sys
import importlib
import subprocess
from importlib.metadata import PackageNotFoundError, version as get_version

#------------------------------------------- 1. Parametros ----------------------------------------
VERSION_PYTHON_LOCAL = "3.8"
VERSION_PYTHON_COLAB = "3.11"
FOLDER_COLAB = "/My Drive/08_MIAR/actividades/proyecto practico"


#-------------------------- 2. Localizar entorno de trabajo: Google colab o local -----------------
# ATENCIÓN!! Modificar ruta relativa a la práctica si es distinta (drive_root)
mount='/content/gdrive'
drive_root = mount + FOLDER_COLAB

# Detectar si estamos en Colab
try:
  from google.colab import drive
  IN_COLAB=True
except:
  IN_COLAB=False

#---------------------------- 3. Montar carpeta de datos local (solo Colab) -----------------------
# Switch to the directory on the Google Drive that you want to use
import os

if IN_COLAB:
    print("Estamos en Colab:", sys.executable)

    from google.colab import drive
    print("Colab: mounting Google drive on", mount)
    drive.mount(mount)

    # Crear carpeta si no existe
    print("\nColab: making sure", drive_root, "exists.")
    os.makedirs(drive_root, exist_ok=True)

    # Cambiar directorio
    print("\nColab: Changing directory to", drive_root)
    os.chdir(drive_root)

else:
    print("Estamos en entorno Local:", sys.executable)

# Mostrar directorio actual
print("Directorio actual:", os.getcwd())
print("Archivos en el directorio:")
print(os.listdir())

#---------------------------- 4. Funcion que instala paquetes sino existen ya ---------------------
# Funcion que instala paquetes sino existen ya.
def instalar_si_falta(paquete_pip, version=None):
    """
    paquete_pip: nombre del paquete pip (ej: 'Pillow', 'tensorflow', 'keras-rl2')
    version: número de versión (ej: '2.5.0') o URL git (ej: 'git+https://...')
    """

    # Comprobar si el paquete está instalado sin importarlo
    try:
        get_version(paquete_pip)
        print(f"✔ {paquete_pip} ya está instalado.")
        return
    except PackageNotFoundError:
        print(f"📦 {paquete_pip} no está instalado. Instalando...")

    # Construir comando pip
    if version is None:
        paquete_instalar = paquete_pip
    else:
        if version.startswith("git+"):
            paquete_instalar = version
        elif "==" in version:
            paquete_instalar = version
        else:
            paquete_instalar = f"{paquete_pip}=={version}"

    # Ejecutar instalación
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete_instalar])
    print(f"✔ Instalado: {paquete_instalar}")

#------------------------------- 5. Instalar librerías necesarias ---------------------------------
# Capturamos version de Python
version_python = f"{sys.version_info.major}.{sys.version_info.minor}"

# En google colab se instala siempre los paquetes en el entorno porque no lo guarda entre sesiones.
if IN_COLAB:
    print(f"\nLa version de Python instalada en Colab es: {version_python}")
    if (version_python == VERSION_PYTHON_COLAB):
        print(f"\nComprobando Paquetes Instalados....\n")
        instalar_si_falta("gym", "0.17.3")
        instalar_si_falta("atari_py", "git+https://github.com/Kojoley/atari-py.git")
        instalar_si_falta("keras-rl2", "1.0.5")
        instalar_si_falta("tensorflow", "2.8")
        print("\nEl entorno esta listo.")
    else:
        print(f"Para ejecutar correctamente este notebook se requiere un entorno con Python {VERSION_PYTHON_COLAB}")
        print(f"Crear un entorno vacio con la version de python requerida y volver a ejecutar esta celda.")
# En local solo lo vamos a instalar sino ha sido instalado ya.
else:
    print(f"\nLa version de Python instalada en este entorno Local es: {version_python}")
    
    if (version_python == VERSION_PYTHON_LOCAL):
        print(f"\nComprobando Paquetes Instalados....\n")
        instalar_si_falta("numpy", "1.19.5")
        instalar_si_falta("Keras", "2.2.4")
        instalar_si_falta("keras-rl2", "1.0.5")
        instalar_si_falta("tensorflow", "2.5.3")
        instalar_si_falta("gym", "0.17.3")
        instalar_si_falta("atari_py", "git+https://github.com/Kojoley/atari-py.git")
        instalar_si_falta("pyglet", "1.5.0")
        instalar_si_falta("h5py", "3.1.0")
        instalar_si_falta("Pillow", "9.5.0")
        instalar_si_falta("opencv-python-headless", "4.7.0.72")
        instalar_si_falta("matplotlib", "3.3.4")
        print("\nEl entorno esta listo.")
    else:       
        print(f"Para ejecutar correctamente este notebook se requiere un entorno con Python {VERSION_PYTHON_LOCAL}")
        print(f"Crear un entorno vacio con la version de python requerida y volver a ejecutar esta celda.")
        
