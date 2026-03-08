# Clasificación de Imágenes con Redes Neuronales: MLP y CNN

## Descripción General del Proyecto

Este proyecto contiene notebooks Python que implementan técnicas de aprendizaje profundo para clasificación de imágenes utilizando dos arquitecturas de redes neuronales diferentes:

1. **MLP (Multilayer Perceptron)**: Clasifica imágenes del dataset CIFAR-100 usando 20 categorías gruesas
2. **CNN (Convolutional Neural Network)**: Clasifica imágenes del dataset CIFAR-10 usando 10 categorías

El objetivo es aprender y comparar diferentes enfoques para visión computarizada, incluyendo preprocesamiento de datos, aumentación de datos, entrenamiento de modelos y evaluación de resultados.

---

## Estructura del Proyecto

```
/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── .gitignore                         # Archivos a ignorar en Git
├── best_model.h5                      # Modelo entrenado guardado
├── classification_cifar10_mlp.ipynb   # Notebook: MLP para CIFAR-100
└── classification_cifar10_cnn.ipynb   # Notebook: CNN para CIFAR-10
```

---

## Configuración del Entorno Virtual

### Paso 1: Crear el Entorno Virtual

```bash
# En Linux/macOS
python3 -m venv venv

# En Windows
python -m venv venv
```

### Paso 2: Activar el Entorno Virtual

```bash
# En Linux/macOS
source venv/bin/activate

# En Windows (cmd)
venv\Scripts\activate.bat

# En Windows (PowerShell)
venv\Scripts\Activate.ps1
```

Cuando el entorno está activado, deberías ver `(venv)` al inicio de tu línea de comandos.

### Paso 3: Instalar las Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Desactivar el Entorno (Opcional)

Cuando termines, desactiva el entorno con:

```bash
deactivate
```

---

## Dependencias

Todas las librerías necesarias están especificadas en `requirements.txt`:

- **TensorFlow >= 2.16.1**: Framework de aprendizaje profundo
- **Keras >= 3.0.0**: API de alto nivel para redes neuronales
- **NumPy >= 1.24.0**: Computación numérica
- **Matplotlib >= 3.7.0**: Visualización de gráficos y imágenes
- **Pandas >= 1.5.0**: Análisis y manipulación de datos
- **SciPy >= 1.10.0**: Herramientas científicas
- **IPython >= 8.0.0**: Shell interactivo mejorado
- **Jupyter >= 1.0.0**: Notebooks interactivos
- **Notebook >= 7.0.0**: Servidor y UI de notebooks

---

## Descripción de los Notebooks

### 1. **classification_cifar10_mlp.ipynb** - Perceptrón Multicapa (MLP)

#### Propósito
Entrenar un Perceptrón Multicapa para clasificar imágenes del dataset CIFAR-100 usando 20 categorías gruesas en lugar de 100 categorías finas.

#### Flujo del Notebook

1. **Carga del Dataset CIFAR-100**
   - Carga 60,000 imágenes de entrenamiento y 10,000 de prueba
   - Cada imagen tiene 32x32 píxeles y 3 canales RGB
   - Usa etiquetas gruesas (20 categorías)

2. **Visualización de Datos**
   - Muestra las primeras 24 imágenes del conjunto de entrenamiento
   - Ayuda a explorar y entender el dataset

3. **Preprocesamiento de Imágenes**
   - Normaliza los valores de píxeles de [0, 255] a [0, 1]
   - Mejora la convergencia durante el entrenamiento

4. **Aumentación de Datos (Data Augmentation)**
   - Aplica transformaciones como:
     - Rotaciones aleatorias (hasta 10 grados)
     - Desplazamientos horizontales y verticales
     - Volteos horizontales aleatorios
     - Zoom aleatorio
   - Aumenta la variabilidad del conjunto de datos sin recopilar nuevas imágenes

5. **Arquitectura MLP**
   - Aplana las imágenes 3D en vectores 1D (32×32×3 = 3,072 características)
   - Utiliza múltiples capas densas con activación ReLU
   - Incluye capas de Dropout para regularización
   - Salida: 20 neuronas con activación softmax (una por categoría)

6. **Entrenamiento del Modelo**
   - Prueba diferentes configuraciones:
     - Diferentes números de épocas
     - Diferentes tasas de aprendizaje
     - Early stopping para prevenir overfitting
   - Optimizadores: RMSprop, Adam, SGD
   - Función de pérdida: categorical crossentropy

7. **Evaluación y Análisis**
   - Calcula métricas: precisión, recall, F1-score
   - Genera matrices de confusión
   - Visualiza curvas de aprendizaje (pérdida y precisión)
   - Compara el desempeño de diferentes modelos

#### Requisitos de Habilidades
- Comprensión básica de redes neuronales
- Conocimiento de preprocesamiento de imágenes
- Familiaridad con normalización y aumentación de datos
- Interpretación de métricas de evaluación

---

### 2. **classification_cifar10_cnn.ipynb** - Redes Convolucionales (CNN)

#### Propósito
Entrenar una Red Convolucional para clasificar imágenes del dataset CIFAR-10 (10 categorías).

#### Flujo del Notebook

1. **Carga del Dataset CIFAR-10**
   - Carga 50,000 imágenes de entrenamiento y 10,000 de prueba
   - Cada imagen tiene 32x32 píxeles y 3 canales RGB
   - 10 categorías de objetos comunes

2. **Visualización y Preprocesamiento**
   - Normaliza imágenes a rango [0, 1]
   - Aplica one-hot encoding a las etiquetas

3. **División del Dataset**
   - Separa datos en: entrenamiento, validación y prueba
   - Validación se usa para monitorear durante el entrenamiento

4. **Arquitectura CNN**
   - Capas convolucionales para extraer características
   - Capas de pooling para reducir dimensionalidad
   - Capas densas para clasificación final
   - Mejor captura de características espaciales que MLP

5. **Entrenamiento y Evaluación**
   - Monitorea métricas en validación
   - Guarda el mejor modelo
   - Evalúa en conjunto de prueba

#### Características Principales
- Capas convolucionales optimizadas para imágenes
- Mejor desempeño que MLP en tareas de visión
- Menos parámetros que MLP para el mismo rendimiento

---

## Cómo Ejecutar los Notebooks

### Opción 1: Usando Jupyter Notebook

```bash
# Asegúrate de tener el entorno virtual activado
source venv/bin/activate

# Inicia Jupyter Notebook
jupyter notebook

# Se abrirá un navegador en http://localhost:8888
# Selecciona el notebook que deseas ejecutar
```

### Opción 2: Usando Jupyter Lab

```bash
# Asegúrate de tener el entorno virtual activado
source venv/bin/activate

# Inicia Jupyter Lab
jupyter lab

# Se abrirá en http://localhost:8888/lab
```

### Opción 3: Usando VS Code

1. Abre VS Code
2. Abre la carpeta del proyecto
3. Abre un notebook (.ipynb)
4. Selecciona el kernel de Python desde el entorno virtual
5. Ejecuta las celdas con Shift+Enter

---

## Ejemplos de Uso

### Ejecutar el MLP
```python
# Cargar librerías necesarias
import keras
from keras.datasets import cifar100
import numpy as np

# Cargar datos
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

# Normalizar
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# El notebook contiene el código completo para entrenar
```

### Entrenar y Guardar un Modelo
```python
# El modelo se entrena y puede guardarse
model.save('best_model.h5')

# Para cargar un modelo guardado
from keras.models import load_model
model = load_model('best_model.h5')

# Hacer predicciones
predictions = model.predict(x_test)
```

---

## Solución de Problemas

### Error: "Module not found"
**Solución**: Asegúrate de tener el entorno virtual activado y haber instalado los requisitos:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "No module named 'tensorflow'"
**Solución**: Reinstala TensorFlow:
```bash
pip install --upgrade tensorflow
```

### Error: "Notebook kernel not found"
**Solución**: Instala ipykernel en el entorno virtual:
```bash
pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python (venv)"
```

### Error: Memoria insuficiente
**Solución**: 
- Reduce el tamaño del batch durante el entrenamiento
- Usa un modelo más pequeño
- Únicamente entrena en conjuntos de datos más pequeños

---

## Notas Importantes

1. **Datos**: CIFAR-10 y CIFAR-100 se descargan automáticamente la primera vez que ejecutas los notebooks.

2. **Tiempo de Entrenamiento**: El entrenamiento puede tomar varios minutos depending en tu hardware (CPU vs GPU).

3. **GPU**: Si tienes una GPU NVIDIA compatible, TensorFlow la detectará automáticamente:
   ```bash
   python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info()['cuda_version'])"
   ```

4. **Reproducibilidad**: Los notebooks fijan semillas aleatorias para reproducir resultados.

5. **Modelos Guardados**: El archivo `best_model.h5` contiene un modelo ya entrenado que puede cargarse sin necesidad de reentrenar.

---

## Referencias

- [Dataset CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Deep Learning for Vision Systems](https://github.com/moelgendy/deep_learning_for_vision_systems)

---

## Autor

Proyecto de taller - Clasificación de imágenes con aprendizaje profundo

## Licencia

Este proyecto está disponible bajo licencia MIT.
