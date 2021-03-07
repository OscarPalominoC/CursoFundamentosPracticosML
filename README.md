# Curso de Fundamentos Prácticos de Machine Learning

![Logo](https://static.platzi.com/media/achievements/badges-fundamentos-machine-learning-5e2fbe0b-271c-4441-846a-f702108dc4f0.png)

---

**Profesora**: Yesi Díaz

# Proyecto del curso
## Clasificador de imágenes

Crea una red neuronal con Tensor Flow para predecir qué tipo de prenda de vestir corresponde a la imagen.

---

# Archivos



# Indice

1. [Fundamentos prácticos](#fundamentos-prácticos)
    * [Introducción al Curso](#introducción-al-curso)
    * [Introducción a Numpy](#introducción-a-numpy)
    * [Introducción y manipulación de datos con Pandas](#introducción-y-manipulación-de-datos-con-pandas)
    * [Introducción a ScikitLearn](#introducción-a-scikitlearn)
    * [Artículo:  Comandos básicos de las librerías usadas en el curso (Numpy, Pandas y ScikitLearn)](#comandos-básicos-de-las-librerías-usadas-en-el-curso-numpy-pandas-y-scikitlearn)
2. [Regresión Lineal y Logística](#regresión-lineal-y-logística)
    * [¿Qué es la predicción de datos?](#qué-es-la-predicción-de-datos)
    * [Sobreajuste y subajuste en los datos](#sobreajuste-y-subajuste-en-los-datos)
    * [Regresión lineal simple y regresión lineal múltiple](#regresión-lineal-simple-y-regresión-lineal-múltiple)

---

# Fundamentos prácticos

## Introducción al Curso

### Machine Learning

Capacidad de un algoritmo de adquirir conocimiento a partir de los datos recolectados para mejorar, describir y predecir resultados

Estrategias de Aprendizaje:

* Aprendizaje Supervisado: Permite al algoritmo aprender a partir de datos previamente etiquetados.
* Aprendizaje no Supervisado: El algoritmo aprende de datos sin etiquetas, es decir encuentra similitudes y relaciones, agrupando y clasificando los datos.
* Aprendizaje Profundo (Deep Learning): Está basado en redes Neuronales

### Importancia del ML

Permite a los algoritmos aprender a partir de datos históricos recolectados por las empresas permitiendo así tomar mejores decisiones.

Pasos a Seguir para Desarrollar un modelo en ML

* Definición del Problema: Es necesario definir previamente el problema que va a resolver nuestro algoritmo
* Construcción de un modelo y Evaluación: Una vez definido el problema procedemos a tratar los datos y a entrenar nuestro modelo que debe tener una evaluación cercana al 100%
* Deploy y mejoras: El algoritmo es llevado a producción (aplicación o entorno para el que fue creado), en este entorno podemos realizar las mejoras pertinentes de acuerdo al comportamiento con los usuario e incluso aprovechando los datos generados en esta interacción

## Introducción a Numpy

​​Numpy (Numerical Python)Biblioteca de Python sencilla de usar y muy rápida, adecuada para el manejo de arreglos (álgebra lineal).

* Es necesario tenerla instalada para usarla (en caso de usar Colab este paso ya está hecho)
* Hay que importar la librería en nuestro código:
```py
import numpy as np
```
La biblioteca trabaja principalmente con arreglos (vectores, matrices y tensores). Para declarar un arreglo escribimos:
```py
np.array(<arreglo>)
```
Podemos asignar una cabecera a nuestros valores de la siguiente manera:
```py
cabecera = [('nobre_datos',tipo_datos)]
datos = [(data)]
arreglo = np.array(datos,dtype=cabecera)
```
#ejemplo:
```py
headers = [('Nombre', 'S10'),('edad',int),('país','S15')]  #'S10' significa string de tamaño 10
data = [('Manuel',12,'Bolivia'),('Jose',10,'Paraguay'),('Daniel',5,'Venezuela'),('Ivon',30,'Chile'),('Lupe',28,'Mexico')]
people = np.array(data,dtype=headers)
```
Para tomar parte de un arreglo usamos slice notation (de manera similar a las listas):
```py
mi_arreglo = np.array([1,2,3],[4,5,6],[7,8,9])
mi_arreglo[1][1] #devuelve 5
mi_arreglo[1:] #devuelve la segunda y tercera fila
```
* Comandos utiles.
    * np.zeros(dimension) → crea un arreglo de dimensión indicada con todos los valores iguales a 0
    * np.ones(dimension) → crea un arreglo de dimensión indicada con todos los valores iguales a 1
    * np.linspace(inicio,fin,n_datos) → crea un arreglo de n datos igualmente espaciados con un inicio y fin indicados
    * arreglo.ndim → devuelve la dimensión del arreglo
    * np.sort(arreglo) → devuelve el arreglo ordenado (en caso de tener encabezado se puede usar el atributo order=’<nombre_dato>’)
    * np.arrange(inicio,fin,pasos) → crea un arreglo con inicio y fin dado aumentando el paso
    * np.full(dimension,valores) → crea un arreglo de la dimensión data cuyos componentes son llenados con los valores
    * np.diag(diagonal,k) → crea una matriz diagonal con los valores dados (k es opcional y determina el número de columnas (signo negativo) o filas(signo positivo) a aumentar en caso de que la matriz no sea cuadrada)

[Notebook Collab generado](https://colab.research.google.com/drive/1Z4uspHbjm0hkWE3GAvebAd82FRkOiBqb#scrollTo=FT4619jcj5Wv)

[Notebook local](/code/1_Numpy.ipynb)

## Introducción y manipulación de datos con Pandas

**Pandas (Panel Data)**

* Es una librería derivada de numpy pensada para el manejo de datos en forma de panel.
* Trabaja con series, data frames y panels, generan indices a los arreglos (a manera de panel o tabla de excel), en una (Series), dos (DataFrames) o tres dimensiones (Panels)

Al igual que numpy hay que importar primero la librería:
```py
import pandas as pd
```
**Creación de series y dataframes:**
```py
serie = pd.Series(<datos>)
dataframe = pd.DataFrame(<datos>)
```
El código anterior genera una serie o dataFrame a partir de los datos ingresados, pueden ser listas, tuplas o , en el caso de un DataFrame, arreglos o diccionarios (en el caso de los diccionarios los indices de las columnas toman el valor de las llaves)

**Algunos comandos útiles**
* `df[nombre_columna]` → selecciona la columna invocada
* `pd.read_csv(nombre_archivo)` → permite importar datos a partir de un csv
* `type(objeto)` → permite saber el tipo de objeto
* `df.shape` → Devuelve el tamaño del df
* `df.colums` → Devuelve las columnas del df
* `df[nombre_columna].describe()` → devuelve un análisis estadístico de la columna
* `df.sort_index()` → ordena de acuerdo al índice (axis=0 → filas; axis=1 → columnas)
* `df.sort_values(by=campo_a_ordenar)` → ordena de acuerdo a los valores

[Notebook Collab generado](https://colab.research.google.com/drive/18LHUd_nAoi4S3xH1C-S2x5Sl13IpuO_2#scrollTo=qefWcdB7mkuO)

[Notebook local](/code/2_Pandas.ipynb)

## Introducción a ScikitLearn

**¿QUÉ ES SCIKIT-LEARN?**

Scikit-Learn es una de estas librerías gratuitas para Python. Cuenta con algoritmos de clasificación, regresión, clustering y reducción de dimensionalidad. Además, presenta la compatibilidad con otras librerías de Python como NumPy, SciPy y matplotlib.

La gran variedad de algoritmos y utilidades de Scikit-learn la convierten en la herramienta básica para empezar a programar y estructurar los sistemas de análisis datos y modelado estadístico. Los algoritmos de Scikit-Learn se combinan y depuran con otras estructuras de datos y aplicaciones externas como Pandas o PyBrain.

La ventaja de la programación en Python, y Scikit-Learn en concreto, es la variedad de módulos y algoritmos que facilitan el aprendizaje y trabajo del científico de datos en las primeras fases de su desarrollo. La formación de un Máster en Data Science hace hincapié en estas ventajas, pero también prepara a sus alumnos para trabajar en otros lenguajes. La versatilidad y formación es la clave en el campo tecnológico.

## Comandos básicos de las librerías usadas en el curso (Numpy, Pandas y ScikitLearn)

### Numpy

Biblioteca de Python comúnmente usada en la ciencias de datos y aprendizaje automático (Machine Learning). Proporciona una estructura de datos de matriz que tiene diversos beneficios sobre las listas regulares.

Importar la biblioteca:
```
import numpy as np
```
Crear arreglo unidimensional:
```
my_array = np.array([1, 2, 3, 4, 5])
Resultado: array([1, 2, 3, 4, 5])
```
Crear arreglo bidimensional:
```
np.array( [[‘x’, ‘y’, ‘z’], [‘a’, ‘c’, ‘e’]])
Resultado:
[[‘x’ ‘y’ ‘z’]
[‘a’ ‘c’ ‘e’]]
```
Mostrar el número de elementos del arreglo:
```
len(my_array)
```
Sumar todos los elementos de un arreglo unidimensional:
```
np.sum(my_array)
```
Obtener el número máximo de los elementos del arreglo unidimensional
```
np.max(my_array)
```
Crear un arreglo de una dimensión con el número 0:
```
np.zeros(5)
Resultado: array([0., 0., 0., 0., 0.])
```
Crear un arreglo de una dimensión con el número 1:
```
np.ones(5)
Resultado: array([1., 1., 1., 1., 1.])
```
Comando de Python para conocer el tipo del dato:
```
type(variable)
```
Ordenar un arreglo:
```
np.order(x)
```
Ordenar un arreglo por su llave:
```
np.sort(arreglo, order = ‘llave’)
```
Crear un arreglo de 0 a N elementos:
```
np.arange(n)
Ej.
np.arange(25)
Resultado:
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24])
```
Crear un arreglo de N a M elementos:
```
np.arange(n, m)
Ej.
np.arange(5, 30)
Resultado:
array([ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
22, 23, 24, 25, 26, 27, 28, 29])
```
Crear un arreglo de N a M elementos con un espacio de cada X valores:
```
np.arange(n, m, x)
Ej.
np.arange(5, 50, 5)
Resultado:
array([ 5, 10, 15, 20, 25, 30, 35, 40, 45])
```
Crear arreglo de NxM:
```
len(my_array)
```
Número de elementos del arreglo:
```
len(my_array)
```
### Pandas

Pandas es una herramienta de manipulación de datos de alto nivel, es construido con la biblioteca de Numpy. Su estructura de datos más importante y clave en la manipulación de la información es DataFrame, el cuál nos va a permitir almacenar y manejar datos tabulados observaciones (filas) y variables (columnas).

Importar la biblioteca:
```
import pandas as pd
```
Generar una serie con Pandas:
```
pd.Series([5, 10, 15, 20, 25])
Resultado:
0 5
1 10
2 15
3 20
4 25
```
Crear un Dataframe:
```
lst = [‘Hola’, ‘mundo’, ‘robótico’]
df = pd.DataFrame(lst)
Resultado:
0
0 Hola
1 mundo
2 robótico
```
Crear un Dataframe con llave y dato:
```
data = {‘Nombre’:[‘Juan’, ‘Ana’, ‘Toño’, ‘Arturo’],
‘Edad’:[25, 18, 23, 17],
‘Pais’: [‘MX’, ‘CO’, ‘BR’, ‘MX’] }
df = pd.DataFrame(data)
```
Resultado:
![DF](/images/df.jpg)

Leer archivo CSV:
```
pd.read_csv(“archivo.csv”)
```
Mostrar cabecera:
```
data.head(n)
```
Mostrar columna del archivo leído:
```
data.columna
```
Mostrar los últimos elementos:
```
data.tail()
```
Mostrar tamaño del archivo leído:
```
data.shape
```
Mostrar columnas:
```
data.columns
```
Describe una columna:
```
data[‘columna’].describe()
```
Ordenar datos del archivo leído:
```
data.sort_index(axis = 0, ascending = False)
```
### Scikit Learn

Scikit Learn es una biblioteca de Python que está conformada por algoritmos de clasificación, regresión, reducción de la dimensionalidad y clustering. Es una biblioteca clave en la aplicación de algoritmos de Machine Learning, tiene los métodos básicos para llamar un algoritmo, dividir los datos en entrenamiento y prueba, entrenarlo, predecir y ponerlo a prueba.

Importar biblioteca:
```
from sklearn import [modulo]
```
División del conjunto de datos para entrenamiento y pruebas:
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
Entrenar modelo:
```
[modelo].fit(X_train, y_train)
```
Predicción del modelo:
```
Y_pred = [modelo].predict(X_test)
```
Matriz de confusión:
```
metrics.confusion_matrix(y_test, y_pred)
```
Calcular la exactitud:
```
metrics.accuracy_score(y_test, y_pred)
```

# Regresión Lineal y Logística

## ¿Qué es la predicción de datos?

El análisis predictivo agrupa una variedad de técnicas estadísticas de modelización, aprendizaje automático y minería de datos que analiza los datos actuales e históricos reales para hacer predicciones acerca del futuro o acontecimientos no conocidos.

Existen algoritmos que se definen como “clasificadores” y que identifican a qué conjunto de categorías pertenecen los datos.

Para entrenar estos algoritmos:

* Es importante comprender el problema que se quiere solucionar o que es lo que se quiere aplicar.
* Obtener un conjunto de datos para entrenar el modelo.

Cuando entrenamos un modelo para llevar a cabo una prueba, es importante cuidar la información que se le suministra, es decir, debemos verificar si existen datos no validos o nulos, si las series de datos esta completa, etc.

### Ejemplo de predicción

Podemos entrenar un modelo con la información historica de cierta cantidad de estudiantes y sus calificaciones en diferentes cursos, un modelo bien entrenado con estos datos debería ser capas de hacer una predicción de que tan bien le irá a un estudiante nuevo en algun tipo de curso al evaluar sus carácteristicas.

* Entrenamos nuestro algoritmo con datos históricos con el fin de que nos pueda proporcionar información util para nuestro fin (una predicción acertada).
* Para ello debemos elegir un modelo, entrenarlo y probar si nuestro modelo acierta.
* La elección del modelo no es al azar, debemos conocer el problema que queremos resolver y obtener la información necesaria.
* Es importante que la información con la que trabajamos esté bien tratada (evitar datos nulos o faltantes) y evitar los datos irrelevantes.

## Sobreajuste y subajuste en los datos

Generalización en Machine Learning

La capacidad de generalización nos indica qué tan bien los conceptos aprendidos por un modelo de aprendizaje automático se aplican a ejemplos específicos que el modelo no vio cuando estaba aprendiendo. El objetivo de un buen modelo de aprendizaje automático es generalizar bien los datos de entrenamiento. Esto nos permite hacer predicciones en el futuro sobre los datos que el modelo nunca ha visto. Sobreajuste y subajuste son terminologías empleados en el aprendizaje automático para hacer referencia a qué tan bien un modelo generaliza nuevos datos ya que el ajuste excesivo y el ajuste insuficiente son las dos causas principales del rendimiento deficiente de los algoritmos de aprendizaje automático.

**Sobreajuste**

El sobreajuste hace referencia a un modelo que se sobre-entrena considerando cada mínimo detalle de los datos de entrenamiento. Esto significa que el ruido o las fluctuaciones aleatorias en los datos de entrenamiento son recogidos y aprendidos como conceptos por el modelo. El problema es que estos conceptos no se aplican a nuevos datos y tienen un impacto negativo en la capacidad de los modelos para generalizar.

Este sobre-entrenamiento suele darse con mayor probabilidad en modelos no lineales, por ello muchos de estos algoritmos de aprendizaje automático también incluyen parámetros o técnicas para limitar y restringir la cantidad de detalles que aprende. Algunos ejemplos de algoritmos no lineales son los siguientes:

* Decision Trees
* Naive Bayes
* Support Vector Machines
* Neural Networks

![Ajuste](/images/ajuste.jpg)

### Subajuste

El subajuste hace referencia a un modelo que no puede modelar los datos de entrenamiento ni generalizar a nuevos datos. Un modelo de aprendizaje automático insuficiente no es un modelo adecuado. Las estrategias para mitigar un ajuste insuficiente son variadas y dependen del contexto.

Como puede deducirse, el subajuste suele darse con mayor probabilidad en modelos lineales, como por ejemplo:

* Logistic Regression
* Linear Discriminant Analysis
* Perceptron

![Ajuste](/images/underfitting-overfitting.png)

### Consejos a tener en cuenta

* Buscar en lo posible dar variedad a los datos buscando todas la posibilidades para así evitar un sesgo en nuestro algoritmo
* Dividir nuestros datos en datos de aprendizaje y datos de evaluación (aproximadamente 70-30)

[Youtube: Las Redes Neuronales... ¿Aprenden o Memorizan? - Overfitting y Underfitting - Parte 1](https://www.youtube.com/watch?v=7-6X3DTt3R8)

## Regresión lineal simple y regresión lineal múltiple

El algoritmo de regresión lineal nos ayuda a conseguir tendencia en los datos, este es un algoritmo de tipo supervisado ya que debemos de usar datos previamente etiquetados.

En la **regresión lineal** generamos, a partir de los datos, una recta y es a partir de esta que podremos encontrar la tendencia o predicción.

Generalmente es importante tener en cuenta varias dimensiones o variables al considerar los datos que estamos suministrando al modelo, recordando siempre cuidar este set de sobreajuste o subajuste.

Cuando nuestro modelo considera más de dos variables el algoritmo de regresión que usamos se conoce como **Regresión Lineal Múltiple** y este trabaja sobre un sistema de referencia conocido como hiperplano.

Los algoritmos de regresión, tanto lineal como múltiple trabajan únicamente con datos de tipo cuantitativos.

[Youtube: Regresión Lineal y Mínimos Cuadrados Ordinarios | DotCSV](https://www.youtube.com/watch?v=k964_uNn3l0)

[Notebook Collab generado](https://colab.research.google.com/drive/1YWl8gLsX5eTBefYxopJyN5OgnBxBetDB)

[Notebook local](/)