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
    * [Regresión lineal simple con Scikit-Learn: división de los datos](#regresión-lineal-simple-con-scikit-learn-división-de-los-datos)
    * [Regresión lineal simple con Scikit-Learn: creación del modelo](#regresión-lineal-simple-con-scikit-learn-creación-del-modelo)
    * [Regresión logística con Scikit-Learn: definición y división de datos](#regresión-logística-con-scikit-learn-definición-y-división-de-datos)
    * [Regresión logística con Scikit-Learn: evaluación del modelo](#regresión-logística-con-scikit-learn-evaluación-del-modelo)
    * [Artículo: Matriz de confusión](#artículo-matriz-de-confusión)
3. [Árboles de decisión](#árboles-de-decisión)
    * [¿Qué es un árbol de decisión y cómo se divide?](qué-es-un-árbol-de-decisión-y-cómo-se-divide)
    * [Comprendiendo nuestro data set para la creación de un árbol de decisión](#comprendiendo-nuestro-data-set-para-la-creación-de-un-árbol-de-decisión)
    * [Creando un clasificador con Scikit-Learn](#creando-un-clasificador-con-scikit-learn)
    * [Entrenamiento del modelo de clasificación](#entrenamiento-del-modelo-de-clasificación)
    * [Visualización del árbol de decisión](#visualización-del-árbol-de-decisión)
4. [K-Means](#k-means)
    * [¿Qué es K-Means?](#qué-es-k-means)
    * [Cargando el data set de Iris](#cargando-el-data-set-de-iris)
    * [Construcción y evaluación del modelo con K-Means](#construcción-y-evaluación-del-modelo-con-k-means)
    * [Graficación del modelo](#graficación-del-modelo)
5. [Aprendizaje profundo](#aprendizaje-profundo)
    * [Introducción al aprendizaje profundo](#introducción-al-aprendizaje-profundo)
    * [Artículo: Conceptos básicos de Tensor Flow](#artículo-conceptos-básicos-de-tensor-flow)


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

## Regresión lineal simple con Scikit-Learn: división de los datos

Procedimiento de Regresión Linear Simple con Scikit-Learn:

Importar las librerías y modelos necesarios:
* pandas: import pandas as pd → para manejo de datos
* matplotlib: import matplotlib.pyplot as plt → permite insertar gráficos estadísticos
* train_test_split: from sklearn import train_test_split → este módulo permite separar nuestros dato en datos de entrenamiento y prueba
* LinearRegression: from sklearn import LinearRegression → El modelo en sí

Importar los datos (en este caso desde un archivo csv):
```py
dataset = pd.read_csv(<nombre_archivo>)
```
Asignar los datos a sus respectivas variables:
```py
x = dataset.iloc[<slice>]
y = dataset.iloc[<slice>] #iloc sólo admite slice notation para la selección de parte de los datos
```
Separar los datos en datos de prueba y datos de test:
```py
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2, random_state=5)
# test_size → porcentaje de los datos de prueba a tomar
# random_state = 0 → tipo de selección de datos (a un mismo valor devuelve la misma selección)
```

## Regresión lineal simple con Scikit-Learn: creación del modelo

**Asignamos nuestro módulo a una variable:**
```py
regressor = LinearRegression() 
# Este paso es necesario para que el entrenamiento se guarde en esta variable, además es más fácil cambiar de modelo en caso sea necesario
```
**Entrenamos nuestro algoritmo (todos los modelos se entrenan así)**
```py
regressor.fit(X_train,Y_train) 
# LinearRegression().fit(X_train,Y_train) funciona pero no se guardan los resultados
```
* Ya tenemos entrenado nuestro algoritmo, así podemos extraer los datos del entrenamiento:
    * predictor.coef_ → devuelve el coeficiente (a) de la recta de regresión (ax+b=0)
    * predictor.intercept_ → devuelve el punto de corte con el eje y (b) de la recta de regresión (ax+b=0)
    * predictor.predict(<val>) → devuelve la predicción del dato de entrada val

**Bonus: Graficado de Resultados**

Usaremos la librería matplotlib.pyplot para realizar nuestros gráficos, recordemos que ya la importamos:
```py
import matplotlib.pyplot as plt
```
Podemos darle un alias a la librería (no lo considero necesario):
```py
viz_train = plt #Yo trabajaré solo con plt
```
Para visualizar una nube de puntos usamos el método scatter:
```py
plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='red')
```
Para visualizar una línea que una los puntos usamos el método plot:
```py
plt.plot(X_train, regressor.predict(X_train), color = 'black') 
#regressor.predict(X_train) devuelve los valores de predicción para los valores de X_train, graficamos así la línea de regresión
```

Ahora para visualizar el gráfico usamos el método show:
```py
plt.show()
```

### El método score
```py
regressor.score(X_test, Y_test)
```
Este método nos devuelve un número entre 0 y 1, es la probabilidad de predecir correctamente los datos de prueba, ¿Cómo podemos mejorar el resultado? Mejorando los datos, aumentando el tamaño del dataset, posiblemente usando una regresión líneal múltiple (evalúa si es programador backend, frontend, la tecnología que usa,etc.).

[Notebook Collab generado](https://colab.research.google.com/drive/1YWl8gLsX5eTBefYxopJyN5OgnBxBetDB)

[Notebook local](/code/3_regresion_lineal_simple.ipynb)

## Regresión logística con Scikit-Learn: definición y división de datos

Modelo de regresión que está más enfocado en la clasificación de datos cualitativos, entregándonos como resultado un 0 o 1 (sí o no). 

![Regresión logística](/images/logistica.jpg)

### Implementación del código

**Importar las librerías necesarias**
```py
import pandas as pd
from sklearn.model_selection import train_test_split #permite dividir nuestros datos
from sklearn import metrics #nos permite evaluar nuestro modelo
from sklearn.linear_model import LogisticRegression() #modelo que vamos a aplicar
import matplotlib.pyplot as plt
import seaborn as sns #nos permite mejorar la presentación de los gráficos
%matplotlib inline #nos permite insetar gráficos en el notebook
```
**Importamos nuestros datos y generamos nuestras variables**
```py
diabetes = pd.read_csv(<nombre_archivo>)
feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'] #columnas presentes en el dataset
x= diabetes[feature_cols]
y= diabetes[['Outcome']]
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.25, random_state=0)
```
**Entrenamos el modelo**
```py
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_test) #Recogemos las predicciones que entrega nuestro modelo para los datos de prueba
```

## Regresión logística con Scikit-Learn: evaluación del modelo

### Matriz de confusión

![Matriz de confusión](/images/matriz-confusion.png)

Representación gráfica que nos permite ver el grado de acierto de nuestro modelo. El gráfico tiene cuatro divisiones: Verdaderos Positivos (VP), Falsos Positivos (FP), Falsos Negativos (FN) y Verdaderos Negativos (VN). Siendo los datos verdaderos los que nos interesa maximizar (valores de la diagonal).

**Graficado de la Matriz de Confusion:**

Los datos necesarios los obtenemos de nuestro modelo (con ayuda del módulo metrics):
```py
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
```
**Definimos los ejes con sus respectivas etiquetas**
```py
class_names = [0,1]
fig,ax = plt.subplots() #obtenemos las variables figura y ejes del gráfico (nos permite cambiar los atributos propios de cada seccion)
tick_marks = np.arange(len(class_names))  #definimos los valores que van a tener las líneas de guía
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names) #definimos los tick marks en el gráfico
```
**Creamos la presentación del gráfico**
```py
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap='Blues_r', fmt='g')
ax.axis.set_label_position('top')
plt.tight_layout()
plt.title('Matriz de confusion', y=1.1)
plt.y_label('Etiqueta Actual')
plt.x_label('Etiqueta de Prediccion)
```
Seaborn nos permite crear un mapa de calor a partir de los valores entregados (la matriz de confusión en este caso), los parámetros que entregamos son: annot → permite colocar los valores sobre el gráfico, cmap → estilo del gráfico, fmt → formato de los valores
- ax.xaxis,set_label_position() → nos permite definir donde colocar la etiqueta del eje x
- plt.tight_layout() → crea un padding en torno al gráfico (lo enmarca)

**Nota**

Otra forma de evaluar nuestro modelo es a través del método accuracy_score del módulo metrics:
```py
metrics.accuracy_score(Y_test,y_pred)
```

[Notebook Collab generado](https://colab.research.google.com/drive/1VZJr08ADqrebfjCBEkKA34ht8hOkDDZp?usp=sharing)

[Notebook local](/code/4_regresion_logistica.ipynb)

## Artículo: Matriz de confusión

Los modelos de clasificación son capaces de predecir cuál es la etiqueta correspondiente a cada ejemplo o instancia basado en aquello que ha aprendido del conjunto de datos de entrenamiento. Estos modelos necesitan ser evaluados de alguna manera y posteriormente comparar los resultados obtenidos con aquellos que fueron entrenados.

Una manera de hacerlo es mediante la matriz de confusión la cual nos permite evaluar el desempeño de un algoritmo de clasificación a partir del conteo de los aciertos y errores en cada una de las clases del algoritmo.

Como su nombre lo dice tenemos una matriz que nos ayuda a evaluar la predicción mediante positivos y negativos como se muestra en la figura.

![Matriz de confusión](/images/matriz-confusion.png)

* **Los verdaderos positivos (VP)** son aquellos que fueron clasificados correctamente como positivos como el modelo.
* **Los verdaderos negativos (VN)** corresponden a la cantidad de negativos que fueron clasificados correctamente como negativos por el modelo.
* **Los falsos negativos (FN)** es la cantidad de positivos que fueron clasificados incorrectamente como negativos.
* **Los falsos positivos (FP)** indican la cantidad de negativos que fueron clasificados incorrectamente como negativos.

Para que lo anterior quede más claro consideremos el siguiente ejemplo.

Un médico tiene cuatro pacientes y a cada uno se les solicitó un examen de sangre y por error el laboratorio realizó también un estudio de embarazo, cuando los pacientes llegan el médico les da los resultado.

A la primera paciente le da la noticia que está embarazada y ella ya lo sabía dado que tiene 3 meses de embarazo, es decir, un verdadero positivo.

El siguiente paciente llega y le dice que no está embarazada y es una clasificación evidente dado que es hombre (Verdadero negativo).

El tercer paciente llega y los resultados le indican que no está embarazada sin embargo tiene cuatro meses de embarazo, es decir, que la ha clasificado como falso negativo.

Y por último el cuarto paciente sus resultados han indicado que está embarazado sin embargo es hombre por lo cual es imposible, dando como resultado un falso positivo.

Lo anterior es un proceso que se realiza por cada instancia a clasificar y nos permite calcular la exactitud y su tasa de error con las siguientes fórmulas.

![Exactitud](/images/exactitud.webp)

![Tasa de error](/images/tasa-error.webp)

Por lo tanto a mayor exactitud nuestro modelo ha aprendido mejor.

# Árboles de decisión

## ¿Qué es un árbol de decisión y cómo se divide?

Forma gráfica y analítica de representar sucesos y sus posibles consecuencias.

**Ventajas**
* Claridad de datos → Podemos ver claramente los caminos tomados
* Tolerantes al ruido y valores faltantes → Aunque no es recomendable podemos hacer nuestro análisis con ruido (con cierto éxito)
* Permite hacer predicciones a través de las reglas extraídas
**Desventajas**
* El criterio de división puede llegar a ser deficiente
* Se puede llegar a un sobreajuste
* Pueden aparecer ramas poco significativas
**Criterios de División de un árbol de decisión:**
* Ganancia de información
* Crear pequeños árboles
**Optimización de nuestro modelo:**
* Evitar el sobreajuste
* Selección de atributos → Seleccionar sólo los atributos relevantes para nuestro modelo
* Campos nulos → Es mejor evitar los campos nulos

[Árboles de Decisión Clasificación – Teoría](https://aprendeia.com/arboles-de-decision-clasificacion-teoria-machine-learning/)

![Árbol de decisión](/images/decision.jpg)

## Comprendiendo nuestro data set para la creación de un árbol de decisión

### Ejemplo de Entrenamiento de un árbol de decisión

Trabajaremos con el dataset Titanic.

**importamos nuestras librerías:**
```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from seaborn import tree

% matplotlib inline
sns.set()
```
**Leemos nuestro dataset (en este caso está dividido en datos de entrenamiento y prueba)**
```py
test_df = pd.read_csv(<nombre_archivo>)
train_df = pd.read_csv(<nombre_archivo>)
```
**Para saber a que tipo de datos pertenece cada columna y la cantidad de datos nulos en cada una usamos el método info:**
```py
train_df.info()
```
**Nota**

Podemos generar visualizaciones rápidas de los datos con el método plot:
```py
train_df.Survived.value_counts().plot(kind='bar',color=('b','r'))
plt.show()
```

## Creando un clasificador con Scikit-Learn

**Tratamiento de Datos**

Para transformar datos cualitativos a un código que entienda la máquina podemos usar un label encoder, se encuentra en el módulo preprocessing:
```py
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
encoder_sex = label_encoder.fit_transform(train_df['Sex'])
```
Obtenemos así un encoder para la columna Sexo (no usaremos este encoder, usaremos el método de dummies)

**Completaremos los valores nulos de Age (con la media de edades) y Embarked (con ‘S’; embarcados en Southampton)**
```py
train_df.Age = train_df.Age.fillna(train_df.Age.median())
train_df.Embarked = train_df.Embarked.fillna('S')
```
**Eliminamos la columnas que no consideramos necesarias**
```py
train_predictors = train_df.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1) #axis=1 se refiere a las columnas (axis=0 → filas)
```
**Separamos las columnas categóricas de las numéricas:**

Para detectar la columnas categóricas analizamos la columna con dtype (debe ser igual a ‘object’), consideramos además que no haya mas de 10 diferentes valores (como factor de seguridad)
```py
categorical_cols = [cname for cname in train_predictors.columns if train_predictors[cname].nunique() <10
                    and train_predictors[cname].dtype=='object']
```
Para detectar la columnas numéricas analizamos la columna con dtype (debe ser igual a 'int64' o 'float64')
```py
numerical_cols = [cname for cname in train_predictors if
                  train_predictors[cname].dtype in ['int64','float64']]
```
Unimos nuevamente las columnas en una sola variable pero con los datos numéricos separados de los categóricos
```py
my_cols = categorical_cols+numerical_cols
train_predictors = train_predictors[my_cols]
```
**Usamos el método get_dummies para codificar las variables numéricas**
```py
dummy_encoded_train_predictors = pd.get_dummies(train_predictors)
```
**Ahora tenemos nuestra data sin valores vacíos y codificada, lista para entrenar nuestro algoritmo**

## Entrenamiento del modelo de clasificación

### Entrenamiento del Modelo

**Generamos nuestras variables de entrenamiento**
```py
y_target = train_df['Survived'].values
x_features_one = dummy_encoded_train_predictors
```
**Podemos dividir los datos en entrenamiento y test con train_test_split pero tenemos un csv con datos de test, usaremos estos datos (previamente tratados al igual que los datos de entrenamiento)**
```py
y_test = test_df['Survived'].values
x_test = dummy_encoded_test_predictors
```
**Entrenamos el modelo:**
```py
tree_one = tree.DecisionTreeClassifier()
tree_one.fit(x_train,y_train)
```
**Obtenemos el grado de precisión de nuestro modelo con el metodo score:**
```py
tree_one_accuracy = tree_one.score(x_test, y_test)
```

## Visualización del árbol de decisión

**Importamos los módulos necesarios**
```py
from io import StringIO #nos permite trabaja con archivos externos
from IPython,display import Image, display #permite interactuar y crear imágenes
import pydotplus #permite usar el lenguaje graphviz para crear imágenes
```
**Exportamos los datos a graphviz y luego los representamos en un archivo png:**
```py
out = StringIO()
tree.export_graphviz(tree_one, out_file=out) # exportamos los datos del árbol en lenguaje graphviz a StringIO
graph = pydotplus.graph_from_dot_data(out.getvalue()) #generamos el gráfico a través de pydotplus
graph.write_png('titanic.png') #guardamos el archivo en formato png
```

![Arbol de decisión](/images/titanic.png)

[Documentación Scikit-Learn: Árboles de decisión](https://scikit-learn.org/stable/modules/tree.html)

[Notebook Collab generado](https://colab.research.google.com/drive/1btCCQlSaDPNkXf9duqMIIgBhM_qnLAqv?usp=sharing)

[Notebook local](/code/5_arbol_decision.ipynb)

# K-Means

## ¿Qué es K-Means?

Crea K grupos a partir de un grupo de observaciones, los elementos deben de tener similitudes.

* Selecciona un valor para K (Centroides)
* Asignamos cada uno de los elementos restante al centro mas cercano.
* Asignamos cada punto punto a su centroide mas cercano
* Repetimos paso 2 y 3 hasta que los centros no se modifiquen.

**Método del codo**

Lo que hace es dividir los siguiente centroides o información hasta graficarlo en un panel o un eje XY

* Calcula el agrupamiento para diferentes de K
* El error al cuadrado para cada punto es el cuadrado de las distancia del punto desde su centro.

[Youtube: StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)

## Cargando el data set de Iris

Este conjunto de datos es tan popular, que es considerado el Hola Mundo de los programadores de ML.

### Agrupando los datos

* Virginica, Versicolor y Setosa
* 50 muestras de cada especie
* Largo y ancho del sépalo y pétalo

**Importamos las librerías**
```py
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
```
**Cargamos el dataset Iris**
```py
iris = datasets.load_iris()
```
**Guardamos los datos del dataset en variables temporales**
```py
x_iris = iris.data
y_iris = iris.target
```
**Creamos los dataframes con las variables X y Y.**
```py
x = pd.DataFrame(iris.data, columns = ['sepal length', 'sepal width', 'petal length', 'petal width'])
y = pd.DataFrame(iris.target, columns = ['Target'])
x.head()

 	sepal length 	sepal width 	petal length 	petal width
0 	         5.1 	        3.5 	        1.4 	        0.2
1 	         4.9 	        3.0 	        1.4 	        0.2
2 	         4.7 	        3.2 	        1.3 	        0.2
3 	         4.6 	        3.1 	        1.5 	        0.2
4 	         5.0 	        3.6 	        1.4 	        0.2
```
**Graficamos un scatter plot, con el fin de explorar los datos del DataFrame**
```py
plt.scatter(x['petal length'], x['petal width'], c = 'blue')
plt.xlabel('Petal Lenght (cm)', fontsize = 10)
plt.ylabel('Petal Width (cm)', fontsize = 10)
```
![Dataset Iris](/images/iris.png)

De acuerdo a la imagen que vemos del análisis exploratorio visual, podríamos decir que hay 2 grandes grupos, fácil, ¿no? Pues no es tan fácil, porque en el grupo grande podrían existir 3 o 4 subgrupos, entonces, ¿Cómo podemos saber cuántos grupos debemos utilizar? **Con el método del Codo**.

## Construcción y evaluación del modelo con K-Means

**Creamos el modelo**
```py
model = KMeans(n_clusters=2, max_iter=1000)
model.fit(x)
y_labels = model.labels_
```
n-clusters es la similitud a K, lo que nos va a permitir generar los centroides en nuestro valor de X, Y. Cada uno de esos grupos está definido por n clusters, lo siguiente es que tenemos que iterar, que es el siguiente parámetro, y esa iteración es cómo vamos a mover K hasta encontrar las distancias más cercanas en cada uno de esos puntos, anteriormente mostrados en el plano X, Y.

**Creamos las predicciones**
```py
y_kmeans = model.predict(x)
print(f'Predicciones {y_kmeans}')

Predicciones [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0]
```
**Probamos el accuracy**
```py
from sklearn import metrics

accuracy = metrics.adjusted_rand_score(y_iris, y_kmeans)
print(accuracy)

0.5399218294207123
```
Con el resultado obtenido podemos decir que es un mal modelo, porque la predicción es prácticamente como arrojar una moneda al aire.

**Código para encontrar el número de ideal centroídes**

> Gracias [danielfzc](https://platzi.com/p/danielfzc/)
```py
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=1000, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
![Centroides](/images/centroides.png)

Podemos observar que el número de centroides ideales es 3. Probamos con el código y verificamos que el accuracy es de 0.7302382722834697.

## Graficación del modelo

**Graficamos con el siguiente código**
```py
plt.scatter(x['petal length'], x['petal width'], c = y_kmeans, s = 30)
plt.xlabel('Petal Length', fontsize = 10)
plt.ylabel('Petal Width', fontsize = 10)
```
![Scatter](/images/scatter-kmeans.png)

¿Qué ocurre con los puntos que se encuentran entre los grupos que parece que no pertenecen al grupo en el que están? K-Means encuentra similitudes, puede que el algoritmo encuentre similitudes con el grupo que no le corresponde, pero eso no significa que esté mal, significa que no vamos a tener un modelo al 100%, o podríamos caer en un sobreajuste. El sobreajuste significa que el algoritmo memoriza la información, más no aprende de ella.

> Buenas tardes a todos, para aquellos que obtuvieron una precisión del 36-37% (al menos para este caso en particular) es debido a que trabajaron con los datos directamente sin estandarizarlos previamente (estoy seguro que el tema de estandarización será tratado en clases posteriores, así que no teman). Haciendo uso de la estandarización pude obtener una precisión de 0.89. A continuación, dejo mi código para que puedan observar la estandarización de datos.
```py
vinos = datasets.load_wine()
variables = vinos.feature_names
x_vinos = vinos.data
y_vinos = vinos.target

# Normalizacion de los valores de “X”
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(vinos.data)
x_scaled = scaler.transform(vinos.data)
x = pd.DataFrame(x_scaled, columns=variables, ) #Si deseas trabajar con los datos sin estandarizar, solo cambia x_scaled por x_vinos
y = pd.DataFrame(y_vinos, columns= ["Target"])
x.head(5)
```

[Archivo Colab generado](https://colab.research.google.com/drive/1bhe4gYgPW6dYVozvLhTHaVqGVqZeXa4T#scrollTo=Kr6FJZxGcDn1)

[Archivo local](/code/6_k_means.ipynb)

# Aprendizaje profundo

## Introducción al aprendizaje profundo

* El aprendizaje profundo no es un tema nuevo, data de los años 50 aproximadamente pero su uso se ha popularizado debido a la existencia de librerias como TensorFlow (Desarrollador por Google) y pyTorch (Desarrollador por Facebook).
* Gracias a las redes neuronales ahora los inputs no necesariamente tienen que ser datos, también pueden ser audios e imágenes.
* Es una subcategoría de ML que crea diferentes niveles, de abstracción que representa los datos y se centra en encontrar similitudes o patrones. Utilizamos [tensores](https://es.wikipedia.org/wiki/C%C3%A1lculo_tensorial) para representar estructuras de datos más complejas.
* Los fundamentos se encuentran en las neuronas. Las redes neuronales artificiales estan basadas en las conexiones neuronales divididas en capas de aprendizaje, estas son: Capa de entrada, capas ocultas y capa de salida.
* Para poder aprender se necesita una función de activación, utilizaremos ReLU, aunque existen otras. ReLu permite el paso de todos los valores positivos sin cambiarlos, pero asigna todos los valores negativos a 0.

## Artículo: Conceptos básicos de Tensor Flow

Tensor Flow es una biblioteca de software de código abierto que permite construir y entrenar redes neuronales, permite detectar y descifrar patrones en los datos. Es un desarrollo de Google y que debido a su flexibilidad y extensa comunidad de programadores ha crecido rápidamente y se ha posicionado como la herramienta líder en el estudio del aprendizaje profundo o también conocido como Deep Learning.

Tensor Flow puede ser usado para ayudar al diagnóstico médico, detectar objetos, procesar imágenes, detección de emociones en el rostro, entre otras aplicaciones. En este curso usamos Tensor Flow para crear nuestra primera red neuronal y diseñar un clasificador de imágenes a partir de un conjunto de datos.

**Importar la biblioteca:**
```py
import tensorflow as tf
```
**Importar el modelo:**
```py
from tensorflow import keras
```
**Cargar conjunto de datos de Tensor Flow:**
```py
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
**Crear modelo secuencial:**
```py
model = keras.Sequential([keras.layers.Flatten(input_shape = (28, 28)), keras.layers.Dense(128, activation = tf.nn.relu), keras.layers.Dense(10, activation = tf.nn.softmax)])
```
**Compilación del modelo:**
```py
model.compile(optimizer = tf.train.AdamOptimizer(), loss = ‘sparse_categorical_crossentropy’, metrics = [‘accuracy’])
```
**Entrenamiento:**
```py
model.fit(train_images, train_labels, epochs = 5)
```
**Evaluación del modelo:**
```py
test_loss, test_acc = model.evaluate( test_images, test_labels )
```
**Predicción del modelo:**
```py
model.predict(test_images)
```
