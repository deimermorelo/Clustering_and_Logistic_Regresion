# Solucion al Taller Clasificando Correos
# Diplomado Python Aplicado a la Ingenieria UPB
# Autor: Deimer David Morelo Ospino
# ID: 502217
# Email: deimer.morelo@upb.edu.co

# Librerias a utilizar para tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Librerias a utilizar para graficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Librerias a utilizar para el preprocesado y el modelado de los datos
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Libreria a utilizar para la configuracion de warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Leemos la url con pandas y creamos el dataframe
# ==============================================================================
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/' \
       + 'Estadistica-machine-learning-python/master/data/spam.csv'
datos = pd.read_csv(url)

# Mostramos por consola toda la informacion asociada a los 3 primeros
# elementos del Dataframe 'datos'   
print(datos.head(3))

# Definimos la variable de respuesta como 1 si es spam y 0 si no lo es 
datos['type'] = np.where(datos['type'] == 'spam', 1, 0)

# Identificamos y mostramos la cantidad de observaciones que hay de cada clase
print("\nNúmero de observaciones por clase")
print(datos['type'].value_counts())
print("")

# Definimos y mostramos el porcentaje de observaciones por clase
print("Porcentaje de observaciones por clase")
print(100 * datos['type'].value_counts(normalize=True))

# Hacemos la división de los datos en train y test
# ==============================================================================
X = datos.drop(columns = 'type')
y = datos['type']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para 
# el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.Logit(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

# Predicciones con intervalo de confianza 
# ==============================================================================
predicciones = modelo.predict(exog = X_train)

# Clasificación predicha
# ==============================================================================
clasificacion = np.where(predicciones<0.5, 0, 1)
clasificacion

# Accuracy de test del modelo 
# ==============================================================================
X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo.predict(exog = X_test)
clasificacion = np.where(predicciones<0.5, 0, 1)
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = clasificacion,
            normalize = True
           )
# Mostramos por consola el porcentaje de accurracy
print("")
print(f"El accuracy de test es: {100*accuracy} %")

# Creamos la matriz de confusión de las predicciones de test
# ==============================================================================
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    clasificacion,
    rownames=['Real'],
    colnames=['Predicción']
)
confusion_matrix

