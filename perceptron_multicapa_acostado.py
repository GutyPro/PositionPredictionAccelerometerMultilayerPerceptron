import tensorflow as tf
import csv
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------- CSV READER ----------------------------------------------------
def import_data(filename):
	rows = []
	labels = []

	#Leo el csv
	with open(filename, 'r') as csvfile:
	    #Creo el csvreader
		csvreader = csv.reader(csvfile)
	    #Salto el header
		next(csvreader, None)
		#Salto la linea vacia
		next(csvreader, None)

		#Extraigo la informacion de cada fila
		for row in csvreader:
			#Me quedo con el campo label(Target)
			label = row[6]
			#Agrego el label al conjunto
			labels.append(label)

			#Me quedo con todos los campos seleccionados en el Analisis de los Datos
			# ['var15', 'ind_var5', 'ind_var8_0', 'ind_var30', 'num_var5', 'num_var30', 'num_var42', 'var36', 'num_meses_var5_ult3']
			row = row[1:2] + row[2:3] + row[3:4]
			#Agrego la fila al conjunto
			rows.append(row)

			#Salto la vacia
			next(csvreader, None)

	return rows, labels

#------------------------------- Definicion Perceptron Multicapa RRNN ------------------------------------------------

# Parámetros usados para entrenar la red
learning_rate = 0.03 #tasa de aprendizaje
num_steps = 3000 #cantidad de pasos de entrenamiento
batch_size = 1000 #cantidad de ejemplos por paso

# Parámetros para la construcción de la red
n_hidden = 4 # número de neuronas en la capa oculta
num_classes = 2 # 2 clases: 0 - Satisfecho y  1 - Insatisfecho

# Definimos la red neuronal
def neural_net (x_dict):
	# x_dic es un diccionario con los valores de entrada
	# x serán los valores de entrada de los campos
	x = x_dict['data'] #en particular vendrán en el campo "data"
	# Conectamos x (la entrada) con la capa oculta: Conexión full
	layer_1 = tf.layers.dense(x, n_hidden)
	# Conectamos la capa oculta con la capa de salida
	out_layer = tf.layers.dense(layer_1, num_classes)
	return out_layer

# Usamos la clase “TF Estimator Template”, para definir cómo será el entrenamiento
def model_fn (features, labels, mode):
	# Llamamos a la función anterior para construir la red
	logits = neural_net(features)

	# Predicciones
	pred_classes = tf.argmax(logits, axis=1)
	pred_probas = tf.nn.softmax(logits)

	# Si es de predicción devolvemos directamente un EstimatorSpec
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions = pred_classes)

	# Definimos nuestro error
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
	# sparse_softmax_cross_entropy_with_logits : Mide el error de probabilidad
	# en tareas de clasificación discretas en las que las clases son mutuamente
	# excluyentes (cada entrada está en exactamente una clase)
	# reduce_mean : Calcula la media de los elementos a través de las dimensiones de un tensor

	#Definimos un optmizador, que trabaja por el método de descenso por gradiente
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

	# Definimos cómo se evaluará la precisión del modelo
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

	# Finalmente devolvemos un objeto: “EstimatorSpec”, indicando todo lo que
	# calculamos para el entrenamiento: modo, predicción, error (loss), método de entrenamiento y métricas
	estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})

	return estim_specs


#Comienza el programa
print("Comienza el programa...")
#------------------- Obtengo los datos de entrenamiento y evaluacion ---------------------------------------------------------------
print("Obteniendo datos de entrenamiento del archivo process-train.csv...")
filename = "process-train.csv"
rowsTrain, labelsTrain = import_data(filename)
print("Datos de entrenamiento OK")

print("Partiendo el conjunto de entrenamiento en entrenamiento y evaluacion...")
#Transformo rows y labels en np array
train_data_np = np.asarray(rowsTrain, np.float32)
train_labels_np = np.asarray(labelsTrain, np.int32)

#Finalmente partimos el conjunto en 0.7 para entrenamiento y 0.3 para testing.
trainX, testX, trainY, testY = train_test_split(train_data_np, train_labels_np, test_size=0.30)

print("Datos de entrenamiento y evaluacion OK")

#----------------------- Entrenamiento de la Red Neuronal --------------------------------------------------------------
print("Entrenamiento de la red neuronal...")
# Construimos un estimador, le decimos que use la función antes definida
model = tf.estimator.Estimator(model_fn)

# Pasamos ahora todos los parámetros que necesita la función definida
input_fn = tf.estimator.inputs.numpy_input_fn(x= {'data': trainX} , y = trainY, batch_size=batch_size, num_epochs=None, shuffle=True)

#Entrenamos el modelo
model.train(input_fn, steps=num_steps)
print("Entrenamiento OK")

#----------------------- Evaluación del modelo de la Red Neuronal --------------------------------------------------------------
print("Evaluación de la red neuronal...")
# Evaluamos el modelo
# Definimos la entrada para evaluar
input_fn = tf.estimator.inputs.numpy_input_fn(x= {'data': testX}, y = testY, batch_size=batch_size, shuffle=False)

# Usamos el método 'evaluate'del modelo
eTrain = model.evaluate(input_fn)
print("Evaluación del modelo OK")

#------------------- Obtengo los datos de prueba ---------------------------------------------------------------
print("Obteniendo datos de prueba del archivo process-test.csv...")
filename = "process-test.csv"
rowsTest, labelsTest = import_data(filename)

#Transformo rows y labels en np array
test_data_np = np.asarray(rowsTest, np.float32)
test_labels_np = np.asarray(labelsTest, np.int32)

print("Datos de prueba OK")

#----------------------- Prueba de la Red Neuronal --------------------------------------------------------------
print("Prueba de la red neuronal...")
# Probamos el modelo
# Definimos la entrada para evaluar
input_fn = tf.estimator.inputs.numpy_input_fn(x= {'data': test_data_np}, y = test_labels_np, batch_size=batch_size, shuffle=True)

# Usamos el método 'evaluate'del modelo
eTest = model.evaluate(input_fn)
print("Prueba del modelo OK")

#----------------------- Resultados --------------------------------------------------------------
print("--------- Resultados Obtenidos ---------- (los valores se encuentran en %)")
print("Precisión en la evaluacion del modelo, persona acostada: ", (eTrain['accuracy'] * 100))
print("Precisión en la prueba del modelo, persona acostada: ", (eTest['accuracy'] * 100))
print("Programa Finalizado")
