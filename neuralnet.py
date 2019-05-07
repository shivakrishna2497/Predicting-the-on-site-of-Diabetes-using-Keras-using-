from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.model_selection import train_test_split
numpy.random.seed(7)
# loaded pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
print(dataset)
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
print("input variables (X)",X)
print("output class variable (Y)",Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)
# evaluating the model with Trainingset
scores = model.evaluate(X_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
# evaluating the model with Testingset
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
