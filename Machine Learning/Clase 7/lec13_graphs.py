#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
File: lec12_graphs.py
Author: Ignacio Soto Zamorano / Ignacio Loayza Campos
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com / ignacio1505[at]gmail[dot]com
Github: https://github.com/ignaciosotoz / https://github.com/tattoedeer
Description: Ancilliary files for Tensors and Perceptron lecture - ADL
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
# importamos de manera explícita el optimizador de Gradiente Estocástica
from keras.optimizers import SGD
#importamos de forma explícita la estructura básica
from keras.models import Sequential
# importamos de forma explícita la definición de capas densas (fully connected) 
from keras.layers import Dense

seed = hash("Desafio LATAM es lolein")%2^32
fetch_lims = lambda x: [np.min(x), np.max(x)]

def get_joint_xy(x, y):
    """TODO: Docstring for get_joint_xy.

    :x: TODO
    :y: TODO
    :returns: TODO

    """
    xlim = fetch_lims(x)
    ylim = fetch_lims(y)

    x_mesh, y_mesh = np.meshgrid(
        np.linspace(xlim[0], xlim[1]),
        np.linspace(ylim[0], ylim[1])
    )

    joint_xy = np.c_[x_mesh.ravel(), y_mesh.ravel()]

    return x_mesh, y_mesh, joint_xy

markers = ['o', '^']


def circles(n = 2000, stddev = 0.05):
    generator = check_random_state(seed)

    linspace = np.linspace(0, 2 * np.pi, n // 2 + 1)[:-1]
    outer_circ_x = np.cos(linspace)
    outer_circ_y = np.sin(linspace)
    inner_circ_x = outer_circ_x * .3
    inner_circ_y = outer_circ_y * .3

    X = np.vstack((np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(n // 2, dtype=np.intp), np.ones(n // 2, dtype = np.intp)])
    X += generator.normal(scale = stddev, size = X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = seed)
    return X_train,y_train,X_test,y_test

def plot_classifier(clf, X_train, Y_train, X_test, Y_test, model_type):
    # Generamos los parámetros de nuestro canvas
    f, axis = plt.subplots(1, 1, sharex = "col", sharey = "row", figsize = (12,8))
    # Representamos los datos de entrenamiento
    axis.scatter(X_train[:,0], X_train[:,1], s = 30, c = Y_train, zorder = 10, cmap = "autumn")
    # Representamos los datos de validación
    axis.scatter(X_test[:,0], X_test[:,1], s = 20, c = Y_test, zorder = 10, cmap ="winter")
    # generamos una grilla multidimensional
    XX, YY = np.mgrid[-2:2:200j, -2:2:200j]
    # Si el modelo es una variante de árbol
    if model_type == "tree":
        # La densidad probabilística se obtendrá de la siguiente forma
        Z = clf.predict_proba(np.c_[XX.ravel(), YY.ravel()])[:,0]
    # si el modelo es una variante de una red neuronal artificial
    elif model_type == "ann":
        # Obtendremos las clases predichas 
        Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    else:
        # de lo contrario generaremos una excepción.
        raise ValueError("model type not supported")
    Z = Z.reshape(XX.shape)
    Zplot = Z >= 0.5
    axis.pcolormesh(XX, YY, Zplot, cmap = "Purples")
    axis.contour(XX, YY, Z, alpha = 1, colors = ["k", "k", "k"], linestyles = ["--", "-", "--"], levels = [-2, 0, 2])
    plt.show()


def one_layer_network(X_train, y_train, neurons = 1, input_init = "uniform", input_activation = "relu", hidden_init = "uniform", hidden_activation = "sigmoid", loss = "binary_crossentropy", verbosity = 0):

    """
    X_train: matriz de atributos de entrenamiento
    y_train: vector objetivo de entrenamiento
    input_init: inicializador de las capas de entrada
    input_activation: forma de activación (por defecto es Rectified Linear Unit)
    hidden_init: inicializador de las capas escondidas
    hidden_activation: forma de activación (por defecto es Sigmoide)
    loss: forma de obtención de la medida de pérdida, por defecto es binary_crossentropy

    """
    # Definimos una serie de capas lineales como arquitectura
    model = Sequential()
    # Añadimos una capa densa (neuronas completamente conectadas) con la cantidad de atributos en nuestra matriz de entrenamiento
    model.add(Dense(neurons, input_dim = X_train.shape[1], kernel_initializer = input_init, activation = input_activation))
    # Añadimos una capa densa con 1 neurona para representar el output
    model.add(Dense(1, kernel_initializer = hidden_init, activation = hidden_activation))
    # compilamos los elementos necesarios, implementando gradiente estocástica y midiendo exactitud de las predicciones como norma de minimización
    model.compile(optimizer = SGD(lr = 1), loss = loss, metrics = ["accuracy"])
    # entrenamos el modelo
    model.fit(X_train, y_train, epochs = 50, batch_size = 100, verbose = verbosity)
    return model

def evaluate_network(net, X_train, y_train, X_test, y_test, show_results = True):
    scores = net.evaluate(X_test, y_test)
    test_acc = scores[1]
    if show_results:
        print("\r"+ " "*60 + "\rAccuracy: %f" % test_acc)
        plot_classifier(net, X_train, y_train, X_test, y_test, "ann")
    return test_acc

def ann_number_of_layers(model, X_train,y_train,X_test = None, y_test=None, layers = 1, n_neurons=12):
    """TODO: Docstring for ann_number_of_layers.

    :model: TODO
    :X_train: TODO
    :X_test: TODO
    :y_train: TODO
    :y_test: TODO
    :layers: TODO
    :n_neurons: TODO
    :returns: TODO

    """
    model = Sequential()
    for i in range(layers):
        if i == 0:
            indim = X_train.shape[1]
        else:
            indim = None

        model.add(Dense(n_neurons, input_dim = indim,
                        kernel_initializer = "glorot_normal",
                        activation = "relu", name = "hidden_{}".format(i + 1)))
    model.add(Dense(1, kernel_initializer = "glorot_normal",
                    activation = 'sigmoid', name = "out"))

    model.compile(optimizer = SGD(lr = 1),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    model.fit(X_train, y_train, epochs = 50, batch_size=100, verbose = 0)

    return model

def ann_learning_rate(X_train, y_train, X_test = None, y_test = None, learning_rate = 1, n_neurons = 12):
    """TODO: Docstring for ann_learning_rate.

    :X_train: TODO
    :y_tr: TODO
    :returns: TODO

    """
    model = Sequential()

    model.add(Dense(n_neurons, input_dim = X_train.shape[1],
                    kernel_initializer = 'glorot_normal',
                    activation = 'relu', name = 'hidden_1'))

    model.add(Dense(1, kernel_initializer = 'glorot_normal',
                    activation = 'sigmoid', name = 'out'))

    model.compile(optimizer=SGD(lr=learning_rate),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=0)
    
    return model

def ann_neurons_number(X_train, y_train, X_test = None, y_test = None, n_neurons=12):
    
    """
    ann_neurons_number: train a neural net with n defined neurons

    """
    # set sequential canvas
    model = Sequential()
    # add a fully connected layer
    model.add(
        # with user defined neurons
        Dense(n_neurons,
                    # input layer size
                    input_dim = X_train.shape[1],
                    # weights are randomly initialized following glorot normal
                    kernel_initializer = 'glorot_normal',
                    # weighted sum is relu activated
                    activation = 'relu', name = 'hidden_1'))
    # add a fully connected layer
    model.add(
        # with 1 neuron and glorot normal initialization
        Dense(1, kernel_initializer='glorot_normal',
                    # weighted sum is sigmoid activated
                    activation = 'sigmoid', name = 'output'))
    # arquitecture is compiled 
    model.compile(
        # following a SGD optimizer
        optimizer=SGD(lr=1),
        # defined loss function is binary crossentropy
        loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 50, batch_size =100, verbose=0)

    return model


def plot_response_surface(model, X, y, x_mesh, y_mesh, joint_xy):
    """TODO: Docstring for plot_response_surface.

    :model: TODO
    :X: TODO
    :y: TODO
    :returns: TODO

    """
    predict_classifier = model.predict(joint_xy).reshape(x_mesh.shape)
    plt.contourf(x_mesh, y_mesh, predict_classifier, cmap='coolwarm')

    for i in np.unique(y):
        plt.scatter(X[y == i][:, 0], X[y == i][:, 1],
                    marker = markers[i], color='indigo', alpha=.5,
                    label='Clase: {}'.format(i))
        plt.xlim(fetch_lims(X))
        plt.ylim(fetch_lims(y))
        plt.legend()


def weight_bias_behavior():
    """TODO: Docstring for weight_bias_behavior.
    :returns: TODO

    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    x=np.linspace(-20,20,100)
    y=np.linspace(-20,20,100)

    _, ax = plt.subplots(1,2, figsize = (10, 6))
    ## Diferentes pesos
    for i in [0.1, 0.3, 0.7, 1, 3]:
        sigmoid_estimate = sigmoid(i * y)
        sns.lineplot(y, sigmoid_estimate, label=r'$\sigma$' + "({} * x)".format(i),
                    ax = ax[0])
    sns.despine()
    ax[0].legend()
    ax[0].set_title('Output frente a diferentes pesos de input', size = 15);

    for i in [-5, -2, 0, 2, 2]:
        sigmoid_estimate = sigmoid(1 * y + i)
        sns.lineplot(y, sigmoid_estimate, label=r'$\sigma$' + "(1 * x + {})".format(i), ax=ax[1])
    sns.despine()
    ax[1].legend()
    ax[1].set_title('Output frente a diferentes sesgos', size = 15);


def softmax_sigmoid_behavior():
    """TODO: Docstring for softmax_sigmoid_behavior.
    :returns: TODO

    """
    softmax = lambda x: np.exp(x) / float(sum(np.exp(x)))
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    x_axis = np.linspace(-10, 10, 2)
    y_axis = np.linspace(-10, 10, 2)

    plt.subplot(1, 2, 1)
    plt.plot(y_axis, sigmoid(y_axis), 'dodgerblue')
    plt.title('Función Sigmoidal')

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, softmax(x_axis), 'tomato')
    plt.title('Función Softmax')
    sns.despine()
