import pandas as pd
import numpy as np
import seaborn as sns
import requests
import matplotlib.pyplot as plt
import sklearn as sk
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import LabelBinarizer,normalize
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

def get_data():
    if not os.path.exists("iris.csv"):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
        r = requests.get(url, allow_redirects=True)
        filename = "iris.csv"
        open(filename, 'wb').write(r.content)
    else:
        pass

def forward_prop(x,dict1,dict2):
    # forward propogation takes the input x and propogates to the hidden units at each layer
    # to produce the final output yhat.
    # Hidden layer1
    layer_1 = tf.add(tf.matmul(x, dict1['hid1']), dict2['bias1'])
    layer_1 = tf.nn.relu(layer_1) #rectified linear unit
    print(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, dict1['hid2']), dict2['bias2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output fully connected layer
    print(layer_2)
    out_layer = tf.matmul(layer_2, dict1['out']) + dict2['out']
    print(out_layer)
    return out_layer


def main():
    tf.disable_v2_behavior()
    get_data()
    df = pd.read_csv("iris.csv", header=None,
 names=['sepal_length','sepal_width','petal_length','petal_width','species'])

    # visualize the data
    #sns.pairplot(df,hue="species",diag_kind="kde")
    #plt.show()

    # we want to classify the 4 iris species using the 4 features. We can do this with TF.

    # need to one-hot encode the species data for "binarization" so it is a vector
    # that our ML algorithm will understand. After the fit transform, y will be our one hot encode vector.
    species_label = LabelBinarizer()
    y = species_label.fit_transform(df.species.values)


    # We want to normalize our 4 input features, to improve our gradient descent
    features = df.columns[0:4]
    x_data = df[features].values
    x_data = normalize(x_data)


    # Now we need to split the dataset into a training and test set, to prevent overfitting
    x_train,x_test,y_train,y_test = train_test_split(x_data,y,test_size=0.3,random_state=1)
    # test size means 80:20 ratio of dataset will be split into the train:test.
    print(x_train.shape)
    # Now we can create our TF Neural Network
    learning_rate = 0.01
    epochs = 100

    # our first layer of neurons will have 256 and our second will have 128
    n_hid_1 = 256
    n_hid_2 = 128
    n_input = x_train.shape[1] #input shape of [120,4]
    n_classes = y_train.shape[1] # classes to predict

    # now we can define our tensors. We can do this by definining dicts, and also including
    # a bias dictionary, that will be added to the output layer.
    # Dictionary of Weights and Biases
    weights = {
      'hid1': tf.Variable(tf.random_normal([n_input, n_hid_1])),
      'hid2': tf.Variable(tf.random_normal([n_hid_1, n_hid_2])),
      'out': tf.Variable(tf.random_normal([n_hid_2, n_classes]))
    }
    biases = {
      'bias1': tf.Variable(tf.random_normal([n_hid_1])),
      'bias2': tf.Variable(tf.random_normal([n_hid_2])),
      'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Inputs
    xx = tf.placeholder("float", shape=[None, n_input])
    Y = tf.placeholder("float", shape=[None, n_classes])

    # Model Outputs
    yhat = forward_prop(xx,weights,biases)
    ypredict = tf.argmax(yhat, axis=1) # returns largest index with the value across axes of a tensor.
    # Will hold 1 for predicted class, 0 for others

    # now we can implement backward propogation
    # TF has a function that applies the softmax and then calculates cross entropy between this and actuals
    # We want to reduce cost, so we can use gradient descent optimizer and then the minimize attribute

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=yhat))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train_op = optimizer.minimize(cost)

    # we can run a tf session to train our neural net.
    # We will run over # of epochs and apply forward and backward propogation steps for each sample


    # Initializing the variables
    init = tf.global_variables_initializer()
    startTime = datetime.now()
    with tf.Session() as session:
        session.run(init)

        #writer.add_graph(session.graph)
        #EPOCHS
        for epoch in range(epochs):
            #Stochasting Gradient Descent
            for i in range(len(x_train)):
                summary = session.run(train_op, feed_dict={xx: x_train[i: i + 1], Y: y_train[i: i + 1]})

            train_accuracy = np.mean(np.argmax(y_train, axis=1) == session.run(ypredict, feed_dict={xx: x_train, Y: y_train}))
            test_accuracy  = np.mean(np.argmax(y_test, axis=1) == session.run(ypredict, feed_dict={xx: x_test, Y: y_test}))

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
            #print("Epoch = %d, train accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy))
        session.close()
    print("Time taken:", datetime.now() - startTime)


if __name__ == "__main__":
    main()
