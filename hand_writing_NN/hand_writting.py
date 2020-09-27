import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def get_model(model_name, x_train, y_train):
    path = os.path.abspath(os.getcwd()) + '\\' + model_name
    if os.path.isdir(path):            # if model directory exists
        ''' load the model '''
        print("Model exists!")
        model = tf.keras.models.load_model(model_name)
        return model
    else: # model does not exist yet, train and save one
        ''' Define a model (architecture in a few lines) '''
        print("Model DOES NOT exists!")
        model = tf.keras.models.Sequential()
        # add first layer - inputs  (flatten the matrix!)
        model.add(tf.keras.layers.Flatten())
        # just two hidden layers - problem is not complex
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))        # 128 neurons in a layer, reli - rectified linear (activation function)
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))        # 128 neurons in a layer, reli - rectified linear (activation function)
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))      # output has x neurons which is equal to classification neuron (in this problem - X = 10 (10 digits)), softmax for probability distribution

        ''' Define parameters for the training of the model '''
        # usually we always try to minimize loss (error), not maximize accuracy, so how youd define 'loss' can have a huge impact on your model
        # Epoch is just a "full pass" through your entire training dataset. So if you just train on 1 epoch, then the neural network saw each
        # unique sample once. 3 epochs means it passed over your data set 3 times.
        model.compile(optimizer='adam',                                      # 'adam' is kind of default...
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3)

        ''' save the model '''
        model.save(model_name)
        return model



def main():
    print('Hello')
    ''' inbuilt keras digits dataset'''
    mnist = tf.keras.datasets.mnist  # 28 x 28 images of hand written digits

    ''' Train and test data '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize (values in matrices between 0 and 1)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # get model (train or just load if folder already exists)
    model_name = 'numbers_reader.model'
    model = get_model(model_name, x_train, y_train)

    # probability distributions...
    predictions = model.predict([x_test])
    # choose maximum probability value for x_test[0]
    print(np.argmax(predictions[0]))
    # draw it...
    plt.imshow(x_test[0], cmap = plt.cm.binary)
    plt.show()

    ''' calculate value loss and accuracy '''
    # you should expect higher loss and lower accuracy than within training data
    # but... if delta is too much, you probably deal with overfitting
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("Val loss: " + str(val_loss) + "\nVal accuracy: " + str(val_acc))

if __name__=="__main__":
    main()

