import tensorflow as tf


class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv2d_1 = tf.keras.layers.Conv2D(filters=6,
        #                                        kernel_size=5,
        #                                        activation='sigmoid',
        #                                        input_shape=(28, 28, 1))
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=6,
                                               kernel_size=5,
                                               activation='sigmoid')
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=16,
                                               kernel_size=5,
                                               activation='sigmoid')
        self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.flat = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(120, activation='sigmoid')
        self.fc_2 = tf.keras.layers.Dense(84, activation='sigmoid')
        self.fc_3 = tf.keras.layers.Dense(10, activation='sigmoid')

    def call(self, input):
        x = self.conv2d_1(input)
        x = self.maxpool_1(x)
        x = self.conv2d_2(x)
        x = self.maxpool_2(x)
        x = self.flat(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x

    def model(self):
        x = tf.keras.Input(shape=(28, 28, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

def prepare_dataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Reshape data to (n, 28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1],
                              x_train.shape[2],
                              1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0],
                            x_test.shape[1],
                            x_test.shape[2],
                            1).astype('float32')
    # Reshape label to (n, 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print("\n### Dataset Shape ###")
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("### Using GPU ###")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    x_train, x_test, y_train, y_test = prepare_dataset()

    model = LeNet()
    print('\n### Shape of each layer ###')
    X = tf.random.uniform((1, 28, 28, 1))
    for layer in model.layers:
        X = layer(X)
        print(layer.name, 'output shape\t', X.shape)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.9,
                                                    momentum=0.0,
                                                    nesterov=False),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print('\n### Fit model on training data ###')
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)
    model.summary()
    print('\nhistory dict:', history.history)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    print('\n# Generate predictions for 3 samples')
    predictions = model.predict(x_test[:3])
    print('predictions:', predictions.argmax(axis=1))
    print('Answer:', y_test[:3])

