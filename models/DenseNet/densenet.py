
try:
    # %tensorflow_version only exists in Colab.
#     %tensorflow_version 2.x
except Exception:
    pass

# !pip install tensorflow==2.0.0
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
print(tf.__version__)

# Parameters
ACCURACY_THRESHOLD = 0.99
NUM_CLASSES = 10
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
CHANNELS = 3

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=4 * growth_rate,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=growth_rate,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        return x

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.features_list = []

    def _make_layer(self, x, training):
        y = BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate)(x, training=training)
        self.features_list.append(y)
        y = tf.concat(self.features_list, axis=-1)
        return y

    def call(self, inputs, training=None, **kwargs):
        self.features_list.append(inputs)
        x = self._make_layer(inputs, training=training)
        for i in range(1, self.num_layers):
            x = self._make_layer(x, training=training)
        self.features_list.clear()
        return x

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filters=out_channels,
                                           kernel_size=(1, 1),
                                           strides=1,
                                           padding="same")
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                              strides=2,
                                              padding="same")

    def call(self, inputs, training=None, **kwargs):
        x = self.bn(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(tf.keras.Model):
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super(DenseNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=num_init_features,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding="same")
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.dense_block_1(x, training=training)
        x = self.transition_1(x, training=training)
        x = self.dense_block_2(x, training=training)
        x = self.transition_2(x, training=training)
        x = self.dense_block_3(x, training=training)
        x = self.transition_3(x, training=training)
        x = self.dense_block_4(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)

        return x
    
    def build_graph(self, input_shape): 
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        
        _ = self.call(inputs)

def densenet_121():
    return DenseNet(num_init_features=64, growth_rate=32,
                    block_layers=[6, 12, 24, 16], compression_rate=0.5,
                    drop_rate=0.5)


def densenet_169():
    return DenseNet(num_init_features=64, growth_rate=32,
                    block_layers=[6, 12, 32, 32], compression_rate=0.5,
                    drop_rate=0.5)


def densenet_201():
    return DenseNet(num_init_features=64, growth_rate=32,
                    block_layers=[6, 12, 48, 32], compression_rate=0.5,
                    drop_rate=0.5)


def densenet_264():
    return DenseNet(num_init_features=64, growth_rate=32,
                    block_layers=[6, 12, 64, 48], compression_rate=0.5,
                    drop_rate=0.5)

def prepare_dataset():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)

    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    print("### Dataset Shape ###")
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    return x_train, x_test, y_train, y_test

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy'):
            if (logs.get('accuracy') > 0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True

def print_model_summary(network):
    network.build_graph((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    network.summary()


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("### Using GPU ###")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    x_train, x_test, y_train, y_test = prepare_dataset()

    model = densenet_121()
    print_model_summary(model)

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                                    momentum=0.0,
                                                    nesterov=False),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    print('\n### Fit model on training data ###')
    callbacks = myCallback()
    history = model.fit(x_train, y_train, epochs=1500, validation_split=0.1, batch_size=1024, callbacks=[callbacks])
    print('\nhistory dict:', history.history)

    model.evaluate(x_test, y_test, verbose=2)

    print('\nhistory dict:', history.history)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print('\n# Generate predictions for 3 samples')
    predictions = model.predict(x_test[:3])
    print('predictions:', predictions.argmax(axis=1))
    print('Answer:', y_test[:3])
