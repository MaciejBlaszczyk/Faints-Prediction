import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def custom_loss(y_pred, y_true):
    diff = y_pred - y_true
    return tf.keras.backend.square(diff)  # Breakpoint in IDE here. =====

class SimpleModel(Model):

    def __init__(self):
        super().__init__()
        self.dense0 = Dense(2)
        self.dense1 = Dense(1)

    def call(self, inputs):
        z = self.dense0(inputs)
        z = self.dense1(z)
        return z

x = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
y = tf.convert_to_tensor([0, 1], dtype=tf.float32)

model0 = SimpleModel()
model0.run_eagerly = True
model0.compile(optimizer=Adam(), loss=custom_loss)
y0 = model0.fit(x, y, epochs=1)  # Values of diff *not* shown at breakpoint. =====

model1 = SimpleModel()
model1.compile(optimizer=Adam(), loss=custom_loss)
model1.run_eagerly = True
y1 = model1.fit(x, y, epochs=1)  # Values of diff shown at breakpoint. =====