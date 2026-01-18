import tensorflow as tf
from tensorflow.keras import layers, models


def build_neural_net(input_dim):
    """
    A simple but effective feed-forward neural network.
    Works best with engineered features + n-grams.

    input_dim = length of your final feature vector
    """

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),

        layers.Dense(1, activation='sigmoid')  # AI (1) vs Human (0)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
