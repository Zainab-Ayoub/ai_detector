import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_model(vocab_size=30000, embedding_dim=128, max_length=500):
    """
    LSTM-based classifier for AI vs Human text detection.
    
    Parameters:
    - vocab_size: size of tokenizer vocabulary
    - embedding_dim: size of word embedding vectors
    - max_length: maximum sequence length (padding)
    """

    model = models.Sequential([
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ),

        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')  # AI = 1, Human = 0
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
