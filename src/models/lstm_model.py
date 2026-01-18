import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_model(vocab_size, embedding_dim, max_length, num_classes):
    """
    LSTM-based classifier for AI vs Human text detection.
    
    Parameters:
    - vocab_size: size of tokenizer vocabulary
    - embedding_dim: size of word embedding vectors
    - max_length: maximum sequence length (padding)
    - num_classes: number of output classes
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

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model