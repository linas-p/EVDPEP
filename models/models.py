from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization, TimeDistributed
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score
from focal_loss import BinaryFocalLoss
from keras import metrics
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def dnn(input_dim, num_out, is_summed, n_timesteps):
    model = keras.Sequential()
    if is_summed:
        model.add(Dense(16, input_shape=(n_timesteps, input_dim), activation='relu', kernel_initializer='he_normal'))
    else:
        model.add(Dense(16, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'))

    model.add(BatchNormalization())
    model.add(Dense(64,  activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dense(64,  activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    if is_summed:
        model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(BatchNormalization())
    model.add(Dense(8, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))    
    model.add(Dense(num_out, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    return model


def lstm(n_timesteps, input_dim, is_summed, num_out):
    model = keras.Sequential()
    model.add(tf.keras.layers.LSTM(40, input_shape=(n_timesteps, input_dim), return_sequences=True, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.LSTM(40, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    if is_summed:
        model.add(tf.keras.layers.LSTM(40, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    else:
        model.add(tf.keras.layers.LSTM(40, return_sequences=True, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dense(num_out, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    return model



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu )(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

    
def create_sig_classifier(n_timesteps, patch_size, batch_size, is_summed, num_out):
    inputs = layers.Input(shape=(n_timesteps, 30))
    representation = tf.reshape(inputs, [batch_size, n_timesteps *  30])
    
    projection_dim = 4
    num_heads = 4
    transformer_units = [
    projection_dim * 2,
    projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 2
    mlp_head_units = [64, 64]  # Size of the dense layers of the final classifier
    representation = layers.Dense(160)(representation)
    encoded_patches = tf.reshape(representation, [batch_size, 40, 4])
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.1)(representation)
    #representation = tf.reshape(representation, [batch_size, 64])
    
    
    # Add MLP.
    #features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)
    # Classify outputs.
    if is_summed:
        features = layers.Dense(40, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(representation)
        logits = layers.Dense(num_out, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(features)
    else:
        features = layers.Dense(160, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(representation)
        features = layers.Dense(n_timesteps * num_out, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(features)
        logits = tf.reshape(features, [batch_size, n_timesteps, num_out])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model    