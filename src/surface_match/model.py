import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import BatchNormalization, Dropout, concatenate, \
    Dense, GlobalAveragePooling2D
from tensorflow.python.layers.base import Layer

from surface_match.config import IMG_SHAPE, SAVED_MODEL, SAVED_MODEL_W
from surface_match.dataset import loss_in_fact


def get_model() -> Model:
    res_net: Model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)

    layer: Layer
    for layer in res_net.layers:
        layer.trainable = False

    image_a = Input(IMG_SHAPE)
    image_b = Input(IMG_SHAPE)

    branch_a = res_net(image_a)
    branch_b = res_net(image_b)

    merged_layers = concatenate([branch_a, branch_b])
    merged_layers = GlobalAveragePooling2D()(merged_layers)

    merged_layers = Dense(1024, activation='selu')(merged_layers)
    merged_layers = Dropout(0.5)(merged_layers)
    merged_layers = BatchNormalization()(merged_layers)

    merged_layers = Dense(256, activation='selu')(merged_layers)
    merged_layers = Dropout(0.5)(merged_layers)
    merged_layers = BatchNormalization()(merged_layers)

    output = Dense(1, kernel_initializer='normal', activation='selu')(merged_layers)
    model = Model(inputs=[image_a, image_b], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='mae',
                  metrics=[loss_in_fact])

    return model


def save_models(model_for_save):
    model_for_save.save(SAVED_MODEL)
    model_for_save.save_weights(SAVED_MODEL_W)
