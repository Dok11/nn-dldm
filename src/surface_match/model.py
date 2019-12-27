import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, concatenate, \
    Flatten, Dense

from surface_match.config import IMG_SHAPE, SAVED_MODEL, SAVED_MODEL_W
from surface_match.dataset import loss_in_fact


def get_image_branch() -> Model:
    shared_input = Input(IMG_SHAPE)

    # 64x64 > 21x21
    shared_layer = Conv2D(64, (4, 4), strides=3, input_shape=IMG_SHAPE, padding='valid')(shared_input)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('selu')(shared_layer)

    # 21x21 > 10x10
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.1)(shared_layer)

    # 10x10 > 8x8
    shared_layer = Conv2D(128, (3, 3), padding='valid')(shared_layer)
    shared_layer = BatchNormalization()(shared_layer)
    shared_layer = Activation('selu')(shared_layer)

    # 8x8 > 4x4
    shared_layer = MaxPooling2D(pool_size=(2, 2))(shared_layer)
    shared_layer = Dropout(0.1)(shared_layer)

    return Model(shared_input, shared_layer, name='shared_model')


def get_model() -> Model:
    shared_model = get_image_branch()
    shared_model.summary()

    image_a = Input(IMG_SHAPE)
    image_b = Input(IMG_SHAPE)

    branch_a = shared_model(image_a)
    branch_b = shared_model(image_b)

    merged_layers = concatenate([branch_a, branch_b])
    merged_layers = Flatten()(merged_layers)

    merged_layers = Dense(512, activation='selu')(merged_layers)
    merged_layers = Dropout(0.5)(merged_layers)
    merged_layers = BatchNormalization()(merged_layers)

    merged_layers = Dense(256, activation='selu')(merged_layers)
    merged_layers = Dropout(0.5)(merged_layers)
    merged_layers = BatchNormalization()(merged_layers)

    output = Dense(1, kernel_initializer='normal', activation='selu')(merged_layers)
    model = Model(inputs=[image_a, image_b], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0025),
                  loss='mae',
                  metrics=[loss_in_fact])

    return model


def save_models(model_for_save):
    model_for_save.save(SAVED_MODEL)
    model_for_save.save_weights(SAVED_MODEL_W)
