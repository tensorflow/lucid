"""

This module show how to import a Keras Model into Lucid.
In this case, it is the Mobilenet architecture available with Keras Applications

"""

import tensorflow as tf
from lucid.modelzoo.vision_models import Model as LucidModel

with tf.keras.backend.get_session().as_default():
    tf.keras.backend.set_learning_phase(0)

    model = tf.keras.applications.MobileNet(
            include_top=True,
            weights='imagenet'
    )

    # You can use suggest_save_args() to get suggestions on the metadata
    # you should use for your model.
    # LucidModel.suggest_save_args()

    LucidModel.save(
        "keras_mobilenet.pb",
        image_shape=[224, 224, 3],
        input_name='input',
        output_names=['softmax/Softmax'],
        image_value_range=[-1,1]
        )
