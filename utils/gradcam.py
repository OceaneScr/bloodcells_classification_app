
# Third-party libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Local import
from utils.config import (BACKBONE_MAP, TARGET_LAYER_MAP)




def grad_cam(image, model, model_name):

    target_layer_name = TARGET_LAYER_MAP[model_name]
    backbone_name = BACKBONE_MAP[model_name]
    
    image = np.array(image).copy()
    image = tf.expand_dims(image, axis=0)

    backbone = model.get_layer(backbone_name)
    backbone_idx = next(
        i for i, layer in enumerate(model.layers) if layer.name == backbone_name
    )

    # Sous-modèle avant la couche target_layer_name pour Grad-CAM
    last_before_backbone_layer = model.layers[backbone_idx - 1]
    pre_backbone_model = tf.keras.Model(
        inputs=model.inputs, outputs=last_before_backbone_layer.output
    )

    # Sous-modèle au niveau de la couche target_layer_name pour Grad-CAM
    target_layer_output = backbone.get_layer(target_layer_name).output
    backbone_model = tf.keras.Model(
        inputs=backbone.inputs, outputs=[target_layer_output, backbone.output]
    )

    # Sous-modèle tête après la couche target_layer_name pour Grad-CAM
    head_input = tf.keras.Input(shape=backbone.output_shape[1:])
    hx = head_input
    for layer in model.layers[backbone_idx + 1 :]:
        hx = layer(hx)
    head_model = tf.keras.Model(inputs=head_input, outputs=hx)

    preprocessed = pre_backbone_model(image, training=False)

    model.layers[-1].activation = None

    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = backbone_model(
            preprocessed,
            training=False
        )

        if isinstance(backbone_output, list):
            backbone_output = backbone_output[0]

        tape.watch(conv_outputs)

        predictions = head_model(backbone_output, training=False)

        class_idx = tf.argmax(predictions[0])
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(
        conv_outputs[0] * pooled_grads,
        axis=-1
    )


    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)
    heatmap /= (tf.reduce_max(heatmap)+1e-8)


    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (image.shape[1], image.shape[2])
    ).numpy()
    heatmap_resized = np.squeeze(
        heatmap_resized, axis=-1
    )  # supprimer la dimension de taille 1 à la fin du tableau heatmap_resized

    img_normalized = image[0].numpy() / 255.0

    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap_resized)[..., :3]
    # N'applique la heatmap que proportionnellement à son intensité

    # threshold = np.percentile(
    #     heatmap_resized,
    #     80
    # )

    # alpha_map = np.where(
    #     heatmap_resized[...,None] > threshold,
    #     0.6,
    #     0
    # )
    alpha_map = heatmap_resized[..., np.newaxis] * 0.7  # alpha variable selon intensité
    superimposed_image = heatmap_colored * alpha_map + img_normalized * (1 - alpha_map)

    model.layers[-1].activation = tf.keras.activations.softmax

    gradcam_image = (superimposed_image * 255).astype(np.uint8)
    return gradcam_image, class_idx




