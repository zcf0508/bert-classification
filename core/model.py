from typing import Union
from transformers import (
  TFBertForSequenceClassification,
  TFPreTrainedModel,
)
import tensorflow as tf

# https://huggingface.co/bert-base-chinese
default_model_path = 'bert-base-chinese'

def modelFactory(path: Union[str, None] = None) -> Union[tf.keras.Model, TFPreTrainedModel]:
  if path:
    return tf.keras.models.load_model(path)
  else:
    config = {
      "learning_rate": 2e-5,
      "num_classes": 2
    }

    model: TFPreTrainedModel = TFBertForSequenceClassification.from_pretrained(
      default_model_path, 
      num_labels=config["num_classes"]
    )

    # optimizer Adam recommended
    optimizer = tf.keras.optimizers.Adam(
      learning_rate=config["learning_rate"],
      epsilon=1e-08, 
      clipnorm=1
    )

    # we do not have one-hot vectors, 
    # we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model
