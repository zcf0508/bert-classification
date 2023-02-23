import tensorflow as tf
from core.tokenizer import getTokenizer, convert_question_to_feature

def predict(question: str , labels: list[str], model: tf.keras.Model):
  tokenizer = getTokenizer()
  result = model.predict(convert_question_to_feature(tokenizer, question))

  result_idx = tf.argmax(tf.keras.layers.Softmax()(result[0]).numpy()[0]).numpy()

  return result_idx, labels[result_idx]
