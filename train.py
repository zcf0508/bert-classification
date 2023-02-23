import time
import os
from typing import Union
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from core.model import modelFactory
from core.tokenizer import getTokenizer, map_example_to_dict, convert_example_to_feature

def _split_dataset(df):
    train_set, x = train_test_split(df, 
        stratify=df['label'],
        test_size=0.1, 
        random_state=42)
    val_set, test_set = train_test_split(x, 
        stratify=x['label'],
        test_size=0.5, 
        random_state=43)

    return train_set, val_set, test_set

def _encode_examples(tokenizer, ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
  
    for _, row in ds.iterrows():
        review = row["text"]
        label = row["y"]
        bert_input = convert_example_to_feature(tokenizer, review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)
    ).map(map_example_to_dict)

def train(data_path: str, labels: list[str], model_path: Union[str, None] = None):
  start_time = time.time()

  model = modelFactory(model_path)
  config = {
    "batch_size": 18,
    "number_of_epochs": 1
  }

  # read data
  df_raw = pd.read_csv(data_path,sep=",",header=None,names=["text","label"])    
  # transfer label
  df_label = pd.DataFrame({
    "label": labels,
    "y": list(range(len(labels))),
  })
  df_raw = pd.merge(df_raw, df_label, on="label", how="left")
  # split data
  train_data, val_data, test_data = _split_dataset(df_raw)

  tokenizer = getTokenizer()
  # train dataset
  ds_train_encoded = _encode_examples(tokenizer, train_data).shuffle(10000).batch(config["batch_size"])
  # val dataset
  ds_val_encoded = _encode_examples(tokenizer, val_data).batch(config["batch_size"])
  # test dataset
  ds_test_encoded = _encode_examples(tokenizer, test_data).batch(config["batch_size"])

  # add a callback to save the model's weights every epoch
  checkpoint_path = f".\\checkpoints\\train_{start_time}\\cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir, 
    save_weights_only=True,
    verbose=1
  )

  bert_history: tf.keras.callbacks.History = model.fit(
    ds_train_encoded, 
    epochs=config["number_of_epochs"], 
    validation_data=ds_val_encoded,
    callbacks=[cp_callback]
  )

  # evaluate test_set
  print("# evaluate test_set:",model.evaluate(ds_test_encoded))

  # save history to log file
  log_file_name = f'./train_logs/train_log_{start_time}.txt'
  # if log file not exist, create it
  if not os.path.exists(log_file_name):
    dir = os.path.dirname(log_file_name)
    if not os.path.exists(dir):
      os.makedirs(dir)
    with open(log_file_name, "w") as f:
      f.write(str(bert_history.history))

  
if __name__ == "__main__":
  labels = ["预约","陈述无法预约"]
  train('./train_data/20230223.txt', labels)
