from transformers import (
  BertTokenizer, 
  PreTrainedTokenizerBase,
)
import tensorflow as tf

default_model_path = './bert-base-chinese'

def getTokenizer() -> PreTrainedTokenizerBase:
  tokenizer: PreTrainedTokenizerBase = BertTokenizer.from_pretrained(default_model_path)
  return tokenizer

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def convert_example_to_feature(tokenizer: PreTrainedTokenizerBase, review):
  config = {
    "max_length": 32
  }
  # combine step for tokenization, WordPiece vector mapping, adding special tokens 
  # as well as truncating reviews longer than the max length
  return tokenizer.encode_plus(
    review, 
    add_special_tokens = True, # add [CLS], [SEP]
    max_length = config["max_length"], # max length of the text that can go to BERT
    pad_to_max_length = True, # add [PAD] tokens
    return_attention_mask = True, # add attention mask to not focus on pad tokens
    truncation=True
  )

def convert_question_to_feature(tokenizer: PreTrainedTokenizerBase, question: str):
  bert_input = convert_example_to_feature(tokenizer, question)
  return tf.data.Dataset.from_tensor_slices(
        (
          [bert_input.input_ids], 
          [bert_input.attention_mask], 
          [bert_input.token_type_ids], 
          [None]
        )
    ).map(map_example_to_dict).batch(1)
