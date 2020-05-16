import sys
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import pdb

def tokenize_sentences(sentences):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # encode_plus will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the [CLS] token to the start.
        #   (3) Append the [SEP] token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to max_length
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 320,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids,attention_masks
