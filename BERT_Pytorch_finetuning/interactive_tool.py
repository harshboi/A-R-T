####################################################
# Author: Rohan Varma
# This script takes in a pytorch model state (.pt or .pth file)
# and provides an interactive console to see which sentences
# are deemed relevant or irrelevant
###################################################


import sys
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import pdb

def print_model_info(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))



def tokenize_sentences(sentences):
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



# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

if len(sys.argv) != 2:
    print("usage: python interactive_tool.py <pytorch_model_path>")
else:

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    print_model_info(model)
    #load specified model state
    model.load_state_dict(torch.load(sys.argv[1],map_location=device))
    model.eval()
    while(True):
        string = input("Enter string to classify: ")
        #tokenize inputted sentence to be compatible with BERT inputs
        token_ids,attention_masks = tokenize_sentences([string])
        #get a tensor containing probabilities of inputted sentence being irrelevant or relevant
        model_outputs = (model(token_ids, token_type_ids=None, attention_mask=attention_masks))
        softmax_layer = torch.nn.Softmax()
        result = softmax_layer(model_outputs[0])
        #identify which output node has higher probability and what that probability is
        prediction = torch.argmax(result).item()
        confidence = torch.max(result).item()
        if prediction:
            print("Relevant")
        else:
            print("Irrelevant")
        print("{:.2f}% confident".format(confidence*100))

