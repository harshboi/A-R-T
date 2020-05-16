####################################################
# Author: Rohan Varma
# This script takes in a pytorch model state (.pt or .pth file)
# as well as two numpy arrays containing text and labels, and
# provides performance metrics for the models performance on the given dataset 
###################################################


import sys
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
import pdb
from sklearn import metrics
import matplotlib.pyplot as plt
import os

batch_size=16
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#prints out accuracy and auc
def print_auc(y_true,y_probas,show_plot=True):
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_probas,pos_label = 1)
    auc_score = metrics.auc(fpr,tpr)
    print("AUC: {}".format(auc_score))
    if show_plot:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()
    return auc_score

def tokenize_sentences(sentences,labels):
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
    labels = torch.tensor(labels,dtype=torch.long)
    return input_ids,attention_masks,labels



def create_dataloader(test_sentences,test_labels,batch_size):
    test_ids,test_masks,test_label_tensors = tokenize_sentences(test_sentences,test_labels)
    test_dataset = TensorDataset(test_ids,test_masks,test_label_tensors)


    print('{:>5,} training samples'.format(len(test_dataset)))


    # Create the DataLoaders for our testing sets.
    test_dataloader = DataLoader(
        test_dataset,  # The training samples.
        batch_size = batch_size # Trains with this batch size.
    )


    return test_dataloader


def write_pred_to_file(sentence,prediction,label,file):
    f = open(file,'a')
    f.write('Tweet: {}\n'.format(sentence.encode('utf8')))
    if prediction == 1:
        f.write('Prediction: Relevant\n')
    else:
        f.write('Prediction: Irrelevant\n')
    if label == 1:
        f.write('Label: Relevant\n\n')
    else:
        f.write('Label: Irrelevant\n\n')
    f.close()


#write predictions to a file 
def record_predictions(text,predictions,labels,output_file,incorrect_preds_file):
    for i,sentence in enumerate(text):
        write_pred_to_file(sentence,predictions[i],labels[i],output_file)
        if predictions[i] != labels[i]:
            write_pred_to_file(sentence,predictions[i],labels[i],incorrect_preds_file)
            


# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

if len(sys.argv) != 6:
    print("usage: python testmodel.py <pytorch_model_path> <text_np_array> <label_np_array> <predictions_output_file_path> <incorrect_predictions_output_file_path>")
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
    #load specified model state
    model.load_state_dict(torch.load(sys.argv[1],map_location=device))
    model.eval()
    model.to(device)
    test_text = np.load(sys.argv[2])
    test_labels = np.load(sys.argv[3])
    output_file = sys.argv[4]
    incorrect_predictions_file = sys.argv[5]

    open(output_file,'w').close()
    open(incorrect_predictions_file,'w').close()

    test_dataloader = create_dataloader(test_text,test_labels,batch_size)
    print("finished creating dataloader")
    total_correct = 0
    batch_num = 0
    softmax_layer = torch.nn.Softmax()
    probas = np.zeros((test_text.shape[0],2))
    for batch in test_dataloader:
        #get a tensor containing probabilities of inputted sentence being irrelevant or relevant
        result = model(batch[0].to(device), token_type_ids=None, attention_mask=batch[1].to(device))[0]
        result = result.detach().cpu()
        result = softmax_layer(result)
        predictions = torch.argmax(result,1).numpy()
        labels = batch[2].numpy()
        #pdb.set_trace()
        result = result.numpy()
        if(result.shape[0] == batch_size):
            probas[batch_num*batch_size:batch_num*batch_size+batch_size] = result
            record_predictions(test_text[batch_num*batch_size:batch_num*batch_size+batch_size],predictions,labels,output_file,incorrect_predictions_file)
            print("tested {} samples".format((batch_num+1)*16))
        else:
            probas[batch_num*batch_size:] = result
            print("tested {} samples".format(batch_num*16+result.shape[0]))
            record_predictions(test_text[batch_num*batch_size:],predictions,labels,output_file,incorrect_predictions_file)
        total_correct += np.sum(predictions == labels)

        print("Correct Predictions:{}".format(total_correct))
        batch_num += 1

    print(float(total_correct)/test_text.shape[0])
    print_auc(test_labels,probas[:,1])
