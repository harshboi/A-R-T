import nltk
import spacy
from spacy.lang.en import English

import numpy as np
import pickle 

from sklearn.feature_extraction.text import TfidfVectorizer
def get_top_tf_idf_words(response, top_n):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]]
    # feature_names = np.array(vectorizer.get_feature_names())

# Note: corpus_data must be a list of dictionary with with key for every index being "text" and value being the sentence
def create_tf_idf_model ( corpus_data ):
    vectorizer = TfidfVectorizer()
    corpus = [ corpus_data[i]['text'] for i in range( len(corpus_data) ) ]
    X = vectorizer.fit_transform(corpus)
    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    return vectorizer

# Supply location of pickle file
def load_tf_idf_model ( pickled_model ):
    vectorizer = pickle.load(open(pickled_model, "rb"))
    return vectorizer

def remove_noise_from_tweet (tweet):
    return( tweet.replace("#", "").replace("|"," ").replace("@"," ") )

def applyNLTK(tweet):
    nlp = spacy.load('en_core_web_sm')
    sen_noun = []
    # sentence = tweet['tweet']
    tokens = nltk.word_tokenize(tweet)
    tagged = nltk.pos_tag(tokens)
    for j in range(len(tagged)):
        if (tagged[j][1] == 'NNP'):
            sen_noun.append(tagged[j][0])
    # tweet['nltk'] = sen_noun
    return sen_noun


# Option = 1 => Get all Nouns, Oprion = 2 => Get position entities
def applySpacy( tweet, option ):
    nlp = spacy.load('en_core_web_sm')
    sentence = tweet.replace("-", " ").replace("#", " ")
    doc = nlp(sentence)
    if option == 1:
        tagged = [chunk.text for chunk in doc.noun_chunks]
        # tweet['spacy'] = tagged
        return tagged
    else:
        pos_ent = []
        for ent in doc.ents:
            pos_ent.append(ent.text)
        # tweet['spacy'] = pos_ent
        return pos_ent


#Alternative Spacy --Currently Not in use
########################################################################################################
# Uses Spacy for finding nouns, named entity recognition
# NEW results from spacy, after removing stop words
########################################################################################################

def get_nouns_wo_stop_words (data, option):
    nlp = English()

    my_doc = nlp(data['text'].replace("-", " ").replace("#", " "))
    # Create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)
    from spacy.lang.en.stop_words import STOP_WORDS

    # Create list of word tokens after removing stopwords
    filtered_sentence =[]

    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)    # Contains the sentence without stop words
    # print(token_list)
    # print(filtered_sentence)

    new_text = filtered_sentence[0] + " "
    for i in range(1,len(filtered_sentence)):
        new_text += filtered_sentence[i] + " "

    nlp = spacy.load('en_core_web_sm')

    #  "nlp" Object is used to create documents with linguistic annotations.
    noun = []
    sen_noun = []
    sentence = new_text
    doc = nlp(sentence)
    if option == 1:
        tagged = [chunk.text for chunk in doc.noun_chunks]
        return tagged
    else:
        pos_ent = []
        for ent in doc.ents:
            pos_ent.append(ent.text)
#             print(ent.text, ent.label_)
        return pos_ent
