import nltk
import spacy
from spacy.lang.en import English



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
