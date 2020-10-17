import os
import copy
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def TrainMultinomialNB(C, D):
    
    no_remove = ""
    punctuations = list(string.punctuation)
    stop_words = stopwords.words('english')
    stop_words_punc = stopwords.words('english') + list(string.punctuation)
    myFilter = stop_words
    
    
    # create a dictionary to read all documents' full path and make its class as keys
    files = {c: [] for c in C}
    # r: root, d: directories, f: files
    for r, d, f in os.walk(D):
        for file in f:
            for c in C:
                if '.'+c in file:
                    files[c].append(os.path.join(r, file))
    
    
    
    # create a dictionary to read all words from each class
    vocabulary = {}
    for c in C:                                                                           # for each class
        for path in files[c]:                                                             # for each document in the class
            file = open(path, 'r', encoding='utf-8', errors='ignore')                     # ignore some decoding issues (especialy in emails)
            text = file.read().lower()                                                    # read into a string: 'text'
            file.close()
            tokens = word_tokenize(text)
            filtered_keys = [i for i in word_tokenize(text) if i not in myFilter]         #

            for k in filtered_keys:                                                       # apply to dictionary
                if k in vocabulary:
                    if c in vocabulary[k]:
                        vocabulary[k][c] += 1
                    else:
                        vocabulary[k][c] = 1
                else:
                    vocabulary[k] = {c: 1}
    # regular vocabulary dict by adding 0 value
    for k in vocabulary:
        for c in C:
            if c not in vocabulary[k]:
                vocabulary[k][c] = 0
    
    
    
    # prior probability of each class
    prior = {}
    totalFiles = 0;
    for c in C:                                                                           # calculate total number of documents
        totalFiles += len(files[c])
    for c in C:
        prior[c] = len(files[c]) / totalFiles
    
    
    
    # calculate prabability of each word/term
    condprob = copy.deepcopy(vocabulary)
    denominator = {}
    for c in C:                                                                           # calculate total number of words/terms
        denominator[c] = 0
        for k in vocabulary:
            denominator[c] += vocabulary[k][c] + 1                                        # apply laplace smoothing by add 1 to each count
    for c in C:
        for k in vocabulary:
            condprob[k][c] = (vocabulary[k][c] + 1) / denominator[c]
            
        
    return vocabulary, prior, condprob


def ApplyMultinomialNB(C, V, prior, condprob, d):
    
    no_remove = ""
    punctuations = list(string.punctuation)
    stop_words = stopwords.words('english')
    stop_words_punc = stopwords.words('english') + list(string.punctuation)
    myFilter = stop_words

    score = {c: math.log(prior[c]) for c in C}
    
    file = open(d, 'r', encoding='utf-8', errors='ignore')
    text = file.read().lower()
    tokens = word_tokenize(text)
    filtered_keys = [i for i in word_tokenize(text) if i not in myFilter]
    
    for c in C:
        for k in filtered_keys:
            if k in V:
                score[c] += math.log(condprob[k][c])
    
    return score


def evaluateMultinomialNB(C, D, D_test):
    
    voc, prior, condprob = TrainMultinomialNB(C, D)
    
    
    # create a dictionary to read all documents' full path and make its class as keys
    files = {c: [] for c in C}
    # r: root, d: directories, f: files
    for r, d, f in os.walk(D_test):
        for file in f:
            for c in C:
                if '.'+c in file:
                    files[c].append(os.path.join(r, file))
    
    result = {c: {'positive': 0, 'negative': 0, 'accuracy': 0} for c in C}
    for c in C:
        for f in files[c]:
            score = ApplyMultinomialNB(C, voc, prior, condprob, f)
            if score[c] == max(score.values()):
                result[c]['positive'] += 1
            else:
                result[c]['negative'] += 1

    pos = 0
    neg = 0
    for c in C:
        result[c]['accuracy'] = result[c]['positive'] / (result[c]['positive'] + result[c]['negative'])
        pos += result[c]['positive']
        neg += result[c]['negative']
    
    overall = pos / (pos + neg)
    
    return result, overall


if __name__ == '__main__':
    C = ["spam", "ham"]
    D = "./train/"
    D_test = "./test/"

    # voc, prior, condprob = TrainMultinomialNB(C, D)

    result, overall = evaluateMultinomialNB(C, D, D_test)

    print("----------------------------------------------------------")
    print("Multinomial Naive Bayes")
    print("----------------------------------------------------------")
    print("Words filter:\t\tstop_words\n")
    print("Result: ")
    print("  spam: ")
    print("\tpositive:\t" + str(result['spam']['positive']) + "\n\tnegative:\t" + str(result['spam']['negative']))
    print("\taccuracy: " + "{:.4%}".format(result['spam']['accuracy']))
    print("  ham: ")
    print("\tpositive:\t" + str(result['ham']['positive']) + "\n\tnegative:\t" + str(result['ham']['negative']))
    print("\taccuracy: " + "{:.4%}".format(result['ham']['accuracy']))
    print("  overall: ")
    print("\taccuracy: " + "{:.4%}".format(overall))
    print("----------------------------------------------------------")
