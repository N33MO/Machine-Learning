import os
import copy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np


def TrainMCAPLogisticRegression(C, D, η, λ, it):
    
    # params for filter out stop words
    no_remove = ""
    punctuations = list(string.punctuation)
    stop_words = stopwords.words('english')
    stop_words_punc = stopwords.words('english') + list(string.punctuation)
    
    myFilter = no_remove
    
    # threshold for gradient ascent
    threshold = it
    
    
    
    # create a dictionary to read all documents' full path
    files = []
    for r, d, f in os.walk(D):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
                
    
    
    # create a dictionary to read all distinct words from training set
    idx = 0
    vocabulary = {}
    for path in files:
        file = open(path, 'r', encoding='utf-8', errors='ignore')
        text = ""
        for line in file:
            text = text + line.strip().lower() + " "
        file.close()
        tokens = word_tokenize(text)
        filtered_keys = [i for i in word_tokenize(text) if i not in myFilter]
        for k in filtered_keys:
            if k not in vocabulary:
                vocabulary[k] = idx
                idx += 1
    
    
    
    # now there are:
    # len(vocabulary) == idx distinct words  (size of array X)
    # len(files) documents                   (number of array X)
    # idx+1 weights                          (size of array w)
    # so we generate matrix X, class y, and vector w
    
    w = np.ones(idx+1)
    X = np.zeros(shape=(len(files), idx))
    y = np.zeros(len(files))                           # initialize as 0 (ham)
    
    # read all files and update X and y
    idx = 0
    for path in files:
        file = open(path, 'r', encoding='utf-8', errors='ignore')
        text = ""
        for line in file:
            text = text + line.strip().lower() + " "
        file.close()
        tokens = word_tokenize(text)
        filtered_keys = [i for i in word_tokenize(text) if i not in myFilter]
        # update X
        for k in filtered_keys:
            X[idx][vocabulary[k]] += 1
        # update y only if spam
        if '.spam' in path:                            # y = 1 for spam email
            y[idx] = 1
        idx += 1
    X = np.hstack((np.ones(len(files)).reshape(len(files), 1), X))    # appen a ones column as index = 0
    
    
    # now we get w, X, and y
    # implement a function for calculate P from w and X[i]
    # set η and λ
#     η = 0.007
#     λ = 0.005
    
    # when n > 36, exp(36) / (1 + exp(36)) = 1.0
    
    w_prev = w
    trend = copy.deepcopy(w)
    for i in range(threshold):
        # ease the final function
        exp = np.dot(X, w)
        exp = np.clip(exp,-36,36)
        numerator = np.exp( exp )                                   # for y predict
        denominator = 1 + numerator                                 #
        y_pred = np.true_divide(numerator, denominator)             # y predict
        y_diff = y - y_pred                                         # y diff
        func = np.transpose(np.transpose(X) * y_diff)               # sum function
        func = func.sum(axis=0)
        
#         w[0] = w[0] - η * λ * w[0]
        w = w + η * func - η * λ * w                                # final function
#         if sum(abs(w_prev[1:] - w[1:])) < 1e-6:
#             break
#         w_prev = w
        trend = np.vstack((trend,copy.deepcopy(w)))
        
    
    return vocabulary, w


def ApplyMCAPLogisticRegression(C, V, w, d):
    
    # params for filter out stop words
    no_remove = ""
    punctuations = list(string.punctuation)
    stop_words = stopwords.words('english')
    stop_words_punc = stopwords.words('english') + list(string.punctuation)
    
    myFilter = no_remove
    
    
    x = np.zeros(len(w)-1)
    y = 1 if '.spam' in d else 0
    
    
    file = open(d, 'r', encoding='utf-8', errors='ignore')
    text = ""
    for line in file:
        text = text + line.strip().lower() + " "
    file.close()
    tokens = word_tokenize(text)
    filtered_keys = [i for i in word_tokenize(text) if i not in myFilter]
    for k in filtered_keys:
        if k in V:
            x[V[k]] += 1
    
    exp = w[0] + np.dot(w[1:], x)
    exp = np.clip(exp,-36,36)
    numerator = np.exp(exp)
    denominator = 1 + numerator
    y_pred = numerator / denominator
    
    return y_pred


def EvaluateMCAPLogisticRegression(C, D, D_test, η, λ, it):
    
    voc, w = TrainMCAPLogisticRegression(C, D, η, λ, it)
    
    # create a dictionary to read all documents' full path
    files = []
    for r, d, f in os.walk(D):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    
    
    result = {c: {'positive': 0, 'negative': 0, 'accuracy': 0} for c in C}
    
    for file in files:
        y_pred = ApplyMCAPLogisticRegression(C, voc, w, file)
        y = 1 if '.spam' in file else 0
        if y == 1:
            if y_pred > 0.5:
                result['spam']['positive'] += 1
            else:
                result['spam']['negative'] += 1
        else:
            if y_pred < 0.5:
                result['ham']['positive'] += 1
            else:
                result['ham']['negative'] += 1
    
    pos = 0
    neg = 0
    for c in C:
        result[c]['accuracy'] = result[c]['positive'] / (result[c]['positive'] + result[c]['negative'])
        pos += result[c]['positive']
        neg += result[c]['negative']
    
    overall = pos / (pos + neg)
    
    
    return result, overall



def printAll(result, overall, η, λ, it):
    print("----------------------------------------------------------")
    print("                 MCAP Logistic Regression                 ")
    print("----------------------------------------------------------")
    print("  η = " + str(η) + ", λ = " + str(λ) + ", iter = " + str(it) + ", filter = 'none'")
    print("Result:")

    print("spam:\t\t" + "{:.4%}".format(result['spam']['accuracy']) + "\t" + "( pos: " + str(result['spam']['positive']) + "\tneg: " + str(result['spam']['negative']) + " )")

    print("ham:\t\t" + "{:.4%}".format(result['ham']['accuracy']) + "\t" + "( pos: " + str(result['ham']['positive']) + "\tneg: " + str(result['ham']['negative']) + " )")

    print("overall:\t" + "{:.4%}".format(overall))

    print("----------------------------------------------------------")


if __name__ == '__main__':
    C = ["spam", "ham"]
    D = "./train/"
    D_test = "./test/"

    # voc, w = TrainMCAPLogisticRegression(C, D)

    η = 0.1
    λ = 0.01
    it = 25


    # result, overall = EvaluateMCAPLogisticRegression(C, D, D_test, λ)

    for λ in [0.001, 0.003, 0.005, 0.007, 0.009, 0.5]:
        result, overall = EvaluateMCAPLogisticRegression(C, D, D_test, η, λ, it)
        printAll(result, overall, η, λ, it)
