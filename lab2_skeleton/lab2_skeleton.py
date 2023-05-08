import numpy as np
from scipy import sparse
from scipy.stats import mode
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import string
import pandas as pd
from math import log

stop_words = []
with open('./stop_words.txt', 'r') as f:
    for word in f.readlines():
        stop_words.append(word)
stop_words = set(stop_words + list(string.punctuation))

# Task1/2 Q1-2
def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g. 
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    # YOUR CODE HERE
    text= re.sub(r'[.,"\'?:!;]', '', text)
    #print(text)
    tokens=[token for token in text.lower().split(' ') if token not in tokens]
    for item in tokens:
        if item in stop_words:
            tokens.remove(item)

    return tokens

# Task1/2 Q1-3
def get_bagofwords(data, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param vocab_dict: a dict from words to indices, type: dict
    return a word (sparse) matrix, type: scipy.sparse.csr_matrix
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
    '''
    data_matrix = sparse.lil_matrix((len(data), len(vocab_dict)))

    #def vectorize(sentence, vocab):
        #return [sentence.split().count(i) for i in vocab]
    '''
    we didn't use this method here
    '''

    for i in range(len(data)):
        for word in data[i]:
            if word in vocab_dict:
                data_matrix[i,vocab_dict[word]]=1

    # YOUR CODE HERE

    return data_matrix

# Task1/2 Q1-1 Note that, here you need to use Q1-2 and Q1-3.
def read_data(file_name, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    df['words'] = df['text'].apply(tokenize)

    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict(zip(vocab, range(len(vocab))))

    data_matrix = get_bagofwords(df['words'], vocab_dict)

    return df['id'], df['label'], data_matrix, vocab

# Task1 Q2-1
def normalize(P):
    """
    normalize P to make sure the sum of the first dimension equals to 1
    e.g.
    Input: [1,2,1,2,4]
    Output: [0.1,0.2,0.1,0.2,0.4] (without laplace smoothing) or [0.1333,0.2,0.1333,0.2,0.3333] (with laplace smoothing)
    """
    # YOUR CODE HERE

    rowsum = np.sum(P, axis=1, dtype=np.float32)
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diags(r_inv)

    return r_mat_inv.dot(P)


# Task1 Q2-2, Q2-3
def train_NB(data_label, data_matrix):
    '''
    :param data_label: [N], type: list
    :param data_matrix: [N(document_number) * V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    return the P(y) (an K array), P(x|y) (a V*K matrix)
    '''
    N = data_matrix.shape[0]
    V = data_matrix.shape[1]

    K = max(data_label) # labels begin with 1
    # YOUR CODE HERE
    P_y= np.zeros((K))
    P_xy=np.zeros((V,K))
    cnt_tmp=np.zeros((V))

    for i in range(0,K):
        P_y[i]=list(data_label.values).count(i+1)/N

    for i in range(N):
        for j in range(V):
            if data_matrix[i,j]==1:
                cnt_tmp[j]+=1

            cat=data_label.values[i]
            P_xy[j][cat-1]+=1 #从一开始

    for j in range(0,V):
        for i in range(0,K):
             if(cnt_tmp[j]==0):
                 P_xy[j,i]=0
             else:
                 P_xy[j,i]/=cnt_tmp[j]

    return P_y, P_xy
    
# Task1 Q3
def predict_NB(data_matrix, P_y, P_xy):
    '''
    :param data_matrix: [N(document_number), V(Vocabulary number)], type:  scipy.sparse.csr_matrix
    :param P_y: [K(class number)], type: np.ndarray
    :param P_xy: [V, K], type: np.ndarray
    return data_pre (a N array)
    '''
    # compute the label probabilities using the P(y) and P(x|y) according to the naive Bayes algorithm
    # YOUR CODE HERE
    N=data_matrix.shape[0]
    K=P_y.shape[0]
    res=np.zeros((N,K))
    prior=np.log(P_y)
    V=P_xy.shape[0]
    for i in range(0,N):
        res[i]+=prior
        for j in range(V):
            if data_matrix[i,j]==1:
                res[i]+=np.log(P_xy[j,:])

    prob=np.exp(res)
    data_pre=np.argmax(prob,axis=1)
    data_pre+=1

    # get labels for every document by choosing the maximum probability
    # YOUR CODE HERE
    return data_pre


#Task2 Q2-1
def tfidf(data_matrix):
    '''
    :param data_matrix: [N(document_number) * V(Vocabulary number)], type: numpy.array
    return the tfidf_matrix (a N*K matrix)
    '''
    def termfreq(matrix, doc_id, term):
        try:
            max_num = np.max(matrix[doc_id])
            return 0.5 + 0.5 * (matrix[doc_id,term]) / float(max_num)
        except ZeroDivisionError:
            return 0

    def inversedocfreq(matrix, term):
        try:
            unique_docs = sum([1 for i in range(matrix.shape[0]) if matrix[i, term] > 0])
            return log(matrix.shape[0] / (1 + unique_docs))+1
        except ZeroDivisionError:
            return 0

    tfidf_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[1]), dtype=float)
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            tfidf_matrix[i][j]=termfreq(data_matrix,i,j)*inversedocfreq(data_matrix,j)
    #YOUR CODE HERE
    return tfidf_matrix

#Task2 Q2-2
def euclidean_distance(x1: np.ndarray, x2: np.ndarray):
    '''
    :param x1, x2 (tfidf vectors)
    return the euclidean distance between x1 and x2
    '''
    dist=np.sqrt(np.sum((x1-x2)**2))
    #YOUR CODE HERE
    return dist

#Task2 Q2-3
def predict_KNN(train_data_matrix, train_data_label, predict_data_matrix, K):
    '''
       :param train_data_matrix: [N(document_number) * V(Vocabulary number)], type: numpy.array
       :param train_data_label: [N(document_number) * C(class_number)], type: numpy.array
       :param predict_data_matrix: [N(document_number) * V(Vocabulary number)], type: numpy.array
       :param K: number of nerbors for calculation type: int
       return the predicted labels
       '''
    train_data_label=train_data_label.values
    train_data_matrix = tfidf(train_data_matrix.todense())
    predict_data_matrix = tfidf(predict_data_matrix.todense())
    out_labels = []
    # Loop through the Datapoints to be classified
    for item in range(predict_data_matrix.shape[0]):
        # Array to store distances
        point_dist = []
        # Loop through each training Data
        for j in range(len(train_data_matrix)):
            dist=euclidean_distance(predict_data_matrix[item],train_data_matrix[j])
            point_dist.append(dist)
            # Calculating the distance
            # YOUR CODE HERE
            # pass
        point_dist=np.array(point_dist)
        rank=np.argsort(point_dist)
        vote_rank=rank[:K]
        ext_labels=train_data_label[vote_rank]

        prob=dict()
        for label in ext_labels:
            if label not in prob.keys():
                prob[label]=1
            else:
                prob[label]+=1

        truth=max(prob,key=prob.get)
        out_labels.append(truth)

        # Sort the bottom K distances and do the voting
        # YOUR CODE HERE
    return np.array(out_labels)

def evaluate(y_true, y_pre):
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    # average='macro', calculate the average precision, recall, and F1 score of all categories.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    return acc, precision, recall, f1

def evaluate_each_category(y_true, y_pre): 
    assert len(y_true) == len(y_pre)
    acc = accuracy_score(y_true, y_pre)
    # average='macro', calculate the precision, recall, and F1 score of each category.
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pre, average=None)
    return acc, precision, recall, f1

if __name__ == '__main__':
    print("Task 1: Naive Bayes exercise:")
    # Read train.csv
    train_id_list_NB, train_data_label_NB, train_data_matrix_NB, vocab_NB = read_data("data/train_NB.csv")
    # Divide train.csv into train set(80%) and validation set (20%).

    X_train_NB, X_validation_NB, y_train_NB, y_validation_NB = train_test_split(train_data_matrix_NB,
                                                                                train_data_label_NB, test_size=0.2)
    print("Vocabulary Size:", len(vocab_NB))
    print("Training Set Size:", len(y_train_NB))
    print("Validation Set Size:", len(y_validation_NB))

    '''

    # Read test.csv
    test_id_list_NB, _, test_data_matrix_NB, _ = read_data("data/test_NB.csv", vocab_NB)
    print("Test Set Size:", len(test_id_list_NB))



    # training
    P_y, P_xy = train_NB(y_train_NB, X_train_NB)
    train_data_pre_NB = predict_NB(X_train_NB, P_y, P_xy)
    acc, precision, recall, f1 = evaluate(y_train_NB, train_data_pre_NB)
    print(
        "Evalution in train set: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (acc, precision, recall, f1))

    # validation
    validation_data_pre_NB = predict_NB(X_validation_NB, P_y, P_xy)
    acc, precision, recall, f1 = evaluate(y_validation_NB, validation_data_pre_NB)
    print("Evalution in Validation set: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (
    acc, precision, recall, f1))

    acc, precision, recall, f1 = evaluate_each_category(y_validation_NB, validation_data_pre_NB)
    print("Evalution on test set in each category: \nAccuracy:", acc, "\nPrecision:", precision, "\nRecall:", recall,
          '\nF1:', f1)

    # Predict the label of each documents in the test set
    test_data_pre_NB = predict_NB(test_data_matrix_NB, P_y, P_xy)

    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list_NB
    sub_df["pred"] = test_data_pre_NB
    sub_df.to_csv("submission_NB.csv", index=False)
    print("Predict Results are saved, please check the submission_NB.csv")

   '''

    print()
    print()
    print("Task 2； KNN exercise:")
    train_id_list_KNN, train_data_label_KNN, train_data_matrix_KNN, vocab_KNN = read_data("data/train_KNN.csv")

    X_train_KNN, X_validation_KNN, y_train_KNN, y_validation_KNN = train_test_split(train_data_matrix_KNN,
                                                                                    train_data_label_KNN, test_size=0.2)

    print("Vocabulary Size:", len(vocab_KNN))
    print("Training Set Size:", len(y_train_KNN))
    print("Validation Set Size:", len(y_validation_KNN))
    # Read test.csv
    test_id_list_KNN, _, test_data_matrix_KNN, _ = read_data("data/test_KNN.csv", vocab_KNN)
    print("Test Set Size:", len(test_id_list_KNN))



    # KNN with K = 3
    validation_data_pre_KNN = predict_KNN(X_train_KNN, y_train_KNN, X_validation_KNN, K=3)
    acc, precision, recall, f1 = evaluate(y_validation_KNN, validation_data_pre_KNN)
    print("Evalution with k = 3 in Validation set: Accuracy: %f\tPrecision: %f\tRecall: %f\tMacro-F1: %f" % (
        acc, precision, recall, f1))
    acc, precision, recall, f1 = evaluate_each_category(y_validation_KNN, validation_data_pre_KNN)
    print("Evalution with k = 3 in each category: \nAccuracy:", acc, "\nPrecision:", precision, "\nRecall:", recall,
          '\nF1:', f1)

    # Predict the label of each documents in the test set
    test_data_pre_KNN = predict_KNN(X_train_KNN, y_train_KNN, test_data_matrix_KNN, K=3)

    sub_df = pd.DataFrame()
    sub_df["id"] = test_id_list_KNN
    sub_df["pred"] = test_data_pre_KNN
    sub_df.to_csv("submission_KNN.csv", index=False)
    print("Predict Results are saved, please check the submission_KNN.csv")
