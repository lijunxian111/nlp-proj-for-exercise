# author: ‘李颖欣’
# student_id: “202064271338”
import numpy as np
from math import sqrt, log
from itertools import chain, product
from collections import defaultdict
import random

def calculate_bow(corpus):
    """
    Calculate bag of words representations of corpus
    Parameters
    ----------
    corpus: list
        Documents represented as a list of string

    Returns
    ----------
    corpus_bow: list
        List of tuple, each tuple contains raw text and vectorized text
    vocab: list
    """
    # YOUR CODE HERE
    def vectorize(sentence, vocab):
        return[sentence.split().count(i) for i in vocab]
    corpus_bow = []
    vocab = sorted(set([token for doc in corpus for token in doc.lower().split()]))
    for i in corpus:
        corpus_bow.append((i, vectorize(i, vocab)))
    return corpus_bow, vocab

def calculate_tfidf(corpus, vocab):
    """
    Parameters
    ----------
    corpus: list of tuple
        Output of calculate_bow()
    vocab: list
        List of words, output of calculate_bow()

    Returns
    corpus_tfidf: list
        List of tuple, each tuple contains raw text and vectorized text
    corpus_tfidf_1: list for q4
        List of tuple, each tuple contains raw text and vectorized text
    ----------

    """
    def termfreq(matrix, doc, term):
        try:
            # YOUR CODE HERE
            total_f = 0
            for w in matrix[doc].keys():
                total_f += matrix[doc][w]
            # print(total_f)
            return matrix[doc][term]/total_f
        except ZeroDivisionError:
            return 0

    def inversedocfreq(matrix, term):
        try:
            # YOUR CODE HERE
            doc_num = len(matrix.keys())
            word_doc_num = 0
            for i in matrix.keys():
                if matrix[i][term] > 1:
                    word_doc_num += 1
                else:
                    word_doc_num += matrix[i][term]
            # print(doc_num/word_doc_num)
            return (doc_num/word_doc_num)
        except ZeroDivisionError:
            return 0

    '''
    function for q4
    '''
    def termfreq1(matrix, doc, term):
        try:
            # YOUR CODE HERE
            K = 0.7
            total_f1 = 0
            max_f1 = 0
            for w in matrix[doc].keys():
                if max_f1 <= matrix[doc][w]:
                    max_f1 = matrix[doc][w]
                else:
                    max_f1 = max_f1
            # for w in matrix.keys():
            #     if max_f1 <= matrix[w][term]:
            #         max_f1 = matrix[w][term]
            #     else:
            #         max_f1 = max_f1
            # print(max_f1)
            # print(K+(1-K)*(matrix[doc][term]/max_f1))
            return (K+(1-K)*(matrix[doc][term]/max_f1))
        except ZeroDivisionError:
            return 0

    # function for q4
    def inversedocfreq1(matrix, term):
        try:
            # YOUR CODE HERE
            doc_num1 = len(matrix.keys())
            word_doc_num1 = 0
            for i in matrix.keys():
                if matrix[i][term] > 1:
                    word_doc_num1 += 1
                else:
                    word_doc_num1 += matrix[i][term]
            # print(log(doc_num1/(1+word_doc_num1))+1)
            return (log(doc_num1/(1+word_doc_num1))+1)
        except ZeroDivisionError:
            return 0

    # YOUR CODE HERE
    doc_term_dic = {}
    for item in corpus:
        tmp_dic = {}
        i = 0
        sent = item[0].split(' ')
        for word in vocab:
            if word in sent:
                tmp_dic[word] = item[1][i]
            else:
                tmp_dic[word] = 0
            i += 1
        doc_term_dic[item[0]] = tmp_dic
    # print(doc_term_dic)
    corpus_tfidf = list()
    corpus_tfidf1 = list()
    tfidf = []
    for item in corpus:
        num_lst = list()
        num_lst1 = list()
        for word in vocab:
            tfidf = termfreq(doc_term_dic,item[0],word)*inversedocfreq(doc_term_dic,word)
            tfidf1 = termfreq1(doc_term_dic,item[0],word)*inversedocfreq1(doc_term_dic,word)
            num_lst.append(tfidf)
            num_lst1.append(tfidf1)
        corpus_tfidf.append((item[0],num_lst))
        corpus_tfidf1.append((item[0],num_lst1))
        tfidf = [corpus_tfidf,corpus_tfidf1]

    return tfidf



def cosine_sim(u,v):
    """
    Parameters
    ----------
    u: list of number
    v: list of number

    Returns
    ----------
    cosine_score: float
        cosine similarity between u and v
    """
    # YOUR CODE HERE
    x = np.array(u)
    y = np.array(v)
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    cosine_score = num / denom
    return cosine_score


def print_similarity(corpus):
    """
    Print pairwise similarities
    """
    for sentx in corpus:
        for senty in corpus:
            print("{:.4f}".format(cosine_sim(sentx[1], senty[1])), end=" ")
        print()
    print()



def q1():
    all_sents = ["this is a foo bar",
                 "foo bar bar black sheep",
                 "this is a sentence"]
    corpus_bow, vocab = calculate_bow(all_sents)
    print(corpus_bow)
    print(vocab)
    print(" ")
    print("Test BOW cosine similarity")
    print_similarity(corpus_bow)

    print("Test tfidf cosine similarity")
    # corpus_tfidf = list(zip(all_sents, calculate_tfidf(corpus_bow, vocab)))
    corpus_tfidf = calculate_tfidf(corpus_bow, vocab)[0]
    print(corpus_tfidf)
    print_similarity(corpus_tfidf)

    print("Test tfidf1 cosine similarity")
    # corpus_tfidf1 = list(zip(all_sents, calculate_tfidf(corpus_bow, vocab)))
    corpus_tfidf1 = calculate_tfidf(corpus_bow, vocab)[1]
    print(corpus_tfidf1)
    print_similarity(corpus_tfidf1)



if __name__ == "__main__":
    q1()
