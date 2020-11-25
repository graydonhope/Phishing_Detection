import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# Get the current working directory 
directory_path = os.path.dirname(os.path.realpath(__file__))

# Load in the data
phishing_data = pd.read_csv(directory_path + '/dataset_csv/data_phishing.csv')
legit_data = pd.read_csv(directory_path + '/dataset_csv/data_legitimate.csv')

def bag_of_words(corpus):
    '''
    Returns the bag of words 
    @corpus: String Array of strings - the data to be parsed in the bag of words operation
    @return scipy Sparse Matrix 
    documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform
    '''
    
    vectorizer = CountVectorizer()
    print(vectorizer.fit_transform(corpus).todense())
    print(vectorizer.vocabulary_)
    return vectorizer.fit_transform(corpus)

def n_grams(corpus):
    '''
    ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams
    '''
    vectorizer = CountVectorizer(ngram_range=(1,3))
    
    return vectorizer.fit_transform(corpus)

# Examples - need to determine how to split up URLs
test_strings = ["Hello my name is graydon", "I am a cat and I have a name"]
bag_of_words_result = bag_of_words(test_strings)
print(bag_of_words_result)
print()
print(n_grams(test_strings))