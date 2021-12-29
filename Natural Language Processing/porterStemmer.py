# Autjor: Boris Kundu
# Run Porter Stemmer on an input sentence.

#Import packages
import nltk
from nltk.stem import PorterStemmer
#nltk.download('punkt')

#Get list pof stop words.
stopwords = set(nltk.corpus.stopwords.words('english'))
#Input sentence
sentence = input("Enter a sentence to run Porter Stemmer:")
#Tokenize sentence
tokens = nltk.word_tokenize(sentence)
print(f'Tokens:\n{tokens}')
#Exclude stop words
tokens2 = [t for t in tokens if t.lower() not in stopwords]
print(f'Tokens after removing stop words:\n{tokens2}')
#Initialize stemmer
ps = PorterStemmer()
print('Displaying stemmed tokens:\n')
for w in tokens2:
    print(w, ":", ps.stem(w))