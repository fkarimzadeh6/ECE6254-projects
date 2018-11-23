# This code is provided by Foroozan Karimzadeh- PhD student at Gatech
# coding: utf-8
# In[155]:

import numpy as np
import json
from sklearn.feature_extraction import text

x = open('fedpapers_split.txt').read()
papers = json.loads(x)

papersH = papers[0] # papers by Hamilton 
papersM = papers[1] # papers by Madison
papersD = papers[2] # disputed papers

nH, nM, nD = len(papersH), len(papersM), len(papersD)

# This allows you to ignore certain common words in English
# You may want to experiment by choosing the second option or your own
# list of stop words, but be sure to keep 'HAMILTON' and 'MADISON' in
# this list at a minimum, as their names appear in the text of the papers
# and leaving them in could lead to unpredictable results
# stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON','MADISON'})
stop_words = {'HAMILTON', 'MADISON'}

## Form bag of words model using words used at least 10 times
vectorizer = text.CountVectorizer(stop_words=stop_words,min_df=10)
X = vectorizer.fit_transform(papersH+papersM+papersD).toarray()

# Uncomment this line to see the full list of words remaining after filtering out 
# stop words and words used less than min_df times
# vectorizer.vocabulary_

# Split word counts into separate matrices
XH, XM, XD = X[:nH,:], X[nH:nH+nM,:], X[nH+nM:,:]

# Estimate probability of each word in vocabulary being used by Hamilton (this is P(word|class=Hamilton))
# Applying Laplace smoothing
fH = [ (XH.sum(axis = 0, dtype='float')+1)/(XH.sum(dtype='float')+XH.shape[1]) ]
print('#########################################################################')
print('P(word|class=Hamilton)= '), print(fH)

# Estimate probability of each word in vocabulary being used by Madison (this is P(word|class=Madison))
# Applying Laplace smoothing
fM = [ (XM.sum(axis = 0, dtype='float')+1)/(XM.sum(dtype='float')+XM.shape[1]) ]
print('P(word|class=Madison)= '), print(fM)

# Compute ratio of these probabilities
fH_num=np.array(fH, dtype='float')
fM_denum=np.array(fM, dtype='float')
fratio = fH_num/fM_denum
print('#########################################################################')
print('fratio=' ), print(fratio)
print('#########################################################################')

# Compute prior probabilities 
piH = nH/(nH+nM)
piM = nM/(nH+nM)
piratio = piH/piM

threshold = 0.0001
for doc in range(len(XD)): # Iterate over disputed documents
    element_power = fratio**XD[doc, :]
    # Compute likelihood ratio for Naive Bayes model
    LR = piratio*np.prod(element_power)
    print('LR of doc (%d) is =' %(doc)), print(LR)
    if LR > threshold:
        print('This document is by Hamilton')
    else:
        print('This document is by Madison')
print('#########################################################################')
    

