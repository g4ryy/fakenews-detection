# Handle imports
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from matplotlib import pyplot as plt

# Can remove after downloading once
# nltk.download('stopwords')
# nltk.download('punkt')


# Additional functions defined

def removePunctuation(text):
    cleanedText = "".join(c for c in text if c not in string.punctuation)
    return cleanedText

def removeStopwords(text):
    stop = stopwords.words('english')
    return ' '.join([word for word in text.split() if word not in (stop)])

# Read data (Obtained from ISOT dataset)
trueData = pd.read_csv('data/True.csv')
fakeData = pd.read_csv('data/Fake.csv')

# Data Cleaning 

# Converting all text & title to lowercase
trueData['text'] = trueData['text'].apply(lambda x: x.lower())
trueData['title'] = trueData['title'].apply(lambda x: x.lower())
fakeData['text'] = fakeData['text'].apply(lambda x: x.lower())
fakeData['title'] = fakeData['title'].apply(lambda x: x.lower())

# Remove punctuations in text & title
trueData['text'] = trueData['text'].apply(removePunctuation)
trueData['title'] = trueData['title'].apply(removePunctuation)
fakeData['text'] = fakeData['text'].apply(removePunctuation)
fakeData['title'] = fakeData['title'].apply(removePunctuation)


# Removing stop words in text
trueData['text'] = trueData['text'].apply(removeStopwords)
fakeData['text'] = fakeData['text'].apply(removeStopwords)



# Data Visualisation

print("Word Cloud for FAKE DATASET: ")
fake_words = ' '.join([text for text in fakeData.text])
fakeWordCloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(fake_words)

plt.figure(figsize=(10,7))
plt.imshow(fakeWordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


print("Word Cloud for TRUE DATASET: ")
true_words = ' '.join([text for text in trueData.text])
trueWordCloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(true_words)

plt.figure(figsize=(10,7))
plt.imshow(trueWordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()






