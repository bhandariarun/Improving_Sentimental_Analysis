import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spellchecker import SpellChecker
import re

def pol_scores(word):
    cleaned_text = re.sub(r'(.)\1{2,}', r'\1\1', word)
    sp=SpellChecker()
    sp.distance=6
    correctedword=sp.correction(cleaned_text)
    sentiment_score=SentimentIntensityAnalyzer().polarity_scores(correctedword)
    # positive=sentiment_score['pos']
    # negative=sentiment_score['neg']
    if len(correctedword)<len(word):
        extra_score=(len(word)-len(correctedword))*0.01
    else:
        extra_score=0
    sentiment_score.update({'extrascore':extra_score})
    return sentiment_score

stop_words = set(stopwords.words('english'))

file_path = 'emotionsfile.txt'

with open(file_path, 'r') as file:
    content = file.read()

words = word_tokenize(content)

filtered_words = [word for word in words if word.lower() not in stop_words]

positive_scores=0
negative_scores=0
for word in filtered_words:
    try:
        scores=pol_scores(word)
        pos=scores['pos']
        neg=scores['neg']
        extra_score=scores['extrascore']
        if pos>0:
            pos=pos+extra_score
        if neg>0:
            neg=neg+extra_score
        positive_scores=positive_scores+pos
        negative_scores=negative_scores+neg
    except:
        pass

print(positive_scores)
print(negative_scores)
