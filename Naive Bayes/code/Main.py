import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from CustomDictFunctions import *
import math

from DrawPieGraph import drawGraph


def numberOfUniqueWords(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict):
    """
    This function calculate number of unique words
    :param dicionaries: key values are words
    :return: number of unique words
    """

    return len(set(sport_dict.keys()) | set(business_dict.keys()) | set(politics_dict.keys()) | set(
        entertainment_dict.keys()) | set(tech_dict.keys()))


def totals(smooth=0):
    """
    calculate Laplacian smoothing for all categories
    :param smooth: smooth is an integer value to increase denominator
    :return: a list calculates denominator sport, business, politics, entertainment and tech in order
    """

    return [sum(sport_dict.values()) + smooth, sum(business_dict.values()) + smooth,
            sum(politics_dict.values()) + smooth, sum(entertainment_dict.values()) + smooth, sum(
            tech_dict.values()) + smooth]


def probabilityCategory(y_train):
    """
    calculate log probability of a category P(sport), P(business) etc.
    :param y_train:
    :return: list of log probability
    """

    countTrainCategories = np.zeros(5, dtype=int)

    for i in y_train:
        if i == "sport":
            countTrainCategories[0] += 1
        elif i == "business":
            countTrainCategories[1] += 1
        elif i == "politics":
            countTrainCategories[2] += 1
        elif i == "entertainment":
            countTrainCategories[3] += 1
        elif i == "tech":
            countTrainCategories[4] += 1
        else:
            "ERROR"
    return np.log(countTrainCategories / len(y_train))


def noSmooth(dict_total, filtered_sentence):
    """
    Generally we need to smooth. However, sometimes we may not need it. Revert the smoothing
    :param dict_total:
    :param filtered_sentence:
    :return: list of probability and a boolean value for the numerator
    """

    count = 0
    bool = False
    for j in filtered_sentence:
        if j in sport_dict.keys():
            count += 1
        if j in business_dict.keys():
            count += 1
        if j in politics_dict.keys():
            count += 1
        if j in entertainment_dict.keys():
            count += 1
        if j in tech_dict.keys():
            count += 1
    if count == len(filtered_sentence) * 5:
        dict_total = totals()
        bool = True
    return dict_total, bool


def calculateYPredict(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, countTrainCategories,
                      dict_total, bigram=False):
    """
    this function calculate y predict.
    :param sport_dict: sport dictionary
    :param business_dict: business dictionary
    :param politics_dict: politics dictionary
    :param entertainment_dict: entertainment dictionary
    :param tech_dict: tech dictionary
    :param countTrainCategories: log probability list for each categories
    :param dict_total: denominator list for each categories
    :param bigram: determine bigram or unigram
    :return: predicted list
    """

    y_predict = []
    for i in X_test:
        # replace punctuation to whitespace from the text, because we need to get meaningful elements from text
        i = i.translate(str.maketrans({key: " " for key in string.punctuation + "£$"}))

        # make an array from text
        i = filterSentences(word_tokenize(i.lower()))

        # make bigram or unigram given list
        filtered_sentence = makeBigramOrUnigram(i, bigram)

        # recalculate we don't need laplacian smoothing
        dict_total, makeNumeratorZero = noSmooth(dict_total, filtered_sentence)

        # add log probabilities in make predict list
        makePredict = np.copy(countTrainCategories)
        for j in filtered_sentence:

            # if there is laplacian smoothing we add one, else we don't need
            if makeNumeratorZero:
                countPredictCategories = np.zeros(5, dtype=int)
            else:
                countPredictCategories = np.ones(5, dtype=int)
            if j in sport_dict.keys():
                countPredictCategories[0] += sport_dict.get(j)
            if j in business_dict.keys():
                countPredictCategories[1] += business_dict.get(j)
            if j in politics_dict.keys():
                countPredictCategories[2] += politics_dict.get(j)
            if j in entertainment_dict.keys():
                countPredictCategories[3] += entertainment_dict.get(j)
            if j in tech_dict.keys():
                countPredictCategories[4] += tech_dict.get(j)

            # Calculate log probability for the given text for each categories
            for k in range(len(makePredict)):
                makePredict[k] += math.log(countPredictCategories[k] / dict_total[k])
        # make predict consists of negative numbers we need to put them all between 0 and 1
        makePredict = makePredict / np.sum(makePredict)

        # however, smaller numbers give higher probabilities due to the above operation. We need to reverse them
        makePredict = 100 / makePredict

        # again make them between 0 and 1
        makePredict /= np.sum(makePredict)
        # find the highest probability
        index = makePredict.argmax()

        # add in the predicted list
        if index == 0:
            y_predict.append("sport")
        elif index == 1:
            y_predict.append("business")
        elif index == 2:
            y_predict.append("politics")
        elif index == 3:
            y_predict.append("entertainment")
        elif index == 4:
            y_predict.append("tech")

    return y_predict


# Read the data with pandas
data = pd.read_csv("English Dataset.csv")

# Print first 5 elements
print(data.head(5))

# Put elements in numpy array except ID
data_array = data.to_numpy()[::, 1::]

# shuffle the array
np.random.shuffle(data_array)

# print(data_array)

# PART 1

# define dictionaries

sport_dict = {}
business_dict = {}
politics_dict = {}
entertainment_dict = {}
tech_dict = {}

# initialize dictionaries with the data set and by removing stopwords
initializeDicts(data_array, sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, False, True)

# draw pie graphs for most common 10 words and explode most 3 words
print()
drawGraph(sport_dict, "sport", data_array)
drawGraph(business_dict, "business", data_array)
drawGraph(politics_dict, "politics", data_array)
drawGraph(entertainment_dict, "entertainment", data_array)
drawGraph(tech_dict, "tech", data_array)

# PART 2

# split the data train and test valur (%80 train, %20 test)
X_train, X_test, y_train, y_test = train_test_split(data_array[:, 0], data_array[:, 1], test_size=0.2)

# merge the train data for my custom function (initializeDicts)
train_data = np.array((X_train, y_train), order='F').T

# reset the dictionaries for reusing
resetDicts(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict)

# UNIGRAM

# initialize dictionaries with the train set with unigram words.
initializeDicts(train_data, sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, False)

# determine number of unique words for laplacian smoothing
word_set_number = numberOfUniqueWords(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict)

# make smoothing
dict_total = totals(word_set_number)

# calculate log probabilities for each categories
countTrainCategories = probabilityCategory(y_train)

# make predict
y_predict = calculateYPredict(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict,
                              countTrainCategories, dict_total)

print(accuracy_score(y_test, y_predict))

# BIGRAM
# reset the dictionaries for reusing
resetDicts(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict)

# initialize dictionaries with the train set with bigram words.
initializeDicts(train_data, sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, True)

# determine number of unique words for laplacian smoothing
word_set_number = numberOfUniqueWords(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict)

# make smoothing
dict_total = totals(word_set_number)

# calculate log probabilities for each categories
countTrainCategories = probabilityCategory(y_train)

# make predict
y_predict = calculateYPredict(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict,
                              countTrainCategories, dict_total, True)

print(accuracy_score(y_test, y_predict))

# Part 3

#### Unigram with stop-words

resetDicts(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict)

initializeDicts(train_data, sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, False)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def makestr(dict):
    str = ""
    for i, j in dict.items():
        i = i + " "
        str += j*i
    return str

#Transform a count matrix to a normalized tf or tf-idf representation.
corpus = [makestr(sport_dict), makestr(business_dict), makestr(politics_dict), makestr(entertainment_dict), makestr(tech_dict)]

vocab = list(set(sport_dict.keys()).union(set(business_dict.keys())).union(set(politics_dict.keys())).union(set(tech_dict.keys())).union(set(entertainment_dict.keys())))

pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                 ('tfid', TfidfTransformer())]).fit(corpus)

counts=pd.DataFrame(pipe["count"].transform(corpus).toarray(), columns=pipe["count"].vocabulary, index=["sport","business","politics","entertainment","tech"])

idf_values=pd.DataFrame(pipe["tfid"].idf_, columns=["idf"], index=pipe["count"].vocabulary)
tfidf_result=pd.DataFrame(pipe.transform(corpus).toarray(), columns=pipe["count"].vocabulary, index=["sport","business","politics","entertainment","tech"]).T

sub_sport_dict={}
sub_business_dict={}
sub_entertainment_dict={}
sub_tech_dict={}
sub_politics_dict={}

for category in ["sport","business", "entertainment", "tech", "politics"]:
    for key in (tfidf_result[category].sort_values(ascending=False)[0:10].index):
        if category=="sport":
            sub_sport_dict[key] = counts[key][category]
        elif category=="business":
            sub_business_dict[key] = counts[key][category]
        elif category=="entertainment":
            sub_entertainment_dict[key] = counts[key][category]
        elif category=="politics":
            sub_politics_dict[key]=counts[key][category]
        else:
            sub_tech_dict[key]=counts[key][category]

    for key in (tfidf_result[category].sort_values(ascending=False)[-10:].index):
        if category == "sport":
            sub_sport_dict[key] = counts[key][category]
        elif category == "business":
            sub_business_dict[key] = counts[key][category]
        elif category == "entertainment":
            sub_entertainment_dict[key] = counts[key][category]
        elif category == "politics":
            sub_politics_dict[key] = counts[key][category]
        else:
            sub_tech_dict[key] = counts[key][category]


word_set_number = numberOfUniqueWords(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict)

dict_total = totals(word_set_number)

countTrainCategories = probabilityCategory(y_train)

y_predict = calculateYPredict(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict, countTrainCategories, dict_total)

print(accuracy_score(y_test, y_predict))

#### Unigram non-stopwords

from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
all_stopwords=set(stopwords.words("english")).union(set(ENGLISH_STOP_WORDS))
all_stopwords.add("said")
tfidf_result.drop(index=all_stopwords,errors="ignore",inplace=True)
sub_sport_dict.clear()
sub_business_dict.clear()
sub_entertainment_dict.clear()
sub_tech_dict.clear()
sub_politics_dict.clear()

for category in ["sport","business", "entertainment", "tech", "politics"]:
    for key in (tfidf_result[category].sort_values(ascending=False)[0:10].index):
        if category=="sport":
            sub_sport_dict[key] = counts[key][category]
        elif category=="business":
            sub_business_dict[key] = counts[key][category]
        elif category=="entertainment":
            sub_entertainment_dict[key] = counts[key][category]
        elif category=="politics":
            sub_politics_dict[key]=counts[key][category]
        else:
            sub_tech_dict[key]=counts[key][category]

    for key in (tfidf_result[category].sort_values(ascending=False)[-10:].index):
        if category == "sport":
            sub_sport_dict[key] = counts[key][category]
        elif category == "business":
            sub_business_dict[key] = counts[key][category]
        elif category == "entertainment":
            sub_entertainment_dict[key] = counts[key][category]
        elif category == "politics":
            sub_politics_dict[key] = counts[key][category]
        else:
            sub_tech_dict[key] = counts[key][category]

word_set_number = numberOfUniqueWords(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict)

dict_total = totals(word_set_number)

countTrainCategories = probabilityCategory(y_train)

y_predict = calculateYPredict(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict,countTrainCategories, dict_total)

print(accuracy_score(y_test, y_predict))

#### Bigram with stopwords
resetDicts(sport_dict,business_dict,politics_dict,entertainment_dict,tech_dict)
initializeDicts(train_data, sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, True)

sport=pd.DataFrame.from_dict(sport_dict, orient="index").T
business=pd.DataFrame.from_dict(business_dict, orient="index").T
entertainment=pd.DataFrame.from_dict(entertainment_dict, orient="index").T
politicts=pd.DataFrame.from_dict(politics_dict, orient="index").T
tech=pd.DataFrame.from_dict(tech_dict, orient="index").T

counts_data=pd.concat([sport,business,entertainment,politicts,tech], ignore_index=True).rename(index={0:"sport", 1:"business",2:"entertainment",3:"politics",4:"tech"}).fillna(0)

tfidf_transformer=TfidfTransformer()
tfidf_transformer.fit(counts_data)
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=counts_data.columns, columns=["idf_weights"])

tf_idf_vector=tfidf_transformer.transform(counts_data)
tfidf_result=pd.DataFrame(tf_idf_vector.toarray(), columns=counts_data.columns, index=counts_data.index).T

sub_sport_dict={}
sub_business_dict={}
sub_entertainment_dict={}
sub_tech_dict={}
sub_politics_dict={}

for category in ["sport","business", "entertainment", "tech", "politics"]:
    for key in (tfidf_result[category].sort_values(ascending=False)[0:10].index):
        if category=="sport":
            sub_sport_dict[key] = counts_data[key][category]
        elif category=="business":
            sub_business_dict[key] = counts_data[key][category]
        elif category=="entertainment":
            sub_entertainment_dict[key] = counts_data[key][category]
        elif category=="politics":
            sub_politics_dict[key]=counts_data[key][category]
        else:
            sub_tech_dict[key]=counts_data[key][category]

    for key in (tfidf_result[category].sort_values(ascending=False)[-10:].index):
        if category == "sport":
            sub_sport_dict[key] = counts_data[key][category]
        elif category == "business":
            sub_business_dict[key] = counts_data[key][category]
        elif category == "entertainment":
            sub_entertainment_dict[key] = counts_data[key][category]
        elif category == "politics":
            sub_politics_dict[key] = counts_data[key][category]
        else:
            sub_tech_dict[key] = counts_data[key][category]

word_set_number = numberOfUniqueWords(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict)

dict_total = totals(word_set_number)

countTrainCategories = probabilityCategory(y_train)

y_predict = calculateYPredict(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict,countTrainCategories, dict_total)

print(accuracy_score(y_test, y_predict))

#### Bigram non-stopwords
resetDicts(sport_dict,business_dict,politics_dict,entertainment_dict,tech_dict)
initializeDicts(train_data, sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, True,stopWordsWillBeRemoved=True)

sport=pd.DataFrame.from_dict(sport_dict, orient="index").T
business=pd.DataFrame.from_dict(business_dict, orient="index").T
entertainment=pd.DataFrame.from_dict(entertainment_dict, orient="index").T
politicts=pd.DataFrame.from_dict(politics_dict, orient="index").T
tech=pd.DataFrame.from_dict(tech_dict, orient="index").T

counts_data=pd.concat([sport,business,entertainment,politicts,tech], ignore_index=True).rename(index={0:"sport", 1:"business",2:"entertainment",3:"politics",4:"tech"}).fillna(0)

tfidf_transformer=TfidfTransformer()
tfidf_transformer.fit(counts_data)
df_idf=pd.DataFrame(tfidf_transformer.idf_,index=counts_data.columns, columns=["idf_weights"])
df_idf.sort_values(by=["idf_weights"])

tf_idf_vector=tfidf_transformer.transform(counts_data)
tfidf_result=pd.DataFrame(tf_idf_vector.toarray(), columns=counts_data.columns, index=counts_data.index).T
tfidf_result

sub_sport_dict={}
sub_business_dict={}
sub_entertainment_dict={}
sub_tech_dict={}
sub_politics_dict={}

for category in ["sport","business", "entertainment", "tech", "politics"]:
    for key in (tfidf_result[category].sort_values(ascending=False)[0:10].index):
        if category=="sport":
            sub_sport_dict[key] = counts_data[key][category]
        elif category=="business":
            sub_business_dict[key] = counts_data[key][category]
        elif category=="entertainment":
            sub_entertainment_dict[key] = counts_data[key][category]
        elif category=="politics":
            sub_politics_dict[key]=counts_data[key][category]
        else:
            sub_tech_dict[key]=counts_data[key][category]

    for key in (tfidf_result[category].sort_values(ascending=False)[-10:].index):
        if category == "sport":
            sub_sport_dict[key] = counts_data[key][category]
        elif category == "business":
            sub_business_dict[key] = counts_data[key][category]
        elif category == "entertainment":
            sub_entertainment_dict[key] = counts_data[key][category]
        elif category == "politics":
            sub_politics_dict[key] = counts_data[key][category]
        else:
            sub_tech_dict[key] = counts_data[key][category]

word_set_number = numberOfUniqueWords(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict)

dict_total = totals(word_set_number)

countTrainCategories = probabilityCategory(y_train)

y_predict = calculateYPredict(sub_sport_dict, sub_business_dict, sub_politics_dict, sub_entertainment_dict, sub_tech_dict,countTrainCategories, dict_total)

print(accuracy_score(y_test, y_predict))
