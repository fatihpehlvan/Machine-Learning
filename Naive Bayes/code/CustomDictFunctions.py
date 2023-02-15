import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def addOrUpdateDict(dict, key):
    """
    if dictionary contains the key increase the value 1
    else create new key and make the value 1
    :param dict: is a dictionary (such as sports_dict, politics_dict...)
    :param key: is a word
    """
    if key in dict.keys():
        dict.update({key: dict.get(key) + 1})
    else:
        dict[key] = 1


def fillDicts(dict, array):
    """
    add to dictionary all words in the array
    :param dict: is a dictionary (such as sports_dict, politics_dict...)
    :param array: is a list words
    """
    for element in array:
        addOrUpdateDict(dict, element)


def resetDicts(sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict):
    """
    resets all dictionaries
    :param sport_dict: sport dictionary
    :param business_dict: business dictionary
    :param politics_dict: politics dictionary
    :param entertainment_dict: entertainment dictionary
    :param tech_dict: tech dictionary
    """

    sport_dict.clear()
    business_dict.clear()
    politics_dict.clear()
    entertainment_dict.clear()
    tech_dict.clear()


def filterSentences(listOfWords):
    """
    remove stopwords
    :param listOfWords: is a list of word
    :return: a list without stopwords
    """
    stop_words = set(stopwords.words('english'))
    stop_words.add("said")
    filtered_sentence = [w for w in listOfWords if w not in stop_words]
    return filtered_sentence


def makeBigramOrUnigram(sentence, bigram):
    """
    make the array bigram or unigram
    :param sentence: is a list of words
    :param bigram: is boolean value. if it is true make bigram
    :return: a bigram or unigram list
    """
    if bigram:
        filtered_sentence = [(sentence[w - 1] + " " + sentence[w]) for w in
                             range(1, len(sentence))]

    # all sentences are already bigram don't need to change it
    else:
        filtered_sentence = sentence
    return filtered_sentence


def initializeDicts(data_array, sport_dict, business_dict, politics_dict, entertainment_dict, tech_dict, bigram,
                    stopWordsWillBeRemoved=False):
    """
    make Bag Of Words
    :param data_array: our data array which contains text and categories
    :param sport_dict: sport dictionary
    :param business_dict: business dictionary
    :param politics_dict: politics dictionary
    :param entertainment_dict: entertainment dictionary
    :param tech_dict: tech dictionary
    :param bigram: is a boolean value to make bigram or unigram
    :param stopWordsWillBeRemoved: is a boolean value to remove stop words from text
    :return:
    """

    # https://stackoverflow.com/questions/41610543/corpora-stopwords-not-found-when-import-nltk-library
    # If there is "Resource stopwords not found" error, you should download below.
    # nltk.download('stopwords')
    # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

    for i in data_array:
        # remove punctuation from array to make more semantic
        i[0] = i[0].translate(str.maketrans({key: " " for key in string.punctuation + "£$"}))
        listOfWords = word_tokenize(i[0].lower())

        # remove elements which length is less than 2 to make more semantic
        listOfSemanticWords = [ele for ele in listOfWords if len(ele) > 2]

        if stopWordsWillBeRemoved:
            sentence = filterSentences(listOfSemanticWords)
        else:
            sentence = listOfSemanticWords
        filtered_sentence = makeBigramOrUnigram(sentence, bigram)

        # add elements to proper dictionary
        if i[1] == "sport":
            fillDicts(sport_dict, filtered_sentence)
        elif i[1] == "business":
            fillDicts(business_dict, filtered_sentence)
        elif i[1] == "politics":
            fillDicts(politics_dict, filtered_sentence)
        elif i[1] == "entertainment":
            fillDicts(entertainment_dict, filtered_sentence)
        elif i[1] == "tech":
            fillDicts(tech_dict, filtered_sentence)
