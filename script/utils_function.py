from tensorflow.keras.utils import pad_sequences
import numpy as np

# Function that takes two arguments: word to be analyzed and a caseLookup dictionary that associates casing styles with some value
def getCasing(word, caseLookup):
    casing = 'other' # default casing
    numDigits = 0
    
    # Checks if the current character is digit 
    for char in word:
        if char.isdigit():
            numDigits += 1 

    digitFraction = numDigits / float(len(word))

    # Check if it is numeric
    if word.isdigit():
        casing = 'numeric'

    elif digitFraction > 0.5:
        casing = 'mainly_numeric'

    elif word.islower():
        casing = 'allLower'

    elif word.isupper():
        casing = 'allUpper'

    elif word[0].isupper():
        casing = 'initialUpper'

    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]

# Fuction to create batches which returns dataset
def createBatches(data):

    lengthList = []
    for i in data:
        lengthList.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in lengthList:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len

def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []
    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []
        flag = False

        for word, char, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            
            charIdx = []
            ADD_TO_DATA = True

            for x in char:
                if x in char2Idx.keys():
                    charIdx.append(char2Idx[x])
                else:
                    flag = True
                    ADD_TO_DATA = False
                    break
            
            if flag:
                break

            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        if ADD_TO_DATA:
            dataset.append([wordIndices, caseIndices, charIndices,labelIndices])

    return dataset

def padding(sentences):
    maxlen = 40
    for sentence in sentences:
        char = sentence[2] # Sentence[2] represents character level information ---> sentence[1] represents case info; sentence[0] represents word-level information
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(sentences):
        sentences[i][2] = pad_sequences(sentences[i][2], 40, padding='post') # It checks the i sentences character and if the length is not 40 then it will add padding after the sequence
    return sentences

def iterate_minibatches(dataset, batch_length):
    start = 0
    for i in batch_length:
        tokens = []
        casing = []
        char = []
        labels = []
        data = dataset[start:i]
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            casing.append(c)
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels), np.asarray(tokens), np.asarray(casing), np.asarray(char)

def createTensor(sentence, word2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    wordIndices = []
    caseIndices = []
    charIndices = []
    for word, char in sentence:
        word = str(word)
        if word in word2Idx:
            wordIdx = word2Idx[word]
        elif word.lower() in word2Idx:
            wordIdx = word2Idx[word.lower()]
        else:
            wordIdx = unknownIdx
        charIdx = []
        for x in char:
            if x in char2Idx.keys():
                charIdx.append(char2Idx[x])
            else:
                charIdx.append(char2Idx['UNKNOWN'])
        wordIndices.append(wordIdx)
        caseIndices.append(getCasing(word, case2Idx))
        charIndices.append(charIdx)
    return [wordIndices, caseIndices, charIndices]

def addCharInformation(sentence):
    return [[word, list(str(word))] for word in sentence]
