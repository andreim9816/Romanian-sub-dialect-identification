from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import re
import numpy as np
import pandas as pd

my_token_pattern = r"(?u)\S\S+"

###############################

# Function that opens a file (path) and returns a list of strings, each string representint a single line
def readFile(path):

    file = open(path , 'r' , encoding = "utf-8")
    return file.readlines()

###############################

# readFiles the samples
train_samples = readFile('train_samples.txt')
test_samples = readFile('test_samples.txt')
validation_samples = readFile('validation_samples.txt')

# Gets the ids for the test samples
test_ids = [row for row in test_samples]
test_ids = [int(row.split('\t')[0]) for row in test_ids]

###############################

# Function that returns an array of labels
def get_labels(file):

    labels = [int(label.split('\t')[1]) for label in file]
    return labels

###############################

# readFiles the labels, too, and generates a list of them
train_labels = readFile('train_labels.txt')
train_labels = get_labels(train_labels)

validation_labels = readFile('validation_labels.txt')
validation_labels = get_labels(validation_labels)

print("Converting the samples")

##############################

# Function that gets the full sample ( train and test) and returns a list of samples, without the ids. Does some of the work, too
def convert_to_list_strings(dataSample):

    finalString = [row.split('\t')[1].replace('\n' , '') for row in dataSample]
    return finalString

##############################

# Converts the samples
train_samples = convert_to_list_strings(train_samples)
test_samples = convert_to_list_strings(test_samples)
validation_samples = convert_to_list_strings(validation_samples)

# Concaternates the training, validation and targets samples into one
train_samples = train_samples + validation_samples
train_labels = train_labels + validation_labels

# CountVectorizer used for the Bag Of Words
print("Started generating the dictionary")

##############################

# Function that preprocesses a String composed of many strings
def my_preprocessor(text):

    words = re.split("\\s+", text)
    words = [word.replace("\n", "") for word in words if len(word) > 3] # gets rid of the '\n' character and keeps only the words of length > 3
    return ' '.join(words)

##############################

vectorizer = TfidfVectorizer(analyzer = 'word' , norm = 'l2' , token_pattern = my_token_pattern , lowercase = False , preprocessor = my_preprocessor )
count_vector = vectorizer.fit_transform(train_samples)  # generates the dictionary and learns

# Classifier
classifier = ComplementNB(alpha = 0.3 , fit_prior = True , norm = False)

###############################

# Function that transforms a sample to a matrix document tfidf
def transform(samples , cv):
    return cv.transform(samples).toarray()

##############################

train_samples = transform(train_samples , vectorizer)
test_samples = transform(test_samples , vectorizer)
validation_samples = transform(validation_samples , vectorizer)

print("Started the training")
classifier.fit(train_samples , train_labels)

print("Started making the prediction")

predictions = classifier.predict(test_samples)
predictions_validation = classifier.predict(validation_samples)

## F1 score
print('F1 score is ', f1_score(predictions_validation , validation_labels))
print('Accuracy is ' , accuracy_score(validation_labels , predictions_validation))
print('Confussion matrix is: )')
print(confusion_matrix(validation_labels , predictions_validation))

print("Generates the output file")
csvFile = pd.DataFrame({ "id" : test_ids , "label" : predictions})
csvFile.to_csv("results.csv" , index = False)

print("The program has ended succesfully")
