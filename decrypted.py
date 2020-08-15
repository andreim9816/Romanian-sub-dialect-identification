from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score

import re
import numpy as np
import pandas as pd

porter_stemmer = SnowballStemmer("romanian")

##############################

my_stop_words = ([
     'a', 'abia', 'acea', 'aceasta', 'aceea', 'aceeasi', 'acei', 'aceia', 'acel', 'acela', 'acelasi',
     'acele', 'acelea', 'acest', 'acesta', 'aceste', 'acestea', 'acestei', 'acestia', 'acestui', 'acesti', 'acestia',
     'acolo', 'acord', 'acum', 'adica', 'ai', 'aia', 'aiba', 'aici', 'al', 'ala', 'alaturi', 'ale', 'alea',
     'alt', 'alta', 'alte', 'altfel', 'alti', 'altii', 'altul', 'am', 'anume', 'apoi', 'ar',
     'are', 'as', 'asa', 'asta', 'astazi', 'astea', 'astfel', 'astazi','atunci', 'au', 'avea', 'avem', 'aveti', 'avut',
     'azi', 'as', 'ati', 'b', 'ba', 'bine', 'buna', 'c', 'ca', 'cam', 'capat', 'care',
     'careia', 'carora', 'caruia', 'catre', 'caut', 'ce', 'cea', 'ceea', 'cei', 'ceilalti', 'cel', 'cele',
     'celor', 'ceva', 'chiar', 'cine', 'cineva',
     'conform', 'contra', 'cu', 'cui', 'cum', 'cumva', 'cand', 'catva',
     'caci', 'carei', 'caror', 'carui', 'catre', 'd', 'da',
     'daca', 'dar', 'dat', 'data', 'dau', 'de', 'deasupra', 'deci', 'deja',
     'deoarece', 'departe', 'despre', 'desi', 'din', 'dinaintea', 'dintr', 'dintr-', 'dintre', 'doar', 'doi',
     'doilea', 'doua', 'drept', 'dupa', 'da', 'e', 'ea', 'ei', 'el', 'ele', 'era', 'eram', 'este', 'eu',
     'exact', 'esti', 'f', 'face', 'fara', 'fata', 'fel', 'fi', 'fie', 'fiecare', 'fii', 'fim', 'fiu', 'fiti', 'foarte',
     'fost', 'fara', 'g', 'geaba', 'gratie', 'h', 'halba', 'i', 'ia', 'iar', 'ieri', 'ii', 'il', 'imi', 'in',
     'inainte', 'inapoi',  'intr', 'intre', 'isi', 'iti', 'j', 'k', 'l', 'la', 'le', 'li',
     'lor', 'lui',  'm', 'ma', 'mai', 'mare', 'mea', 'mei', 'mele', 'mereu', 'meu', 'mi', 'mie',
     'mine', 'mod', 'mult', 'multa', 'multe', 'multi', 'multa', 'multi', 'multumesc', 'ma', 'n', 'ne',
     'nevoie', 'ni', 'nici', 'niciodata', 'nicaieri', 'nimeni', 'nimeri', 'nimic', 'niste', 'niste', 'noastre',
     'noastra', 'noi', 'noroc', 'nostri', 'nostru', 'nou', 'noua',  'nostri', 'nu', 'numai', 'o', 'opt', 'or',
     'ori', 'oricare', 'orice', 'oricine', 'oricum', 'oricat',  'oriunde', 'p', 'pai',
     'parca', 'patra', 'patru', 'patrulea', 'pe', 'pentru', 'peste', 'pic', 'plus', 'poate', 'pot', 'prea',
     'prima', 'primul', 'prin', 'printr', 'putini', 'putin', 'putina', 'putina', 'pana', 'r', 'rog', 's', 'sa',
     'sa-mi', 'sa-ti', 'sai', 'sale', 'sau', 'se', 'si', 'spate', 'spre', 'sub', 'sunt', 'suntem',
     'sunteti', 'sus', 'suta', 'sa', 'sai', 'sau', 't', 'ta', 'tale', 'te', 'ti', 'timp',
     'tine', 'toata', 'toate', 'toata', 'tocmai', 'tot', 'toti', 'totul', 'totusi', 'totusi', 'toti', 'trei', 'treia',
     'treilea', 'tu', 'tuturor', 'tai', 'tau', 'u', 'ul', 'ului', 'un', 'una', 'unde', 'undeva', 'unei', 'uneia',
     'unele', 'uneori', 'unii', 'unor', 'unora', 'unu', 'unui', 'unuia', 'unul', 'v', 'va', 'vi', 'voastre', 'voastra',
     'voi', 'vom', 'vor', 'vostru', 'voua', 'vostri', 'vreme', 'vreo', 'vreun', 'va', 'x', 'z', 'zece', 'zero', 'zi',
     'zice', 'ii', 'il', 'imi', 'impotriva', 'in', 'inainte', 'inaintea', 'incotro', 'incat', 'intre',
     'intrucat', 'iti', 'ala', 'alea', 'asta', 'astea', 'astia', 'sapte', 'sase', 'si', 'stiu', 'ti', 'tie',
      'abi', 'ace', 'aceast', 'aceeas', 'acelas', 'adic', 'aib', 'aic', 'aiur', 'alatur', 'altcev', 'altcinev',
      'anum', 'apo', 'asemen', 'astaz', 'aste', 'asti', 'asupr', 'atar', 'atunc', 'ave', 'avet', 'bin', 'bun',
      'cac', 'car', 'caru', 'catr', 'catv', 'cee', 'ceilalt', 'cev', 'cin', 'cinc', 'cinev', 'citev', 'citv',
      'contr', 'cumv', 'dac', 'dator', 'deasupr', 'dec', 'degrab', 'dej', 'deoarec', 'depart', 'des', 'despr',
      'dinaint', 'doil', 'dou', 'dup', 'fac', 'far', 'fat', 'fiec', 'fit', 'foart', 'geab', 'grat', 'halb', 'ier',
      'impotr', 'inaint', 'inapo', 'intruc', 'lang', 'ling', 'main', 'mar', 'mel', 'miin', 'min', 'multum', 'nevoi',
      'nic', 'nicaier', 'niciod', 'nimen', 'nimer', 'nist', 'noastr', 'nostr', 'numa', 'oric', 'oricin', 'oriund',
      'pan', 'parc', 'patr', 'patrul', 'pest', 'pin', 'poat', 'pre', 'prim', 'sal', 'sapt', 'sas', 'spat', 'suntet',
      'sut', 'tal', 'tin', 'toat', 'tocm', 'totus', 'tre', 'treil', 'ulu', 'undev', 'une', 'uneor', 'uni', 'voastr',
      'vostr', 'vou', 'vrem', 'zec', 'zic', 'nevo', 'num'])

##############################

# Function that opens a file (path) and returns a list of strings
def read(path):
    with open(path , encoding = "utf-8") as file:
        x = file.readlines()
    return x

##############################

# Function that gets the full sample ( train and test) and returns a list of samples, without the ids. Does some of the work, too
def convert_to_list_strings(data):

    finalString = []

    for string in data:

        string = string.split('\t')[1] # Gets the sample after the TAB character
        string = ''.join([ ch for ch in string if ch.isdigit() == False]) # Replaces the digits

        # Replaces some Romanian specific letters
        string = string.replace('ș', 's')
        string = string.replace('â', 'a')
        string = string.replace('î', 'i')
        string = string.replace('ț', 't')
        string = string.replace('ă', 'a')

        string = string.replace('Ş', 'S')
        string = string.replace('Ă', 'A')
        string = string.replace('Î', 'I')
        string = string.replace('Ţ', 'T')
        string = string.replace('Â', 'A')

        finalString.append(string) # appends
    return finalString

##############################

# Preprocessing a String composed of many strings
def my_preprocessor(text):

    text = re.sub("\\W[;”,\"„\'.’’01234566789\s»>)(<%…“\n|$NE$]\s*", " ", text)  # Removes special chars
    text = text.lower() # lowers the text

    # Using the stemmer, converts some words
    words = re.split("\\s+" , text)
    stemmed_words = [porter_stemmer.stem(word = word) for word in words] #
    return ' '.join(stemmed_words)

##############################

# Function that normalizes the data, considering a specific Norm, given by the <<type>> argument

def normalize_data(train_data , test_data , source_data , target_data , type = None):
    scaler = None

    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm = 'l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm = 'l2')

    if scaler is not None:

        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        scaled_source_data = scaler.transform(source_data)
        scaled_target_data = scaler.transform(target_data)

        return (scaled_train_data, scaled_test_data , scaled_source_data , scaled_target_data)

    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data , source_data , target_data)

###############################

# Function that returns an array of labels

def get_labels(file):

    labels = [label.split('\t')[1].replace("\n", '') for label in file]
    labels = [int(label) for label in labels]

    return labels

###############################
# Reads the samples
train_samples = read('train_samples.txt')
test_samples = read('test_samples.txt')

# Gets the ids for the test samples
test_ids = [x for x in test_samples]
test_ids = [int(x.split('\t')[0]) for x in test_ids]

validation_source_samples = read('validation_source_samples.txt')
validation_target_samples = read('validation_target_samples.txt')

# Reads the labels, too, and generates a list of them
train_labels = read('train_labels.txt')
train_labels = get_labels(train_labels)

validation_source_labels = read('validation_source_labels.txt')
validation_source_labels = get_labels(validation_source_labels)

validation_target_labels = read('validation_target_labels.txt')
validation_target_labels = get_labels(validation_target_labels)

print("Converting the samples")

# Getting rid of the ids
train_samples = convert_to_list_strings(train_samples)
test_samples = convert_to_list_strings(test_samples)
validation_source_samples = convert_to_list_strings(validation_source_samples)
validation_target_samples = convert_to_list_strings(validation_target_samples)

# Concaternates the training, validation and targets samples into one
train_samples = train_samples + validation_source_samples + validation_target_samples
train_labels = train_labels + validation_source_labels + validation_target_labels


print("l1 = " + str(len(train_samples)))
print("l2 = " + str(len(train_labels)))

# CountVectorizer used for the Bag Of Words
print("Started generating the dictionary")

cv = TfidfVectorizer(preprocessor = my_preprocessor , max_features = 40000 , min_df = 3 , max_df = 0.45 , stop_words = my_stop_words , norm = 'l1')
count_vector = cv.fit_transform(train_samples) # generates the dictionary and learns

print("Dictionary's length is " + str(len(cv.vocabulary_)))

vectorized_train = cv.transform(train_samples) # transforms documents to document-term matrix.
scaled_train = vectorized_train.toarray() # matrix look-alike

vectorized_test = cv.transform(test_samples)
scaled_test = vectorized_test.toarray() # as a matrix

vectorized_source = cv.transform(validation_source_samples)
scaled_source = vectorized_source.toarray()

vectorized_target = cv.transform(validation_target_samples)
scaled_target = vectorized_target.toarray()

# trans

print("Normalizes the data")
classifier = svm.LinearSVC(C = 50)
#scaled_train , scaled_test , scaled_source_data , scaled_target_data = normalize_data(vectorized_train , vectorized_test , vectorized_source , vectorized_target , 'l1')

print("Started the training")
classifier.fit(scaled_train , train_labels)

print(train_labels)
print("Started making the prediction")

predictions = classifier.predict(scaled_test)
predictions_source = classifier.predict(scaled_source)
predictions_target = classifier.predict(scaled_target)

print(predictions_source)
print("Length of the sources " + str(len(predictions_source)))

## F1 score
print('F1 score for the source is ', f1_score(predictions_source, validation_source_labels))
print('F1 score for the target is ', f1_score(predictions_target, validation_target_labels))


print("Generates the output file")
output = pd.DataFrame( data = { "id" : test_ids , "label" : predictions} )
output.to_csv("results.csv" , index = False)

print("The program has ended succesfully")
