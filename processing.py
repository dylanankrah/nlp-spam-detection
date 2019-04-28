# -*- coding: utf-8 -*-

# viz
import seaborn as sn
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# dataframes
import pandas as pd

# text processing
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# ml
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# --- Formattage

def Format(dataset):
    """
    Formatte le dataset en entrée, en retirant les colonnes inutiles et en créeant une nouvelle feature
    """
    dataset.drop(columns='Unnamed: 0', inplace=True) # la colonne ID est déjà présente par défaut
    dataset['length'] = dataset['SMS'].apply(len) # on peut rajouter une feature "longueur du message", la corrélation est possible à priori
    # dataset['label'] = dataset['label'].map({'ham': 0, 'spam': 1}, inplace=True) # on convertit les catégories ham/spam en 0/1 (optionnel)
    
    
# --- Visualisation pre/post

def VizHistogram(dataset):
    """
    Renvoie un histogramme des répartitions en longueur des messages, puis par ham/spam
    """
    dataset['length'].plot(bins=100, kind='hist', cmap='coolwarm') # histogramme: répartition en longueur
    dataset.hist(column='length', by='label', bins=100, figsize=(12,4)) # histogramme: idem, par catégorie
    
def VizWordCloud(dataset,cat):
    """
    Génération du wordcloud selon la catégorie
    """
    words = ' '.join(list(dataset[dataset['label'] == cat]['SMS']))
    wordCloud = WordCloud(width=512, height=512).generate(words)
    plt.figure(figsize=(10,8), facecolor='k')
    plt.imshow(wordCloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
def VizReport(clreport, cm, ax):
    """
    Affiche le classification_report et une confusion matrix
    """
    print(clreport) # renvoie les scores type accuracy, recall, f1
    dfCM = pd.DataFrame(cm, index=[['actual','actual'],['ham','spam']], columns=[['predicted','predicted'],['ham','spam']])
    sn.heatmap(dfCM, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax)
    
    
# --- Processing
    
def TextProcess(sms):
    """
    On retire la ponctuation et les stopwords (mots très récurrents, dans la langue anglaise ici), puis on retourne la liste des mots du message
    """
    
    noPunctuationSMS = [c for c in sms if c not in string.punctuation] # on retire la ponctuation
    noPunctuationSMS = ''.join(noPunctuationSMS) # on reforme le message
    
    return [word for word in noPunctuationSMS.split() if word.lower() not in stopwords.words('english')] # on retire les mots vides

# --- Pipeline

def Predict(dataset, model='NB'):
    """
    Séparation du dataset en set training et set test (25-75%), puis prédictions. Pour la pipeline:
        - On convertit les messages en un bag-of-words (ligne = mot unique, colonne = message, 0/1) (comptage)
        - On convertit le BOW en une matrice TF-IDF (pondération)
        - On utilisera un modèle type Naive Bayes par défaut.
        
    Renvoie un array [liste des prédicitons, liste des vrais labels], un classification_report et une confusion matrix
    """
    smsTrain, smsTest, labelTrain, labelTest = train_test_split(dataset['SMS'], dataset['label'], test_size=0.25) # splitting
    
    # création de la pipeline
    if model=='SVC':
        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=TextProcess)),
            ('tfidf', TfidfTransformer()),
            ('classifier', LinearSVC(loss='hinge')),
            ])
    
    else:
        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=TextProcess)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB()),
            ])
    
    pipeline.fit(smsTrain, labelTrain) # fitting
    predictions = pipeline.predict(smsTest) # prédiction
    predVStrue = [predictions,labelTest]
    
    return(predVStrue, classification_report(predictions, labelTest), confusion_matrix(labelTest,predictions))
    
    
# --- Export en CSV
    
def ExportToCSV(true, pred, filename):
    """
    Crée un fichier CSV qui recense les prédictions vs. vrais labels sur le training set
    """
    df = true.to_frame(name='trueLabels')
    df['predictedLabels'] = pred
    df.to_csv(filename)
    
