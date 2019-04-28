# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from processing import VHistogram, VWordCloud, Format, Predict, VReport, ExportToCSV

# --- Import des données

dataset = pd.read_csv('data/spam.csv', encoding='latin-1')
# import nltk
# nltk.download() > exécuté une fois

# --- Formattage des données et création d'une nouvelle feature

Format(dataset)

# --- Data visualisation

# VHistogram(dataset)
# VWordCloud(dataset)

# --- Prédiction et export des résultats

[predictedLabelsNB,trueLabelsNB], classificationReportNB, confusionMatrixNB = Predict(dataset,'NB')
[predictedLabelsSVC,trueLabelsSVC], classificationReportSVC, confusionMatrixSVC = Predict(dataset,'SVC')

# --- Export des résultats sous format .csv


ExportToCSV(trueLabelsNB, predictedLabelsNB, 'resultsNB.csv')
ExportToCSV(trueLabelsSVC, predictedLabelsSVC, 'resultsSVC.csv')

# --- Data viz post prédiction

fig, (ax1,ax2) = plt.subplots(1,2)

print('NB: \n')
VReport(classificationReportNB, confusionMatrixNB, ax=ax1)

print('SVC: \n')
VReport(classificationReportSVC, confusionMatrixSVC, ax=ax2)

plt.show()