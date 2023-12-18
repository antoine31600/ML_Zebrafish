
# Identification et classement des différents évènements de nage de poissons zèbres

Ce dépôt Github a été créé dans le cadre d'un projet effectué par Bastien Bédard (bastien.bedard.1@ulaval.ca), Camille Poitras(camille.poitras.2@ulaval.ca) et Antoine Chartier (antoine.chartier.1@ulaval.ca) en collaboration avec Antoine Légaré (antoine.legare.1@ulaval.ca), étudiant en 2ème année au doctorat en biophotonique au centre de recherche Cervo. Ce projet s'inscrit dans le cours d'introduction à l'apprentissage automatique (GLO-4101) de l'Université Laval pour le professeur Christian Gagné.

Les données utilisées dans le cadre de ce projet ainsi que le fichier "utilities.py" ont été fournies par Antoine Légaré

Ce projet a pour but d'utiliser des données des mouvements de nageoires de poissons-zèbres provenant du CERVO afin d'identifier les évènements de nages à l'aide d'apprentissage supervisé, puis de les classifier selon différentes méthodes de nages à l'aide d'apprentissage non-supervisé.


# Méthodes

L'identification de nage a été réalisé avec une segmentation manuelle des événements de nages des poissons-zèbres de différents types. Les données présentes dans les fichiers "dataset1-wholebrain" et "dataset2-mutants" ont servis de "groundtruth" et les évènements à l'intérieur de ces données ont été segmentées manuellement et enregistrées dans le fichier "SegmentationManuell_ML.xlsx", tout ceci dans le dossier "data". Les événements segmentés ont été transformés en vecteur binary par les fichiers "ConversionBinaryVector" et "utilities" afin de servir d'entrainement à un RNN de type LSTM présent dans "RNN_event_identifications", tout ceci dans le dossier "function". Tout ceci afin d'avoir un RNN entrainé à identifier les évènements de nages.

De ces événements ont ensuite été extrait les différentes propriétés de nages. Puis ces vecteurs de propriétés ont été utilisés afin de les classifier avec des méthodes de partitionnement et de réduction dimensionnelle par le fichier "Clustering.py". Les classes utilisées ont été déterminées par des experts du comportent du poisson zèbres pour ne garder seulement que les 10 plus importantes.

Des exemples de résultats sont présents dans le dossier "Example_results"


# Utilisation

-Télécharger l'entiereté du dépot et le déziper dans un dossier de votre choix

-Le RNN présent dans ce dépot est déja entrainé. Cependant, si besoin est de le réentrainer, il faut simplement changer le path des datasets présents dans le fichier "ConversionBinaryVector" par les votres.
