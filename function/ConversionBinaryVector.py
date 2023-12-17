from utilities import *
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import ast

plt.rcParams['font.size'] = 18

path_dataset1 = 'C:/Users/bedba/OneDrive/Git/test/projet_M-L/dataset1-wholebrain/'
path_dataset2 = 'C:/Users/bedba/OneDrive/Git/test/Projet_M-L/dataset2-mutants/'

paths = [path_dataset1, path_dataset2]

files1 = identify_files(path_dataset1, ['tail_angles', '.npy'])
files2 = identify_files(path_dataset2, ['tail_angles', '.npy'])

fps=399.75

tails, filess= [], []

for path in paths:
    files = identify_files(path, ['tail_angles', '.npy'])
    for file in files:
        filess.append(file)
def truc_pour_tail_data(ind):
    file = filess[ind]
    if ind <= 19:
        angles = np.load(paths[0] + file)
    else:
        angles = np.load(paths[1] + file)
    tail = TailAnalysis(angles, fps)
    tail.detect_swim_bouts(threshold=0.75)
    tails.append(tail)
    return file, angles, tail
file, angles, tail = truc_pour_tail_data(0)
#fig, ax = plt.subplots(figsize=(15, 5))
ymax = np.max(np.abs(tail.curvature))
plt.plot(tail.curvature, color='black')
print(len(tail.curvature))
plt.fill_between(np.arange(len(tail.curvature)), -100 * ymax, 200 * tail.events - 100, color='red', alpha=0.25)
plt.ylim([-1.1 * ymax, 1.1 * ymax])
plt.title(file)
plt.plot(np.arange(len(tail.curvature)), np.zeros(len(tail.curvature)), c='y', linewidth=0.7)
print(file)
# plt.xlim(94900,95500)
# plt.show()



# Remplacez 'votre_fichier.xlsx' par le chemin de votre fichier Excel
fichier_excel = ("C:/Users/bedba/OneDrive/Git/test/projet_M-L/SegmentationManuelle_ML.xlsx")

df = pd.read_excel(fichier_excel)

# Supprimer les lignes avec des valeurs NaN dans la deuxième colonne
df = df.dropna(subset=[df.columns[1]])


idx_signal_train = list(df[df.columns[0]])
binary_matrix = np.zeros((38, 625000), dtype=int)

dict_vide = {}# print(df.columns[0][2])
for i, row in df.iterrows():
    liste = []
    # Parcourir les éléments de la ligne (colonnes)
    for col, value in row.iteritems():
        # Vérifier si la valeur est NaN
        if pd.isna(value):
            pass
        else:
            # Vérifier si la valeur est un entier
            if isinstance(value, int):
                pass
            else:
                # Convertir la chaîne '[a, b]' en une liste d'entiers
                try:
                    liste_entiers = ast.literal_eval(value)
                    if isinstance(liste_entiers, list) and all(isinstance(x, int) for x in liste_entiers):
                        # print(liste_entiers)
                        liste += [liste_entiers]
                    else:
                        print(f"Colonne: {col}, Valeur: {value} (non convertie en liste d'entiers valide)")
                except (ValueError, SyntaxError):
                    print(f"Colonne: {col}, Valeur: {value} (non convertie en liste d'entiers)")
    dict_vide[row[0]] = liste


def comb(liste):
    li = []
    for duo in liste:
        print(duo)
        li += list(np.arange(duo[0], duo[1], 1))
    return li
    
def code_cool(liste):
    bin = np.zeros(625000)
    bin[comb(liste)] = 1
    return bin

for keys in dict_vide.keys():
    binary_matrix[keys] = np.array(code_cool(dict_vide[keys]))
plt.plot(np.arange(0,625000,1), 15*np.array(code_cool(dict_vide[0])))
plt.show()
print(binary_matrix)