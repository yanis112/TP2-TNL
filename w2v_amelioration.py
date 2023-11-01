from collections import Counter
import math
import numpy as np
import os
from tqdm import tqdm
import random

from w2v import PATH_test, PATH_train, PATH_EMBEDDING
from w2v import openfile, compute_loss, compute_grad_m, compute_grad_cneg, compute_grad_cpos, create_word_embeddings, save_word_embeddings_to_file, plot_loss_curve

## Paramètres

n = 100  # Dimension des embeddings
L = 5  # Taille des contextes gauche et droit  
eta = 0.1  # Taux d'apprentissage  
k = 10  # Le nombre de contextes négatifs par contexte positif
e = 5 # Nombre d'itérations  

seuil = 1e-5


def calculate_subsampling_probabilities(text, seuil=seuil):
    '''
    renvoie le dictionnaire des probabilités de délétion dans le cas du subsampling en prenant un seuil en compte
    '''
    word_frequencies = Counter(text)  # Compte les occurrences de chaque mot
    total_words = len(text)  # Calcule le nombre total de mots dans le corpus
    # Calcule les fréquences relatives de chaque mot
    relative_frequencies = {word: frequency**(3/4) / total_words for word, frequency in word_frequencies.items()}
    # Calculez les probabilités de suppression pour chaque mot
    deletion_probabilities = {word: 1 - math.sqrt(seuil / relative_frequency) for word, relative_frequency in relative_frequencies.items()}
    return deletion_probabilities


def subsampling(text):
    '''
    Fonction qui réalise le subsampling du texte: on supprime une partie des mots qui ont des occurences trop nombreuses EX: "le" car ils ne contiennent pas d'info
    la probabilité de supprimer le mot dépend d'un seuil et de sa fréquence dans le corpus.
    '''
    dictio_suppr=calculate_subsampling_probabilities(text, 10e-5)
    new_text=[]
    for k in text:
        probkeep= 1-dictio_suppr[k]
        if random.random()<probkeep:
            new_text.append(k)
    return(new_text)


def negative_sampling(dictionnaire_probabilites, k):
    ''' 
    Fonction qui renvoie une liste de k mots tirés aléatoirement selon les probabilités du dictionnaire
    '''
    if k <= 0:
        raise ValueError("La valeur de k doit être supérieure à zéro.")
    mots_choisis = []
    for _ in range(k):
        # Générer un nombre aléatoire pondéré par les probabilités du dictionnaire
        choix = random.choices(list(dictionnaire_probabilites.keys()), list(dictionnaire_probabilites.values()))[0]
        mots_choisis.append(choix)
    return mots_choisis

def train_word_embeddings(text, n, e, L, eta, k):
    """
    Entraînement des embeddings
    """
    word_embeddings = create_word_embeddings(text, n)  # Créer des embeddings initiaux pour chaque mot du texte
    print("Embeddings initiaux créés")
    text=subsampling(text)  #Nous appliquons le subsampling du texte pour éliminer les mots trop redondants
    dict_frequencies=calculate_subsampling_probabilities(text)  # Dictionnaire des probabilités de délétion pour le subsampling
    loss_history = []  # Initialiser une liste pour enregistrer la perte à chaque itération

    # Boucle d'entraînement sur un nombre d'itérations e
    for iteration in range(e):
        print(f"Itération {iteration + 1}/{e}")
        # Parcourir chaque mot dans le texte
        for target_word_index, target_word in enumerate(tqdm(text)):
            # Choisir aléatoirement un mot contexte dans la fenêtre centrée
            window_start = max(0, target_word_index - L)
            window_end = min(len(text), target_word_index + L + 1)
            context_word_index = np.random.randint(window_start, window_end)
            while target_word_index==context_word_index:
                context_word_index = np.random.randint(window_start, window_end)
            context_word = text[context_word_index]

            # Sélection aléatoire des mots négatifs par le biais d'un échantillonnage négatif
            negative_samples=negative_sampling(dict_frequencies,k)

            # Calcul des gradients en utilisant les fonctions de dérivées
            cpos = word_embeddings[context_word]
            cneg_list = [word_embeddings[neg_word] for neg_word in negative_samples]
            grad_m = compute_grad_m(word_embeddings[target_word], cpos, cneg_list)
            grad_cpos = compute_grad_cpos(word_embeddings[target_word], cpos)
            grad_cneg_list = [compute_grad_cneg(word_embeddings[target_word], cneg) for cneg in cneg_list]
            
            # Mettre à jour les embeddings en utilisant la descente de gradient stochastique
            word_embeddings[target_word] -= eta * grad_m
            word_embeddings[context_word] -= eta * grad_cpos
            for i, neg_word in enumerate(negative_samples):
                word_embeddings[neg_word] -= eta * grad_cneg_list[i]
            current_loss = compute_loss(word_embeddings[target_word], cpos, cneg_list)
            loss_history.append(current_loss)
    
    return word_embeddings, loss_history


if __name__ == '__main__':
    #On définit le texte et le vocabulaire
    texte=openfile(PATH_train)
    vocab=list(set(texte))

    # On applique le subsampling
    texte=subsampling(texte)

    # On lance l'entraînement
    trained_word_embeddings, list_loss = train_word_embeddings(texte, n, e, L, eta, k)
    save_word_embeddings_to_file(trained_word_embeddings, os.path.join(PATH_EMBEDDING, 'embeddings_ameliore_train.txt'))
    print("Embeddings enregistrés !")
    plot_loss_curve(list_loss)