from collections import Counter
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import random

from w2v import PATH_test, PATH_train

def openfile(file: str) -> list[str]:
    """
    prend un chemin ou une liste de chemain vers un ou des fichier(s) texte(s) 
    et renvoie la liste de mots que contients ce(s) texte(s)
    """
    text = []
    print(file)
    with open(file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            if line[-1] == '\n':
                line = line[:-2]
            line = line.split(' ')

            text += line[2 : -2]

    return text

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def proba_positive(m, c):
    dot_product = np.dot(m, c)
    return sigmoid(dot_product)

def proba_negative(m, c):
    dot_product = np.dot(m, c)
    return 1 - sigmoid(dot_product)

def calculate_deletion_probabilities(text, seuil):
    '''
    renvoie le dictionnaire des probabilité de délétion
    '''
    # Tokenisation : Divisez le texte en mots
     # Vous pouvez utiliser une tokenisation plus avancée si nécessaire
    
    # Utilisez Counter pour compter les occurrences de chaque mot
    word_frequencies = Counter(text)
    
    # Calculez le nombre total de mots dans le corpus
    total_words = len(text)
    
    # Calculez les fréquences relatives de chaque mot
    relative_frequencies = {word: frequency / total_words for word, frequency in word_frequencies.items()}
    
    # Calculez les probabilités de suppression pour chaque mot
    deletion_probabilities = {word: 1 - math.sqrt(seuil / relative_frequency) for word, relative_frequency in relative_frequencies.items()}
    
    return deletion_probabilities


def subsampling(text):
    '''
    Fonction qui réalise le subsampling du texte: on supprime une partie des mots qui ont des occurences trop nombreuses EX: "le" car ils ne contiennent pas d'info
    la probabilité de supprimer le mot dépend d'un seuil et de sa fréquence dans le corpus.
    '''
    dictio_suppr=calculate_deletion_probabilities(text, 10e-5)
    new_text=[]
    for k in text:
        probkeep= 1-dictio_suppr[k]
        if random.random()<probkeep:
            new_text.append(k)
    return(new_text)


# Fonction pour calculer la perte
def compute_loss(m, cpos, cneg_list):
    """
    Calcule la valeur de la fonction de perte définie par la formule donnée.

    Args:
        m (numpy.ndarray): Le vecteur m.
        cpos (numpy.ndarray): Le vecteur cpos.
        cneg_list (list): Une liste de vecteurs cneg.

    Returns:
        float: La valeur de la perte.
    """
    p_pos = sigmoid(np.dot(cpos, m))
    p_neg_sum = np.sum(np.fromiter((np.log(sigmoid(-np.dot(cneg, m))) for cneg in cneg_list), dtype=float))

    
    loss = -np.log(p_pos)-p_neg_sum
    return loss

# Fonction pour calculer le gradient par rapport à cpos
def compute_grad_cpos(m, cpos):
    """
    Calcule le gradient de la perte par rapport au vecteur cpos.

    Args:
        m (numpy.ndarray): Le vecteur m.
        cpos (numpy.ndarray): Le vecteur cpos.

    Returns:
        numpy.ndarray: Le gradient par rapport à cpos.
    """
    grad_cpos = (sigmoid(np.dot(cpos, m)) - 1) * m
    return grad_cpos


# Fonction pour calculer le gradient par rapport à cneg
def compute_grad_cneg(m, cneg):
    """
    Calcule le gradient de la perte par rapport à un vecteur cneg.

    Args:
        m (numpy.ndarray): Le vecteur m.
        cneg (numpy.ndarray): Le vecteur cneg.

    Returns:
        numpy.ndarray: Le gradient par rapport à cneg.
    """
    grad_cneg = sigmoid(np.dot(cneg, m)) * m
    return grad_cneg


# Fonction pour calculer le gradient par rapport à m
def compute_grad_m(m, cpos, cneg_list):
    """
    Calcule le gradient de la perte par rapport au vecteur m.

    Args:
        m (numpy.ndarray): Le vecteur m.
        cpos (numpy.ndarray): Le vecteur cpos.
        cneg_list (list): Une liste de vecteurs cneg.

    Returns:
        numpy.ndarray: Le gradient par rapport à m.
    """
    grad_m = (sigmoid(np.dot(cpos, m)) - 1) * cpos
    for cneg in cneg_list:
        grad_m += sigmoid(np.dot(cneg, m)) * cneg
    return grad_m


def select_negative_samples(context_word, vocab, word_embeddings, threshold, num_negatives):
    negative_samples = []
    count_negatives = 0

    vocabulary=vocab
    # Mélanger le vocabulaire pour parcourir aléatoirement
    random.shuffle(vocabulary)
    
    for word in vocabulary:
        # Vérifier s'il y a assez de mots négatifs
        if count_negatives >= num_negatives:
            break
        
        # Calcul de la similarité entre le mot cible et le mot potentiellement négatif
        similarity = cosine_similarity(word_embeddings[word].reshape(1,-1),word_embeddings[context_word].reshape(1,-1))[0][0]

        #print(similarity)
        # Si la similarité est inférieure au seuil, ajouter le mot potentiellement négatif
        if similarity > threshold:
            negative_samples.append(word)
            count_negatives += 1 
                
    return negative_samples

# Fonction pour créer des embeddings aléatoires pour chaque mot unique
def create_word_embeddings(text, embedding_size):
    # Initialiser un dictionnaire pour stocker les embeddings de chaque mot
    word_embeddings = {}
    
    # Générer des embeddings aléatoires pour chaque mot du vocabulaire
    for word in vocab:
        word_embeddings[word] = np.random.rand(embedding_size)
    
    return word_embeddings

# Fonction pour l'entraînement des embeddings
def train_word_embeddings(text, embedding_size, k, window_size,learning_rate,neg_number):
    # Créer des embeddings initiaux pour chaque mot du texte
    word_embeddings = create_word_embeddings(text, embedding_size)
    print("embeddings crées")
     # Initialiser une liste pour enregistrer la perte à chaque itération
    loss_history = []

    
    # Boucle d'entraînement sur un nombre d'itérations (k)
    for iteration in range(k):
        print(f"Itération {iteration + 1}/{k}")
        # Parcourir chaque mot dans le texte
        for target_word_index, target_word in enumerate(tqdm(text)):
            # Choisir aléatoirement un mot contexte dans la fenêtre centrée
            window_start = max(0, target_word_index - window_size)
            window_end = min(len(text), target_word_index + window_size + 1)
            context_word_index = np.random.randint(window_start, window_end)
            while target_word_index==context_word_index:
                context_word_index = np.random.randint(window_start, window_end)
            context_word = text[context_word_index]
            
            # Sélectionner 4 mots négatifs au hasard
            negative_samples = np.random.choice(text, size=neg_number, replace=False)
            negative_samples = [word for word in negative_samples if word != target_word]
            cpos = word_embeddings[context_word]
            cneg_list = [word_embeddings[neg_word] for neg_word in negative_samples]
            
            grad_m = compute_grad_m(word_embeddings[target_word], cpos, cneg_list)
            grad_cpos = compute_grad_cpos(word_embeddings[target_word], cpos)
            grad_cneg_list = [compute_grad_cneg(word_embeddings[target_word], cneg) for cneg in cneg_list]
            
            # Mettre à jour les embeddings en utilisant la descente de gradient stochastique
            word_embeddings[target_word] -= learning_rate * grad_m
            word_embeddings[context_word] -= learning_rate * grad_cpos
            for i, neg_word in enumerate(negative_samples):
                word_embeddings[neg_word] -= learning_rate * grad_cneg_list[i]
            current_loss = compute_loss(word_embeddings[target_word], cpos, cneg_list)
            loss_history.append(current_loss)
    
    
    return word_embeddings, loss_history


# Fonction pour enregistrer les embeddings dans un fichier texte au format spécifié
def save_word_embeddings_to_file(embeddings, filename):
    with open(filename, 'w', encoding='utf8') as f:
        # Écrire le nombre de plongements et la dimension
        f.write(f"{len(embeddings)} {len(embeddings[next(iter(embeddings))])}\n")
        
        # Écrire chaque mot et son embedding
        for word, embedding in embeddings.items():
            embedding_str = ' '.join(str(value) for value in embedding)
            f.write(f"{word} {embedding_str}\n")


# Fonction pour tracer la courbe d'apprentissage
def plot_loss_curve(loss_history):
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel('Itération')
    plt.ylabel('Perte')
    plt.title('Courbe d\'apprentissage')
    plt.show()

# Paramètres
embedding_size = 100  # Taille des embeddings  
k = 5 # Nombre d'itérations  
window_size = 5  # Taille de la fenêtre centrée  
learning_rate = 0.1  # Taux d'apprentissage  
neg_number=10
window_size = 3  # Taille de la fenêtre glissante

if __name__ == '__main__':

    #On définit le texte et le vocabulaire
    texte=openfile(PATH_test)
    vocab=list(set(texte))

    #print(len(texte))
    #print(len(subsampling(texte)))
    #print(subsampling(texte))

    texte=subsampling(texte)

    trained_word_embeddings,list_loss = train_word_embeddings(texte, embedding_size, k, window_size,learning_rate,neg_number)
    #trained_word_embeddings,list_loss=train_word_embeddings_parallel(texte, embedding_size, k, window_size,learning_rate,neg_number)
    save_word_embeddings_to_file(trained_word_embeddings,'embeddings.txt')
    print("embeddings saved in file !!")
    plot_loss_curve(list_loss)