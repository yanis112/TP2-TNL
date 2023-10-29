import numpy as np
from nltk.tokenize import word_tokenize  # Assurez-vous d'avoir installé NLTK via pip
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH="C:/Users/Yanis/Documents/Cours Centrale Marseille/NLP/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.test.tok"
PATH="C:/Users/Yanis/Documents/Cours Centrale Marseille/NLP/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.train.tok"

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

texte=openfile(PATH)


#print(texte)

# Fonction pour créer la matrice de co-occurrence
def create_co_occurrence_matrix(text, window_size):
    # Tokenisation du texte en mots
    words = text
    # Création d'un vocabulaire unique à partir des mots dans le texte
    vocab = list(set(words))
    
    # Initialisation de la matrice de co-occurrence
    co_occurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=int)
    
    # Remplissage de la matrice de co-occurrence
    for i, target_word in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:  # Assurez-vous que le mot cible ne soit pas lui-même
                context_word = words[j]
                if context_word in vocab and target_word in vocab:
                    co_occurrence_matrix[vocab.index(target_word)][vocab.index(context_word)] += 1
    
    return co_occurrence_matrix, vocab


# Taille de la fenêtre glissante
window_size = 3


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def proba_positive(m, c):
    dot_product = np.dot(m, c)
    return sigmoid(dot_product)

def proba_negative(m, c):
    dot_product = np.dot(m, c)
    return 1 - sigmoid(dot_product)

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
    p_neg_sum = np.sum(np.log(sigmoid(-np.dot(cneg, m))) for cneg in cneg_list)
    
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


# Fonction pour créer des embeddings aléatoires pour chaque mot unique
def create_word_embeddings(text, embedding_size):
    # Créer un vocabulaire unique à partir du texte
    vocab = list(set(text))
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
            # Choisissez aléatoirement un mot contexte dans la fenêtre centrée
            window_start = max(0, target_word_index - window_size)
            window_end = min(len(text), target_word_index + window_size + 1)
            context_word_index = np.random.randint(window_start, window_end)
            context_word = text[context_word_index]
            
            # Sélectionnez 4 mots négatifs au hasard
            #negative_samples = [word for word in text if word != target_word][:4] #PAs con c'est toujours les mêmes !!
            negative_samples = np.random.choice(text, size=neg_number, replace=False)
            negative_samples = [word for word in negative_samples if word != target_word]
            
            # Calculez les gradients en utilisant les fonctions de dérivées
            cpos = word_embeddings[context_word]
            cneg_list = [word_embeddings[neg_word] for neg_word in negative_samples]
            
            grad_m = compute_grad_m(word_embeddings[target_word], cpos, cneg_list)
            grad_cpos = compute_grad_cpos(word_embeddings[target_word], cpos)
            grad_cneg_list = [compute_grad_cneg(word_embeddings[target_word], cneg) for cneg in cneg_list]
            
            # Mettez à jour les embeddings en utilisant la descente de gradient stochastique
            # Taux d'apprentissage (ajustez selon vos besoins)
            word_embeddings[target_word] -= learning_rate * grad_m
            word_embeddings[context_word] -= learning_rate * grad_cpos
            for i, neg_word in enumerate(negative_samples):
                word_embeddings[neg_word] -= learning_rate * grad_cneg_list[i]
            # Calculez et enregistrez la perte à la fin de chaque itération
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
embedding_size = 100 #avec 50 on a obtenu 53% accuracy # Taille des embeddings (ajustez selon vos besoins)
iternb = 5  # Nombre d'itérations (ajustez selon vos besoins)
window_size = 5  # Taille de la fenêtre centrée (ajustez selon vos besoins)
learning_rate = 0.1  # Taux d'apprentissage (ajustez selon vos besoins)
neg_number=10 # nombre de samples out


# Entraînement des embeddings
trained_word_embeddings,list_loss = train_word_embeddings(texte, embedding_size, iternb, window_size,learning_rate,neg_number)
save_word_embeddings_to_file(trained_word_embeddings,'embeddings.txt')
print("embeddings saved in file !!")
plot_loss_curve(list_loss)