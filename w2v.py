import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

## Paramètres

n = 100  # Dimension des embeddings  
L = 5  # Taille de la fenêtre centrée  
eta = 0.1  # Taux d'apprentissage  
k = 10  # Le nombre de contextes négatifs pour un contexte positif
e = 1 # Nombre d'itérations  

# minc = 2


PATH_test="./data/Le_comte_de_Monte_Cristo.test.tok"
PATH_train="./data/Le_comte_de_Monte_Cristo.train.tok"
PATH_EMBEDDING="./embeddings/"


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
    ''' 
    Fonction sigmoïde
    '''
    return 1 / (1 + np.exp(-x))


def compute_loss(m, cpos, cneg_list):
    """
    Calcule la valeur de la fonction de perte définie par la formule donnée.
    """
    p_pos = sigmoid(np.dot(cpos, m))
    p_neg_sum = np.sum(np.fromiter((np.log(sigmoid(-np.dot(cneg, m))) for cneg in cneg_list), dtype=float))
    loss = -np.log(p_pos)-p_neg_sum
    return loss


def compute_grad_cpos(m, cpos):
    """
    Calcule le gradient de la perte par rapport au vecteur cpos.
    """
    grad_cpos = (sigmoid(np.dot(cpos, m)) - 1) * m
    return grad_cpos


def compute_grad_cneg(m, cneg):
    """
    Calcule le gradient de la perte par rapport à un vecteur cneg.
    """
    grad_cneg = sigmoid(np.dot(cneg, m)) * m
    return grad_cneg


def compute_grad_m(m, cpos, cneg_list):
    """
    Calcule le gradient de la perte par rapport au vecteur m.
    """
    grad_m = (sigmoid(np.dot(cpos, m)) - 1) * cpos
    for cneg in cneg_list:
        grad_m += sigmoid(np.dot(cneg, m)) * cneg
    return grad_m


def create_word_embeddings(text, n):
    """
    Créer des embeddings aléatoires pour chaque mot unique
    """
    # Créer un vocabulaire unique à partir du texte
    vocab = list(set(text))
    # Initialiser un dictionnaire pour stocker les embeddings de chaque mot
    word_embeddings = {}
    # Générer des embeddings aléatoires pour chaque mot du vocabulaire
    for word in vocab:
        word_embeddings[word] = np.random.rand(n)
    return word_embeddings


def train_word_embeddings(text, n, e, L, eta, k):
    """
    Entraînement des embeddings
    """
    # Créer des embeddings initiaux pour chaque mot du texte
    word_embeddings = create_word_embeddings(text, n)
    print("Embeddings initiaux créés")
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
            
            # Sélectionner k mots négatifs au hasard
            negative_samples = np.random.choice(text, size=k, replace=False)
            negative_samples = [word for word in negative_samples if word != target_word]
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


def save_word_embeddings_to_file(embeddings, filename):
    """
    Enregistrer les embeddings dans un fichier texte au format demandé
    """
    with open(filename, 'w', encoding='utf8') as f:
        # Écrire le nombre de plongements et la dimension
        f.write(f"{len(embeddings)} {len(embeddings[next(iter(embeddings))])}\n")
        # Écrire chaque mot et son embedding
        for word, embedding in embeddings.items():
            embedding_str = ' '.join(str(value) for value in embedding)
            f.write(f"{word} {embedding_str}\n")


def plot_loss_curve(loss_history):
    """
    Tracer la courbe d'apprentissage
    """
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel('Itération')
    plt.ylabel('Perte')
    plt.title('Courbe d\'apprentissage')
    plt.show()


if __name__ == '__main__':
    # On définit le texte et le vocabulaire
    texte=openfile(PATH_train)  # liste de mots du texte
    vocab=list(set(texte))  # liste des mots uniques du texte

    # On lance l'entraînement
    trained_word_embeddings,list_loss = train_word_embeddings(texte, n, e, L, eta, k)
    save_word_embeddings_to_file(PATH_EMBEDDING,'embeddings_train.txt')
    print("Embeddings enregistrés !")
    plot_loss_curve(list_loss)