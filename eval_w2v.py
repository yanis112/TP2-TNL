
PATH="C:/Users/Yanis/Documents/Cours Centrale Marseille/NLP/tlnl2/tp/Le_comte_de_Monte_Cristo.100.sim"
PATH_EMBEDDING="/." #"C:/Users/Yanis/embeddings.txt"
PATH_EMBEDDING="C:/Users/Yanis/Documents/Cours Centrale Marseille/NLP/tlnl3/embeddings.txt"
# Définition d'une liste pour chaque colonne
colonne1 = []
colonne2 = []
colonne3 = []

# Ouvrir le fichier texte en lecture
with open(PATH, 'r') as fichier:
    # Lire chaque ligne du fichier
    for ligne in fichier:
        # Diviser la ligne en colonnes en utilisant l'espace comme séparateur
        mots = ligne.split()
        # Ajouter chaque mot à la colonne respective
        colonne1.append(mots[0])
        colonne2.append(mots[1])
        colonne3.append(mots[2])

# Afficher les listes résultantes
print("Colonne 1 :", colonne1)
print("Colonne 2 :", colonne2)
print("Colonne 3 :", colonne3)

# Définition des listes pour stocker les données
mots = []
embeddings = []

# Ouvrir le fichier d'embeddings en lecture
with open(PATH_EMBEDDING, 'r') as fichier:
    for ligne in fichier:
       
        if len(mots) == 1:
             # Diviser la ligne en deux parties : mot et embedding
            partie = ligne.split()
            # Pour la deuxième ligne, utilisez "espace" comme mot et tout le reste de la ligne comme embedding
            mot = " "
            print("PARTIE:",partie)
            # Convertir l'embedding en liste de nombres
            embedding = [float(val) for val in partie]
            # Ajouter le mot et l'embedding aux listes respectives
            mots.append(mot)
            embeddings.append(embedding)

        else:
                # Diviser la ligne en deux parties : mot et embedding
            partie = ligne.split(maxsplit=1)
            mot = partie[0]  # Le premier élément est le mot

            # Le reste de la ligne est l'embedding (sans le '\n')
            embedding_str = partie[1].rstrip()
            
            # Convertir l'embedding en liste de nombres
            embedding = [float(val) for val in embedding_str.split()]
            # Ajouter le mot et l'embedding aux listes respectives
            mots.append(mot)
            embeddings.append(embedding)

#print("Mots :", mots[1])
#print("Embeddings :", embeddings[1])
#print("len_embedd:",len(embeddings[1]))

def get_embedding(mot,mots,embeddings):
    try:
        ind=mots.index(mot)
        return(embeddings[ind])
    except:
        print("Word out of vocabulary, putting embedding 0")
        return([0 for k in range(50)])


import numpy as np

def cosine_similarity(vector1, vector2):
    """
    Calcule la similarité cosinus entre deux vecteurs.

    Args:
    vector1 (numpy.ndarray): Le premier vecteur.
    vector2 (numpy.ndarray): Le deuxième vecteur.

    Returns:
    float: La similarité cosinus entre les deux vecteurs.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    if norm_vector1 == 0 or norm_vector2 == 0:
        # Éviter une division par zéro si l'une des normes est nulle
        return 0.0
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

    
print("embeddding found",get_embedding('homme',mots,embeddings))

def evaluate(colonne1,colonne2,colonne3):

    good=0
    for k in range(len(colonne1)):
        a,b,c=get_embedding(colonne1[k],mots,embeddings),get_embedding(colonne2[k],mots,embeddings),get_embedding(colonne3[k],mots,embeddings)
        sim1,sim2=cosine_similarity(a,b),cosine_similarity(a,c)
        if sim1>sim2:
            good+=1
    return(100*good/len(colonne1))

print("ACCURACY: ", evaluate(colonne1,colonne2,colonne3),"%")





