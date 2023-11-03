import numpy as np

PATH_EVAL="./data/Le_comte_de_Monte_Cristo.100.sim"
PATH_EMBEDDING="./embeddings/embeddings_ameliore_train.txt"                        #embeddings_amelioration_yanis.txt

def recuperation_eval(path_eval=PATH_EVAL):
    """
    récupère les plongements m, m+ et m- issus du fichier d'évaluation
    """
    colonne1 = []  # représentant le plongement m
    colonne2 = []  # représentant le plongement m+
    colonne3 = []  # représentant le plongement m-
    with open(path_eval, 'r') as fichier:
        for ligne in fichier:
            # Diviser la ligne en colonnes en utilisant l'espace comme séparateur
            mots = ligne.split()
            # Ajouter chaque mot à la colonne respective
            colonne1.append(mots[0])
            colonne2.append(mots[1])
            colonne3.append(mots[2])
    # print("Colonne 1 :", colonne1)
    # print("Colonne 2 :", colonne2)
    # print("Colonne 3 :", colonne3)
    return colonne1, colonne2, colonne3


def recuperation_embedding(path_embedding=PATH_EMBEDDING):
    """
    récupère les embeddings des mots correspondants issus du fichier txt dont le chemin est en variable
    """
    # Définition des listes pour stocker les données
    mots = []
    embeddings = []

    with open(path_embedding, 'r') as fichier:
        for ligne in fichier:
            if len(mots) == 1:
                # Diviser la ligne en deux parties : mot et embedding
                partie = ligne.split()
                # Pour la deuxième ligne, "espace" comme mot et tout le reste de la ligne comme embedding
                mot = " "
                #print("PARTIE:",partie)
                # Convertir l'embedding en liste de nombres
                embedding = [float(val) for val in partie]
                # Ajouter le mot et l'embedding aux listes respectives
                mots.append(mot)
                embeddings.append(embedding)

            else:
                # Diviser la ligne en deux parties : mot et embedding
                partie = ligne.split(maxsplit=1)
                mot = partie[0]  # Le premier élément est le mot

                # Le reste de la ligne est l'embedding
                embedding_str = partie[1].rstrip()
                
                # Convertir l'embedding en liste de nombres
                embedding = [float(val) for val in embedding_str.split()]
                # Ajouter le mot et l'embedding aux listes respectives
                mots.append(mot)
                embeddings.append(embedding)

    #print("Mots :", mots[1])
    #print("Embeddings :", embeddings[1])
    #print("len_embedd:",len(embeddings[1]))
    return mots, embeddings

def get_embedding(mot, mots, embeddings):
    """
    Obtenir l'embedding précit d'un mot
    """
    try:
        ind=mots.index(mot)
        return(embeddings[ind])
    except:
        print("Word out of vocabulary !")
        return(False)


def cosine_similarity(vector1, vector2):
    """
    Calcule la similarité cosinus entre deux vecteurs
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    if norm_vector1 == 0 or norm_vector2 == 0:
        # Éviter une division par zéro si l'une des normes est nulle
        return 0.0
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def evaluate(colonne1,colonne2,colonne3):
    """
    évalue la qualité des embeddings issus d'un modèle particulier en utilisant le fichier d'embeddings qu'il a créé 
    en se basant sur les 3 listes de plongements m, m+ et m- fournis.
    """
    good = 0
    for k in range(len(colonne1)):
        try:
            a, b, c = get_embedding(colonne1[k],mots,embeddings), get_embedding(colonne2[k],mots,embeddings), get_embedding(colonne3[k],mots,embeddings)
            sim1, sim2 = cosine_similarity(a,b), cosine_similarity(a,c)
            if sim1>sim2:
                good+=1
        except:
            print("Word out of vocabulary, pas comptabilisé")
    return(100*good/len(colonne1))


if __name__ == "__main__":
    colonne1, colonne2, colonne3 = recuperation_eval()
    mots, embeddings = recuperation_embedding()
    #print("embeddding found", get_embedding('homme', mots, embeddings))
    print("ACCURACY: ", evaluate(colonne1, colonne2, colonne3),"%")



