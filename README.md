---
title: "Compte rendu de projet - IA embarqué"
author: "Chloé Larroze, Yann Rosenblum"
---
## Compte rendu de projet - IA embarqué

Ce projet consiste à concevoir, entraîner et déployer un réseau de neurones (DNN) pour la maintenance prédictive, en utilisant le jeu de données AI4I 2020 Predictive Maintenance Dataset. L'objectif final est de déployer le modèle sur un microcontrôleur STM32L4R9 à l'aide de STM32Cube.AI.

Le projet couvrira les étapes suivantes :
- Prétraitement des données
- Conception et entraînement du modèle
- Évaluation des performances
- Conversion du modèle pour de l’embarqué
- Intégration pour exécution sur STM32L4R9

## Introduction

Avec l'émergence de l'industrie 4.0, le secteur manufacturier connaît une transformation sans précédent grâce aux nouvelles technologies comme l'IoT, l'intelligence artificielle et l'analyse des données.

<div align="center">
    <img src="./images/figure1.png" alt="Les neuf piliers de l’industrie 4.0" width="400px"/>
    <p><em>Figure 1 : Les neuf piliers de l’industrie 4.0</em></p>
</div>

Remplacer une machine entière coûte bien plus cher que de changer un simple composant, d'où l'intérêt d'installer des capteurs pour surveiller leur état en temps réel. En collectant et analysant ces données, les entreprises peuvent optimiser la maintenance, réduire les coûts et éviter les arrêts imprévus. À cet effet, nous nous demanderons 
> Comment concevoir et déployer un modèle de maintenance prédictive efficace sur un microcontrôleur à ressources limitées, tout en garantissant des performances optimales pour la détection des pannes ?

Pour répondre à cette attente, il faudra concevoir un modèle de maintenance prédictive capable d’analyser les données tout en étant optimisé pour une exécution sur un microcontrôleur à ressources limitées. Cela impliquera de trouver un équilibre entre la précision du modèle et sa consommation en mémoire ainsi qu’en puissance de calcul. 

## Utilisation 
1. **Entraîner le modèle** en exécutant le notebook sur Google Colab.
2. **Exporter et convertir le modèle** en format `.h5`.
3. **Flasher et exécuter sur la carte STM32** via STM32CubeIDE.
4. **Communiquer en UART** avec la carte via l'éxecution du programme python `ports.py` 

## Installation et Prérequis

Dans le cadre de ce projet, nous avons utilisé un ensemble de bibliothèques et d’outils pour la conception, l'entraînement et le déploiement du modèle de maintenance prédictive sur un microcontrôleur STM32.

### Environnement de travail

- **Google Colab (Jupyter Notebook)** : utilisé pour le développement, l'entraînement et l'analyse du modèle en Python.
- **STM32Cube IDE** : IDE pour le déploiement sur microcontrôleur STM32.
- **X-CUBE-AI** : bibliothèque fournie par STMicroelectronics permettant de convertir et d’exécuter un modèle de deep learning sur un STM32.

Après l'entraînement du modèle sous Google Colab, nous avons converti le modèle en format `h5` (optimisé pour l’embarqué) avec X-CUBE-AI, puis importé le modèle dans STM32CubeIDE pour le déploiement sur le STM32L4R9.

### Compatibilité avec TensorFlow
Nous avons utilisé TensorFlow 2.12, car les versions plus récentes causaient un problème inexpliqué de gestion du batch size lors de la conversion et du déploiement dans STM32Cube.AI.


## Dataset utilisé

### Présentation

Le dataset **AI4I 2020 Predictive Maintenance Dataset** contient 10 000 entrées et est structuré en 14 colonnes, représentant différentes caractéristiques des machines ainsi que des indicateurs de panne. Chaque instance décrit donc l’état de fonctionnement d’une machine et indique si une panne a eu lieu. Ces pannes sont classifiées en 5 types, trouvables dans la dernière colonne du tableau. On retrouve ainsi les entrées suivantes : 


<div align="center">
    <img src="./images/figure2.png" alt="Entrées du dataset" width="500px"/>
    <p><em>Figure 2 : Entrées du dataset</em></p>
</div>

De plus, le dataset ne contient pas de valeurs manquantes, de valeurs dupliquées, ce qui évitera les biais liés à la redondance des données. Enfin, les différentes pannes sont bien catégorisées. Nous verrons cependant dans les parties ultérieures qu’un nettoyage des données sera tout de même nécessaire. À présent, explorons un peu notre jeu de données afin de nous familiariser avec. 

### Anomalies du dataset

Bien que cela n’influence en rien notre étude, nous pouvons nous représenter la proportion de types de machines dans le graphique ci-dessous :  

<div align="center">
    <img src="./images/figure3.png" alt="Proportion de machines par type" width="200px"/>
    <p><em>Figure 3 : Proportion de machines par type</em></p>
</div>

Une première analyse des labels montre que le dataset est fortement déséquilibré, avec seulement environ 3.5% des machines en panne. Ce déséquilibre nous posera problème lors de l'entraînement du modèle, car il risque d’être biaisé en faveur des classes majoritaires.

On peut ensuite se représenter la distribution des pannes par types : 
<div align="center">
    <img src="./images/figure4.png" alt="Proportion de machines par type" width="400px"/>
    <p><em>Figure 4 : Distribution des pannes selon leur type</em></p>
</div>
Deux problèmes principaux émergent alors de ce graphe : 

- Tout d'abord, nous remarquons que 9 machines ont rencontré une panne sans explication apparente, ce qui représente environ 3% du total des pannes. **Concrètement, le Machine failure est à 1 tandis que tous les types d’erreurs restent à 0**. 
- Ensuite, nous constatons une perte de 10 occurrences de panne par rapport aux données initiales (le graphique précédent indiquait 19 pannes aléatoires de type RNF). Cela suggère que **lorsque la panne est aléatoire (RNF), la variable Machine Failure n’est pas systématiquement renseignée à 1**.

Heureusement, les pannes RNF apparaissent dans un nombre limité d'observations et, par définition, leur caractère aléatoire les rend imprévisibles. Il pourrait donc être pertinent de supprimer ces lignes lors de la préparation finale des données. Cette même logique s'applique aux 9 cas de pannes RNF non étiquetées comme des échecs de machine.


## Pipeline global

Toutes les réponses détaillées et les implémentations précises sont disponibles dans le Google Colab (Jupyter Notebook) associé, où l’ensemble des codes et analyses sont regroupés.

### Prétraitement des données

#### Suppression 
Comme précédemment évoqué, il nous faudra dans un premier temps  nettoyer notre ensemble de données. Les deux problèmes rencontrés ( RNF et problème de Machine Failure) seront gérés dans notre code par les lignes de code suivantes :  


```python
# Suppression des observations où RNF=1 mais Machine failure=0
idx_RNF = data.loc[(data["RNF"] == 1) & (data["Machine failure"] == 0)].index
data.drop(index=idx_RNF, inplace=True)

# Suppression des observations où Machine failure=1 mais aucune panne spécifique n’est renseignée
idx_unknown_failure = data.loc[(data["Machine failure"] == 1) & (data[failure_types].sum(axis=1) == 0)].index
data.drop(index=idx_unknown_failure, inplace=True)
```

#### Normalisation
Un autre point à prendre en compte est la variation des échelles entre nos différentes variables d’entrée. Les températures sont exprimées en kelvins, le couple en newtons-mètres et la vitesse de rotation en tours par minute, ce qui entraîne des ordres de grandeur très différents. Or, la majorité des algorithmes d’apprentissage  sont sensibles à ces disparités et risquent de privilégier les variables ayant les valeurs les plus élevées, faussant ainsi l’entraînement du modèle. Pour éviter cet effet de dominance et garantir la "stabilité" de nos loss et accuracy, nous normaliserons les données d'entrée, géré par les lignes : 

```python 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Important : utiliser les mêmes paramètres que le train set
```


### Entraînement du modèle

À présent, notre ensemble de données est prêt à être exploité. Le résultat que nous attendons en sortie de notre DNN est de savoir en premier lieu s' il y a une panne ou non, et de détailler le type de panne dans ce cas. Nous allons donc créer une colonne intitulée “No error”, qui prendra ainsi en compte l’ensemble des cas où la machine n’a pas eu de problème. Cela sera géré dans le code par les lignes :  

```python 
Y = data[failure_types].copy()
no_error_condition = (Y == 0).all(axis=1) #no error condition to handle the case where no failure happens
Y = np.where(no_error_condition, 'No Error', Y.idxmax(axis=1))
Y = pd.get_dummies(Y) #we need to convert it to one-hot
```

Ainsi, nos données de sorties contiendront  les labels "No Error" ou bien un des quatre types de pannes. Nous opterons pour l'architecture du DNN suivante : 

<div align="center">
    <img src="./images/model.png" alt="Proportions des classes avant et après SMOTE" width="200px"/>
    <p><em>Figure 5 : Schéma DNN</em></p>
</div>

 Maintenant que l'ensemble de données a été traité et notre modèle créé, nous pouvons le ré-équilibrer. En effet, comme nous l’avons observé dans la partie analyse du dataset, ce dernier est fortement déséquilibré, avec les machines rencontrant une panne ne représentant que 3% du dataset. Il faut donc rééquilibrer cela, sans quoi uniquement les machines fonctionnelles seront détectées, comme visible sur la matrice de confusion ci-dessous.

<div align="center">
    <img src="./images/figure5.png" alt="Proportions des classes avant et après SMOTE" width="300px"/>
    <p><em>Figure 6 :  Matrice de confusion DNN sans équilibrage</em></p>
</div>

Pour ce faire,ous avons combiné deux approches : le **SMOTE** et **l’undersampling**. Le SMOTE nous permettra de générer des échantillons synthétiques pour les classes minoritaires. Plsu concrètement, le SMOTE sélectionne un échantillon minoritaire, identifie ses k plus proches voisins, et génère un nouvel échantillon en interpolant entre ces points. 

En complément, nous avons appliqué l’undersampling à la classe majoritaire afin de réduire son influence et d’éviter un fort déséquilibre dans l’ensemble d’apprentissage. Contrairement à une approche uniquement basée sur de l’undersampling, qui peut entraîner une perte d’information et donc poser un risque de sous-apprentissage, la combinaison des deux méthodes permet d'équilibrer les données tout en préservant leur "diversité" en évitant un biais. 


<div align="center">
    <img src="./images/figure6.png" alt="Proportions des classes avant et après SMOTE" width="600px"/>
    <p><em>Figure 7 : Proportions des classes dans le dataset avant et après SMOTE</em></p>
</div>

### Performance du modèle
#### Performances globales
L'entraînement du modèle a donné les résultats suivants :

- **Performance sur le set TRAIN** : ACCURACY = `0.97`
- **Performance sur le set TEST** : ACCURACY = `0.91`

<div align="center">
    <img src="./images/figure7.png" alt="Courbes de loss et d’accuracy" width="600px"/>
    <p><em>Figure 8 : Courbes de loss et d’accuracy</em></p>
</div>

Malgré l’utilisation de techniques telles que la batch normalisation, L2 ou dropout pour éviter l’overfitting, ce sont malheuresement les meilleurs résultats que nous ayons pu obtenir.  

#### Métriques
Dans le cas de notre modèle, nous avons fait le choix d'une classification multi-classe car nous cherchons à identifier plusieurs types d'erreurs distinctes parmi les échantillons de données. Comme évoqué dans les parties précédentes, chaque instance appartient à une seule classe spécifique (HDF, No error, OSF, PWF, TWF), ce qui justifie l'approche multi-classes plutôt qu'une simple classification binaire.

À cet effet, nous utiliserons plusieurs métrique pour en évaluer la pertinence : 
- la **précision** : proportion d'instances prédites correctement parmi celles prédites dans une classe donnée (ex : une précision de 0,59 pour HDF signifie que 59 % des prédictions pour HDF étaient correctes);
- le **rappel** : capacité du modèle à identifier toutes les instances réelles d'une classe spécifique (ex : un rappel de 1,00 pour HDF indique que le modèle a identifié toutes les occurrences "réelles" de cette classe);
- le **F1-score** : moyenne harmonique de la précision et du rappel;
- le **support** : nombre d'occurrences réelles de chaque classe dans l'ensemble de test.

Dans nos résultats, la précision élevée pour "No error" (1,00) combinée au rappel de 0,90 suggère que le modèle identifie correctement la majorité des cas sans erreur, mais manque certains. Pour la classe "TWF", par exemple, une précision de 0,03 et un rappel de 0,80 indiquent que, bien que le modèle identifie la plupart des cas réels de TWF, il génère également de nombreuses fausses prédictions pour cette classe.

## Déploiement sur STM32CubeIDE

### Présentation de la carte STM32L4R9
La STM32L4R9 est un microcontrôleur ultra-basse consommation basé sur le cœur Arm Cortex M4. Elle intègre jusqu'à 2 Mo de mémoire Flash et 640 Ko de SRAM, et offre des fonctionnalités telles qu'un contrôleur LCD-TFT et une interface MIPI DSI. Ces caractéristiques la rendent adaptée aux applications nécessitant une interface graphique et une faible consommation d'énergie. 

<div align="center">
    <img src="./images/carte.png" alt="Courbes de loss et d’accuracy" width="200px"/>
    <p><em>Figure 9 : Carte STM32L4R9 </em></p>
</div>

### Analyse du modèle sous X-CUBE-AI 

La première étape à réaliser sous CubeIDE est l'analyse. Nous utiliserons le package STM32CubeMX qui permet l'évaluation, l'optimisation et la conversion de modèles d'IA pour une exécution sur microcontrôleur. Son rôle est de génèrer du code C optimisé pour l'inférence de réseaux neuronaux. L'analyse nous fournit les informations suivantes concernant l'utilisation des ressources :

|                | FLASH (ro) | %    | RAM (rw) | %    |
|-------------- |-----------|------|---------|------|
| RT total      | 10,392 B   | 64.9% | 2,480 B | 89.6% |
| TOTAL        | 16,016 B  |       | 2,768 B |      |

### X-cube-ai.c
C'est dans ce code que nous retrouverons l'implémentation du DNN, avec une communication via UART pour l’acquisition et l’envoi des données.

1. Initialisation du modèle et synchronisation UART : 
La fonction ai_boostrap() permet de créer et initialiser le réseau de neurones `predictive`. Les données du modèle seront stockées dans les buffers ai_input et ai_output. Le programme procède ensuite à la synchronisation entre l'ordinateur et la carte ( via la fonction synchronize_UART()) et attends un byte de synchronisation (0xAB). En réponse, il envoie un ACK (0xCD).

2. Acquisition et Prétraitement : Avec la méthode acquire_and_process_data(), il attend des données binaires sur l’UART et reconstitue des floats à partir des bytes reçus (on a des ) pour stocker ces valeurs dans le tableau data.

3. Exécution du modèle IA ( avec ai_run())
ai_predictive_run() effectue une prédiction sur les données reçues, puis envoie en post-traitement et envoi (post_process()) pour récupérer la sortie du modèle.

4. Envoie les valeurs via UART.

## Pistes d'améliorations
todo


## Conclusion

Ce projet a permis de démontrer qu’un modèle de maintenance prédictive peut être adapté aux contraintes d’un microcontrôleur embarqué. Cependant, l'optimisation des performances reste un des principaux obstacles, notamment en raison du déséquilibre du dataset et des limites en puissance de calcul du STM32.
