import streamlit as st

# Basic tools :
import numpy as np                      
import pandas as pd  

# Plot
import matplotlib.pyplot as plt         
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler,LabelBinarizer 
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# Predictors :
from sklearn.linear_model import LinearRegression,LogisticRegression, LogisticRegressionCV, RidgeCV, LassoCV, ElasticNet
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
# from sklearn import neighbors 
# from sklearn.naive_bayes import GaussianNB , ComplementNB,CategoricalNB              

# Metrics : 
from sklearn.metrics import mean_squared_error, r2_score,roc_curve, roc_auc_score, auc 
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score, accuracy_score

# Optimization / Validation :
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score

from sklearn import svm, datasets,preprocessing

# cell multiple outputs
from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

# Others :
from itertools import cycle
from scipy import interp
from sklearn.pipeline import Pipeline
import time

from sklearn.decomposition import PCA

from sklearn.base import TransformerMixin


#################################################
#################################################
################### FONCTIONS ###################
#################################################
#################################################

# retourne l'intersection de deux listes : utile pour afficher seulement les modèles d'une marque donnée
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

# Pipelines de préparation
def preprocessing_2():

    # on définit les colonnes et les transformations pour 
    # les colonnes quantitatives
    col_quanti=['nb_km','p_fiscale']
    
    transfo_quanti = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='median')),
        ('standard', StandardScaler())])

    # on définit les colonnes et les transformations pour
    # les variables qualitatives

    # qualitatives nominales
    col_quali_nom= ['marque','modele','categorie','boite_vitesse', 'mise_circulation' ]
    transfo_quali_nom = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # on définit l'objet de la classe ColumnTransformer
    # qui va permettre d'appliquer toutes les étapes
    preparation = ColumnTransformer(
        transformers=[
            ('quanti', transfo_quanti , col_quanti),
            ('quali_nom', transfo_quali_nom , col_quali_nom)
        ], remainder='drop')
    
    return preparation 

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

# dictionnaire des regresseurs avec les "best params"
regression = {  
        'RF': { 'model':RandomForestRegressor(),
            'param':{
                'clf__n_estimators': [100],
                'clf__max_depth': [10],
                },
            },  
        'Elastic': { 'model': ElasticNet(),
                'param': {'clf__alpha':[0.01],
                        'clf__l1_ratio' : [0.7]
                        }
            },         
        'SVR':{ 'model': SVR(),
                'param': {'clf__C': [1], 'clf__gamma': [1],'clf__kernel': ['poly'],
                        },
        }}

# Dictionnaires des marques et categories de voitures, utile pour les classer
categories_beke = {"Monospace":['Berlingo', 'Grand c4 spacetourer', '5008', 'Rifter', 'Scénic', 'Trafic'],
              "Citadine": ['C1', 'C3', '108', '208', 'Twingo iii'],
              "Berline": ['C3', 'C4 cactus', 'C4 cactus business', '308', '308 sw', '508', 
                           '508 nouvelle', 'Clio iv business', 'Clio v', 'Kadjar', 'Mégane iii', 
                           'Mégane iv', 'Megane iv berline'],
              "Suv 4x4": ['C3 aircross', 'C5 aircross', '2008', '3008', '5008', 'Captur',
                           'Kadjar', 'Kadjar nouveau']  }
categories_lacentrale = {
    "Monospace":['Scenic 3', 'Espace 5', 'Espace 4', 'Scenic 4', 'Avantime',
       'Grand scenic 4', 'Grand scenic 3', 'Modus', 'Grand modus',
       'Grand scenic 2', 'C4 picasso 2', 'Grand c4 picasso 2',
       'C4 picasso', 'C4 spacetourer', 'Berlingo 3 multispace',
       'C3 picasso', 'Xsara picasso', 'Grand c4 spacetourer',
       'Berlingo 2 multispace', 'Jumpy 3', '5008', 'Rifter',
       'Partner 2 tepee'],
    "Citadine":['Clio 4', 'Clio 5', 'Twingo 2 rs', 'Clio 3', 'Twingo 3',
       'Clio 2 campus', 'Twingo 2', 'Clio 4 rs', 'C3', '2cv', 'Ds3', 'C1',
       'C2', '206+', '208', '207', '108', '208 gti'],
    "Berline":['Megane 3', 'Megane 4', 'Talisman', 'Laguna 3', 'Safrane',
       'Latitude', 'Vel satis', 'Megane 4 rs', 'C4', 'C5', 'Ds4', 'C6',
       '308', '508', '308  gti'],
    "Suv 4x4":['Kadjar', 'Captur', 'Koleos 2', 'Koleos', 'C4 aircross',
       'C4 cactus', 'C5 aircross', 'C3 aircross', '2008', '3008', '4008',
       '508 rxh'],} 
modeles_lacentrale = {
    "Renault":['Scenic 3', 'Clio 4', 'Kadjar', 'Megane 3', 'Megane 4', 'Captur',
       'Espace 5', 'Clio 5', 'Talisman', 'Espace 4', 'Scenic 4',
       'Twingo 2 rs', 'Koleos 2', 'Clio 3', 'Laguna 3', 'Avantime',
       'Safrane', 'Twingo 3', 'Grand scenic 4', 'Grand scenic 3',
       'Clio 2 campus', 'Twingo 2', 'Modus', 'Grand modus',
       'Grand scenic 2', 'Latitude', 'Vel satis', 'Clio 4 rs',
       'Megane 4 rs', 'Koleos'],
    "Citroen": ['C3', 'C4 picasso 2', 'C4 aircross',
        'Grand c4 picasso 2', 'C4 cactus', 'C5 aircross', 'C3 aircross',
        'C4', 'C5', 'C4 picasso', '2cv', 'Ds4', 'C4 spacetourer', 'Ds3',
        'Berlingo 3 multispace', 'C3 picasso', 'Xsara picasso', 'C6', 'C1',
        'C2', 'Grand c4 spacetourer', 'Berlingo 2 multispace', 'Jumpy 3'],
    "Peugeot": ['2008', '308', '5008', '3008', '206+', '208', 'Rifter',
        'Partner 2 tepee', '207', '508', '108', '4008', '308  gti',
        '508 rxh', '208 gti']}

# Compilation des pipelines de préaration, modèle et prédictions.
def modele_entier_2(cl, df_train, df_test):
    model = Pipeline(steps=[('preparation', preprocessing_2()),
                        ('to_dense', DenseTransformer()),
                        ('clf', cl['model'] )
                ])   
    param_grid = cl['param']
    # ---------- PROCESSING & ENTRAINEMENT ----------

    # on sépare la cible du reste des données (dataset d'entraînement) 
    X = df_train.drop(['nom', 'ref', 'co2', 'carburant', 'couleur', 'nb_portes', 'nb_places',
       'conso_mixte', 'p_din', 'critair', 'prix'], axis=1) # ON GARDE LES COLONNES CORRELEES FORTEMENT
    y = df_train['prix']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
    
    # Debut du decompte du temps
    start_time = time.time()

    # Model training avec Gridsearch
    grid = GridSearchCV(estimator=model, 
                        param_grid=param_grid, 
                        n_jobs=-1, cv=10, verbose=False, 
                        scoring='neg_root_mean_squared_error')
    grid.fit(X_train,y_train)
    print(grid.best_estimator_)
    
    # Score de l'entraînement
    rmse = grid.score(X_test, y_test)
    print(("RMSE  : %.5f" % rmse))
    
    # Temps d'entraînement
    times = (time.time() - start_time)
    print("Temps d'entraînement' : %s secondes ---" % times)    
    
    # --------------- PREDICTIONS ---------------    

   #  on sépare la cible du reste des données (dataset de test) 
    X_reel = df_test.drop(['nom', 'ref', 'co2', 'carburant', 'couleur', 'nb_portes', 'nb_places',
       'conso_mixte', 'p_din', 'critair', 'prix'], axis=1) # ON GARDE LES COLONNES CORRELEES FORTEMENT
    y_reel = df_test['prix']    
    y_pred = grid.predict(X_reel)   

    return y_pred

# Permet de garder resultat fonction en cache
@st.cache
#load data
def get_data():
    print('bonjour')
    return pd.read_csv('lacentrale_clean_finale.csv'), pd.read_csv('beke_processed.csv')

#################################################
#################################################
##################### PAGE ######################
#################################################
#################################################


##################### PREAMBLE ######################
st.title("Acheter une voiture d'occasion")
st.markdown("Marques françaises seulement, d'autres marques bientôt disponibles...")
lacentrale, beke = get_data() # load datasets

##################### INPUT SPACES ######################
algorithme = st.sidebar.selectbox(
    'Quel estimateur désirez-vous?',
     ('RF', 'Elastic', 'SVR'))
marque = st.selectbox('Choisissez la marque', ('Renault', 'Citroen', 'Peugeot'))
categorie = st.selectbox('Quelle type de voiture cherchez vous ?', ('Monospace', 'Citadine', 'Suv 4x4', 'Berline'))
modele = st.selectbox('Choisissez le modèle', (intersection(modeles_lacentrale[marque], categories_lacentrale[categorie])))
carburant = st.selectbox('Quel moteur ?', ('Diesel', 'Essence'))
nb_km = st.slider("Kilométrage",0,100000)
boite_vitesse = st.selectbox('Choisissez votre boite de vitesses', ('Manuelle', 'Auto'))
mise_circulation = st.slider("Année",2000,2019)
p_fiscale = st.slider("Puissance Fiscale",3,14)

##################### ARRAY FOR PREDICTION ######################
dico = {'nom':'nom', 'ref':'ref', 
    "marque": marque, "modele":modele, "categorie":categorie, 
    'co2': 'A', "nb_km":nb_km, 'carburant':'carburant',
    "boite_vitesse":boite_vitesse, 'couleur':'couleur', 'nb_portes':2, 'nb_places':3,
    "mise_circulation":int(mise_circulation), 'conso_mixte':5.5, "p_fiscale":p_fiscale, 
    'p_din':110, 'critair':1, 'prix':21366} 
data_user = pd.DataFrame([dico])

##################### PRIX LACENTRALE ######################
if st.button("Prix La Centrale"):
    rfr_result = modele_entier_2(regression[algorithme], lacentrale, data_user)
    st.success(f"Prix La Centrale : {round(int(rfr_result),0)} €")

##################### PRIX BEKEAUTOCENTER ######################
if st.button("Prix Bekeautocenter : si vous trouvez moins cher ailleurs, on vous rembourse la différence !"):
    try: 
        prix_beke = beke[(beke.marque == marque) &
        (beke.categorie == categorie) &
        ((nb_km - 2000 <= beke.nb_km)&(beke.nb_km < nb_km + 2000)) &
        (beke.boite_vitesse == boite_vitesse) &
        (beke.mise_circulation == mise_circulation) ]['prix'].min()
        st.success(f"Nous avons la voiture pour vous à partir de : {round(int(prix_beke),0)} € seulement !")
    except:
        try:
            prix_beke = beke[(beke.marque == marque) &
            (beke.categorie == categorie) &
            (beke.boite_vitesse == boite_vitesse) &
            (beke.mise_circulation >= mise_circulation) ]['prix'].min()
            st.success(f"Quelle chance ! Nous avons des {categorie} {marque} (boîte {boite_vitesse}, année {mise_circulation} ou plus), à partir de : {round(int(prix_beke),0)} € seulement !")
        except:
            try:
                prix_beke = beke[(beke.marque == marque) &
                (beke.categorie == categorie)]['prix'].min()
                st.success(f"Nous aimons les clients exigeants ! Découvrez nos {categorie} {marque}, à partir de : {round(int(prix_beke),0)} € seulement !")
            except:
                st.warning("Appelez nous pour chercher ensemble la voiture de vos rêves !")