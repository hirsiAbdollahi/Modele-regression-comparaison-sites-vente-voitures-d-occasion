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


regression = {
        
    'RF': { 'model':RandomForestRegressor(),
            'param':{
                'clf__n_estimators': [100,200,300],
                'clf__max_depth': [1,5,10],
#                   'clf__min_samples_split': [1,5,10,15]
#                   'cl_max_leaf_nodes': [ 100, 200, 300, 400, 500, 600, 650, 700, 800]
                },
            },
        'Lasso': { 'model': LassoCV(),
                'param': {'clf__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            },       
        'Ridge': { 'model': RidgeCV(),
                'param': {'clf__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            },        
        'Elastic': { 'model': ElasticNet(),
                # 'param': {'clf__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                #         'clf__l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                #         }
                'param': {'clf__alpha':[0.001],
                        'clf__l1_ratio' : [0.1]
                        }
            },        
        'LR': { 'model': LogisticRegression(),
                'param': {'clf__penalty' : ['l1', 'l2'], 'clf__C' : np.logspace(-4, 4, 20), 'clf__solver' : ['liblinear']},
            },        
        'SVR':{ 'model': SVR(),
                'param': {'clf__C': [0.1,1, 10, 100], 'clf__gamma': [1,0.1,0.01,0.001],'clf__kernel': ['rbf', 'poly', 'sigmoid'],
                        },
            },       
        'XGB':{ "model":XGBRegressor(),
            "param":{"clf__learning_rate": [0.05,1,5],
                    'clf__n_estimators': [100,50],
#                        "clf__max_depth": [5,10,15]
                },
            },
        'GradientBoost':{ "model":GradientBoostingRegressor(),
            "param":{"clf__model__alpha": [0.0, 0.5, 1.0],
                        "clf__ccp_alpha": [0.0, 0.5]
                },
            },
        'decisionTree':{ "model":GradientBoostingRegressor(),
            "param":{"clf__criterion": ['mse', 'mae'],
                'clf__min_samples_leaf': [5, 10, 15, 20, 25],
                'clf__max_depth': [6, 9, 12, 15, 20],
                },
            },          
}


def modele_entier_2(cl, df_train, df_test):

    model = Pipeline(steps=[('preparation', preprocessing_2()),
                        ('to_dense', DenseTransformer()),
                        ('clf', cl['model'] )
                ])   

    # ---------- PROCESSING & ENTRAINEMENT ----------

    param_grid = cl['param']

#     on sépare la cible du reste des données (dataset d'entraînement) 
#     X = df_train.drop(['prix','nom','ref'], axis=1) # ON GARDE TOUTES LES COLONNES
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
                        scoring='neg_root_mean_squared_error'
)
    grid.fit(X_train,y_train)
    
    print(grid.best_estimator_)
    
    # Score de l'entraînement
    rmse = grid.score(X_test, y_test)
    print(("RMSE  : %.5f" % rmse))
    
    # Temps d'entraînement
    times = (time.time() - start_time)
    print("Temps d'entraînement' : %s secondes ---" % times)    

    
    # --------------- PREDICTIONS ---------------
    
#     on sépare la cible du reste des données (dataset de test) 
#     X_reel = df_test.drop(['prix','nom','ref'], axis=1) # ON GARDE TOUTES LES COLONNES
    X_reel = df_test.drop(['nom', 'ref', 'co2', 'carburant', 'couleur', 'nb_portes', 'nb_places',
       'conso_mixte', 'p_din', 'critair', 'prix'], axis=1) # ON GARDE LES COLONNES CORRELEES FORTEMENT

    y_reel = df_test['prix']  
    
    y_pred = grid.predict(X_reel)   
    
    # dic = {
    # "prix_prediction": y_pred,
    # "prix_beke":beke['prix'].values,
    # "p_beke-p_pred":beke['prix'].values - y_pred,
    # }

    # comparaison = pd.DataFrame(data=dic)
                                    
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

###
#title
st.title("Acheter une voiture d'occasion")
st.markdown("Marques françaises seulement, d'autres marques bientôt disponibles...")

##################### INPUT ######################

marque = st.selectbox('Choisissez la marque', ('Renault', 'Citroen', 'Peugeot'))

if marque == 'Renault':
    modele = st.selectbox('Choisissez le modèle', ('Scenic 3', 'Clio 4', 'Kadjar', 'Megane 3', 'Megane 4', 'Captur',
       'Espace 5', 'Clio 5', 'Talisman', 'Espace 4', 'Scenic 4',
       'Twingo 2 rs', 'Koleos 2', 'Clio 3', 'Laguna 3', 'Avantime',
       'Safrane', 'Twingo 3', 'Grand scenic 4', 'Grand scenic 3',
       'Clio 2 campus', 'Twingo 2', 'Modus', 'Grand modus',
       'Grand scenic 2', 'Latitude', 'Vel satis', 'Clio 4 rs',
       'Megane 4 rs', 'Koleos'))
elif marque == 'Citroen':
    modele = st.selectbox('Choisissez le modèle', ('C3', 'C4 picasso 2', 'C4 aircross',
    'Grand c4 picasso 2', 'C4 cactus', 'C5 aircross', 'C3 aircross',
    'C4', 'C5', 'C4 picasso', '2cv', 'Ds4', 'C4 spacetourer', 'Ds3',
    'Berlingo 3 multispace', 'C3 picasso', 'Xsara picasso', 'C6', 'C1',
    'C2', 'Grand c4 spacetourer', 'Berlingo 2 multispace', 'Jumpy 3'))
else :
    modele = st.selectbox('Choisissez le modèle', (
    '2008', '308', '5008', '3008', '206+', '208', 'Rifter',
    'Partner 2 tepee', '207', '508', '108', '4008', '308  gti',
    '508 rxh', '208 gti'))

categorie = st.selectbox('Spécifiez la catégorie qui va avec le modèle', ('Monospace', 'Citadine', 'Suv 4x4', 'Berline'))

nb_km = st.slider("Kilométrage",0,300000)

boite_vitesse = st.selectbox('Choisissez votre boite de vitesses', ('Manuelle', 'Auto'))

mise_circulation = st.slider("Année",2000,2019)

p_fiscale = st.slider("Puissance Fiscale",2,19)

# Array for prediction

dico = {'nom':'nom', 'ref':'ref', 
    "marque": marque, "modele":modele, "categorie":categorie, 
    'co2': 'A', "nb_km":nb_km, 'carburant':'carburant',
    "boite_vitesse":boite_vitesse, 'couleur':'couleur', 'nb_portes':2, 'nb_places':3,
    "mise_circulation":int(mise_circulation), 'conso_mixte':5.5, "p_fiscale":p_fiscale, 
    'p_din':110, 'critair':1, 'prix':21366} 

data_user = pd.DataFrame([dico])

##################### Prediction ######################

lacentrale, beke = get_data()

elastic_result = modele_entier_2(regression['Elastic'], lacentrale, data_user)

st.success(f"prix : {round(int(elastic_result),0)}")




