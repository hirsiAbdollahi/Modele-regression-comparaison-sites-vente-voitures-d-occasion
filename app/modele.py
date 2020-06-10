
import numpy as np
import pandas as pd
import pickle

# Basic tools :
import numpy as np                      
import pandas as pd  


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


# Others :
from itertools import cycle
from scipy import interp
from sklearn.pipeline import Pipeline
import time

lacentrale = pd.read_csv('lacentrale_clean_finale.csv')

lacentrale.mise_circulation=lacentrale.mise_circulation.astype(str)


def preprocessing():
    
    # on définit les colonnes et les transformations pour 
    # les colonnes quantitatives
    col_quanti=['nb_km','p_fiscale']

    transfo_quanti = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='median')),
        ('standard', StandardScaler())])


    col_quali_nom= ['marque','categorie','modele','boite_vitesse', 'mise_circulation' ]


    transfo_quali_nom = Pipeline(steps=[
        ('imputation', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # on définit l'objet de la classe ColumnTransformer
    # qui va permettre d'appliquer toutes les étapes

    preparation = ColumnTransformer(
        transformers=[
            ('quanti', transfo_quanti , col_quanti),
            # ('quali_ord', transfo_quali_ord , col_quali_ord),
            ('quali_nom', transfo_quali_nom , col_quali_nom)
        ], remainder='drop')
    
    return preparation 

preparation = preprocessing()


regression = {
        
       'RF': { 'model':RandomForestRegressor(),
              'param':{
                  'clf__n_estimators': [100,200,300,500,1000],
                'clf__max_depth': [1,5,10,15,50,70],
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
                'param': {'clf__alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         'clf__l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                         }
             },
    
        'LR': { 'model': LogisticRegression(),
                'param': {'clf__penalty' : ['l1', 'l2'], 'clf__C' : np.logspace(-4, 4, 20), 'clf__solver' : ['liblinear']},
             },
       
        'SVR':{ 'model': SVR(),
                'param': {'clf__C': [0.1,1, 10, 100], 'clf__gamma': [1,0.1,0.01,0.001],'clf__kernel': ['rbf', 'poly', 'sigmoid'],
                         },
             },
      
#         'XGB':{ "model":XGBRegressor(),
#               "param":{"clf__learning_rate": [0.05,1,5],'clf__n_estimators': [100,50],
# #                        "clf__max_depth": [5,10,15]
#                   },
#             },
    
        'GradientBoost':{ "model":GradientBoostingRegressor(),
              "param":{"clf__model__n_estimators": [ 500, 600,700,800,1000],
#                         "clf__max_depth": [2, 3, 4]
                  },
            },
    
        'decisionTree':{ "model":GradientBoostingRegressor(),
              "param":{"clf__criterion": ['mse', 'mae'],
                'clf__min_samples_leaf': [5, 10, 15, 20, 25],
                'clf__max_depth': [6, 9, 12, 15, 20],
                  },
            },
         
        
}



from sklearn.base import TransformerMixin
class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def modele_entier(cl, df_train, df_test=None):
    model = Pipeline(steps=[('preparation', preparation),
                            # ('to_dense', DenseTransformer()),
                         ('clf', cl['model'] )
                    ])
    
    # PROCESSING & ENTRAINEMENT

    param_grid = cl['param']

    # on sépare la cible du reste des données (dataset d'entraînement)
    X =df_train.drop(['nom', 'ref', 'co2', 'carburant', 'couleur', 'nb_portes', 'nb_places',
       'conso_mixte', 'p_din', 'critair', 'prix'], axis=1) 
    y = df_train['prix']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=42)
    # Debut du decompte du temps
    start_time = time.time()

    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=10, verbose=False)
    
    fit = grid.fit(X_train,y_train)
    
                                    
    return fit


# la fonction  retourne le fit qu'on  insere ensuite dans notre pickle 
fit = modele_entier(regression['SVR'], lacentrale)


pickle.dump(fit, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))