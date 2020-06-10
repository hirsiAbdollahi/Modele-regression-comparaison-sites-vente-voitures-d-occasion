import oyaml as yaml
import numpy as np
import pickle
import pandas as pd
import json


from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy 
from flask_wtf import FlaskForm 
from wtforms import SelectField


app = Flask(__name__)

# configuration d'une db pour stocker les marques/modeles/categorie pour le select
# 'sqlite:///database2.db' = beke
# 'sqlite:///database1.db' = la centrale
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database2.db'
app.config['SECRET_KEY'] = 'secret'

#recuperation du pickle avec le fit du modele
model = pickle.load(open('model.pkl', 'rb'))

db = SQLAlchemy(app)


# class table voiture de la db 
class Voiture(db.Model):
    __tablename__ = 'Voiture'
    id = db.Column(db.Integer, primary_key=True)
    marque = db.Column(db.String(50))   
    categorie = db.Column(db.String(50))
    modele= db.Column(db.String(50))
    

# formulaire
class Form(FlaskForm):
    
    marque = SelectField('marque', choices=[('Peugeot', 'Peugeot'), ('Renault', 'Renault'), ('Citroen', 'Citroen') ]) 
    categorie = SelectField('categorie', choices=[('Citadine', 'Citadine'), ('Monospace', 'Monospace'), ('Suv 4x4', 'Suv 4x4'),('Berline', 'Berline')])
    modele = SelectField('modele', choices=[])
    carburant = SelectField('carburant', choices=[('Essence', 'Essence'), ('Diesel', 'Diesel')])
    km = SelectField('kilometrage', choices=[(50000, 50000), (100000, 100000), (200000, 200000)])
    boite_vitesse = SelectField('boite de vitesse', choices=[('Manuelle', 'Manuelle'), ('Auto', 'Auto')])
    mise_circulation = SelectField('mise en circulation ', choices=[('2019', '2019'),('2018', '2018'),('2017', '2017'), ('2016', '2016'),('2015', '2015')])
    p_fiscale = SelectField('Puissance fiscale', choices=[(8, 8), (6, 6), (4, 4)])




@app.route('/', methods=['GET', 'POST'])
def index():
    form = Form()
    form.modele.choices = [(modele.modele, modele.modele) for modele in Voiture.query.filter_by(marque='Peugeot', categorie='Berline').all()]

    # output=''

    # confirmation du form
    if request.method == 'POST':
        features = [request.form['marque'], request.form['modele'], request.form['categorie'],request.form['km'],  request.form['carburant'],
                request.form['boite_vitesse'],request.form['mise_circulation'], request.form['p_fiscale']]
    

        # recuperation l'user input dans une df pour
        features_df = pd.DataFrame(data=[features], columns=['marque', 'modele', 'categorie', 'nb_km', 'carburant', 'boite_vitesse', 'mise_circulation','p_fiscale'])
        
       
        # prediction en utilise le pickle et la df user input
        prediction = model.predict(features_df)
        output = round(prediction[0])
        
        # Flash vers vue pour afficher la prediction
        flash( 'Le prix selon le modele La centrale: ' + str(output) + ' €')
        
        # df_beke = donnée scrappé beke 
        df_beke = pd.read_csv("beke_processed_app.csv")

        # on verifie que le modele que l'user rentre est bien present chez beke
        if  df_beke[df_beke['modele']== request.form['modele']].empty is False:

            
            # on verifie que le modele + année que l'user rentre est bien present chez beke
            
            if df_beke[(df_beke['modele']== request.form['modele']) & (df_beke['mise_circulation']== int(request.form['mise_circulation']) )].empty is False:
                prix =  df_beke[(df_beke['modele']== request.form['modele']) & (df_beke['mise_circulation']== int(request.form['mise_circulation']))]['prix'].iloc[0]
                flash( 'Le prix pour la meme voiture chez nous: ' + str(prix) + ' €')
                

            else:
                prix =df_beke[df_beke['modele']== request.form['modele']]['prix'].iloc[0] 
                vehicule= df_beke[df_beke['modele']== request.form['modele']]['mise_circulation'].iloc[0]

                flash('Nous avons pas le modele  ' + str(request.form['mise_circulation']  ) )
                flash('Par contre, nous avons le meme modele année ' +  str(vehicule) + ': ' + str(prix) + ' €')  
                
               
        else:
            flash('Comparaison non possible: Modele non disponible chez Beaukocenter')  
            
    

    return render_template('index.html', form=form, )


# route pour quand on select un marque/ categorie le modele correspondans s'affiche 
@app.route('/modele/<marque>&<categorie>')
def modele(marque, categorie):
    modeles = Voiture.query.filter_by(marque=marque, categorie=categorie).all()

    modeleList = []
    
    for modele in modeles:
        modeleDict = {} 
        modeleDict ['id'] = modele.id
        modeleDict ['modele'] = modele.modele
        modeleList.append(modeleDict)

    return jsonify({'modeles' : modeleList})






if __name__ == '__main__':
    app.run(debug=True)