import oyaml as yaml
import numpy as np
import pickle
import pandas as pd
import json


from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy 
from flask_wtf import FlaskForm 
from wtforms import SelectField


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secret'
model = pickle.load(open('model.pkl', 'rb'))

db = SQLAlchemy(app)

class Voiture(db.Model):
    __tablename__ = 'Voiture'
    id = db.Column(db.Integer, primary_key=True)
    marque = db.Column(db.String(50))
    categorie = db.Column(db.String(50))
    modele= db.Column(db.String(50))
    

class Form(FlaskForm):
    
    marque = SelectField('marque', choices=[('Peugeot', 'Peugeot'), ('Renault', 'Renault'), ('Citroen', 'Citroen') ]) 
    categorie = SelectField('categorie', choices=[('Citadine', 'Citadine'), ('Monospace', 'Monospace'), ('Suv 4x4', 'Suv 4x4'),('Berline', 'Berline')])
    modele = SelectField('modele', choices=[])
    carburant = SelectField('carburant', choices=[('Essence', 'Essence'), ('Diesel', 'Diesel')])
    km = SelectField('kilometrage', choices=[(50000, 50000), (100000, 100000), (200000, 200000)])
    boite_vitesse = SelectField('boite de vitesse', choices=[('Manuelle', 'Manuelle'), ('Auto', 'Auto')])
    mise_circulation = SelectField('mise en circulation ', choices=[('2018', '2018'),('2017', '2017'), ('2016', '2016'),('2015', '2015')])
    p_fiscale = SelectField('Puissance fiscale', choices=[(8, 8), (6, 6), (4, 4)])




@app.route('/', methods=['GET', 'POST'])
def index():
    form = Form()
    form.modele.choices = [(modele.id, modele.modele) for modele in Voiture.query.filter_by(marque='Peugeot', categorie='Berline').all()]


    if request.method == 'POST':
        features = [request.form['marque'], request.form['modele'], request.form['categorie'],request.form['km'],  request.form['carburant'],
                request.form['boite_vitesse'],request.form['mise_circulation'], request.form['p_fiscale']]
    
        features_df = pd.DataFrame(data=[features], columns=['marque', 'modele', 'categorie', 'nb_km', 'carburant', 'boite_vitesse', 'mise_circulation','p_fiscale'])
        prediction = model.predict(features_df)

        output = round(prediction[0], 2)
       
        return '<h1>le prix est de {}</h1>'.format(output)
        
     

    return render_template('index.html', form=form )


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