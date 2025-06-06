from flask import Flask, render_template, request, jsonify
import pandas as pd

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Agora você pode importar a função desejada
from Utils.Uteis import Modelo


app = Flask(__name__)

utilModelo = Modelo()

model = utilModelo.carregarModelo("../Modelo/modelo.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recebe os dados do formulário
        data = request.form.to_dict()
        
        # Prepara os dados para o modelo
        input_data = prepare_input(data)
        
        # Faz a predição
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Converte para resultado legível
        result = "Aprovar" if prediction == 1 else "Recusar"
        
        return render_template('result.html', 
                            result=result, 
                            probability=f"{probability*100:.2f}%",
                            input_data=data)
    
    except Exception as e:
        return str(e)

def prepare_input(form_data):
    """Converte os dados do formulário para o formato do modelo"""
    # Cria um DataFrame com todas as colunas esperadas pelo modelo
    input_df = pd.DataFrame(columns=[...])  # Liste todas as 56 colunas aqui
    
    # Preenche os valores recebidos
    for key, value in form_data.items():
        if key in input_df.columns:
            input_df[key] = [float(value) if value.replace('.','',1).isdigit() else 0]
    
    # Preenche valores padrão para colunas não enviadas
    input_df = input_df.fillna(0)
    
    return input_df

if __name__ == '__main__':
    app.run(debug=True)