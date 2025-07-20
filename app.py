import numpy as np
from flask import Flask, request, render_template, jsonify
import joblib
import os
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Configuração do App Flask ---
app = Flask(__name__)
MODEL_PATH = '/home/bruno/Estudos/Novo/modelo_xgboost.joblib'

# --- Lógica do Modelo ---
def treinar_e_salvar_modelo():
    """Treina o modelo XGBoost e o salva em um arquivo."""
    print("Iniciando o treinamento do modelo XGBoost...")
    
    # 1. Carregar dados (already normalized by load_diabetes)
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # 2. Dividir em treino e teste
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Criar o regressor (sem StandardScaler, pois os dados já são pré-normalizados)
    regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
    
    # 4. Treinar o modelo
    regressor.fit(X_train, y_train)
    
    # 5. Salvar o modelo
    joblib.dump(regressor, MODEL_PATH)
    print(f"Modelo treinado e salvo em {MODEL_PATH}")
    return regressor

def carregar_modelo():
    """Carrega o modelo do arquivo, ou treina se o arquivo não existir."""
    if not os.path.exists(MODEL_PATH):
        return treinar_e_salvar_modelo()
    else:
        print(f"Carregando modelo existente de {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

# Carrega o modelo ao iniciar a aplicação
modelo = carregar_modelo()

# Define real-world normalization parameters for each feature
# These are approximate values based on typical ranges for the Diabetes dataset features
# and are used to map real-world inputs to the normalized scale of load_diabetes features.
NORMALIZATION_PARAMS = {
    'age': {'min_real': 19, 'max_real': 79, 'mean': 49.0, 'std': 13.1},
    'bmi': {'min_real': 18, 'max_real': 40, 'mean': 26.3, 'std': 4.4},
    'bp': {'min_real': 60, 'max_real': 150, 'mean': 94.6, 'std': 13.8},
    's1': {'min_real': 100, 'max_real': 300, 'mean': 189.2, 'std': 34.6},
    's2': {'min_real': 50, 'max_real': 200, 'mean': 115.5, 'std': 30.4},
    's3': {'min_real': 20, 'max_real': 100, 'mean': 49.7, 'std': 12.9},
    's4': {'min_real': 1, 'max_real': 10, 'mean': 4.1, 'std': 1.3},
    's5': {'min_real': 50, 'max_real': 200, 'mean': 138.0, 'std': 28.4},
    's6': {'min_real': 50, 'max_real': 200, 'mean': 91.2, 'std': 11.4},
}

# Define feature names and their real-world ranges/mappings for the frontend
FEATURE_INFO = {
    'age': {'label': 'Idade', 'unit': 'anos', 'min': 19, 'max': 79, 'default': 49, 'description': 'Idade do paciente em anos.'},
    'sex': {'label': 'Sexo', 'options': {'Male': 0.0506801187398187, 'Female': -0.0446416365065942}, 'default': 'Male', 'description': 'Sexo do paciente (Male para masculino, Female para feminino).'},
    'bmi': {'label': 'IMC', 'unit': 'kg/m²', 'min': 18, 'max': 40, 'default': 25, 'description': 'Índice de Massa Corporal.'},
    'bp': {'label': 'Pressão Sanguínea Média', 'unit': 'mmHg', 'min': 60, 'max': 150, 'default': 90, 'description': 'Pressão arterial média.'},
    's1': {'label': 'Colesterol Total (TC)', 'unit': 'mg/dL', 'min': 100, 'max': 300, 'default': 200, 'description': 'Nível de colesterol total.'},
    's2': {'label': 'Colesterol LDL', 'unit': 'mg/dL', 'min': 50, 'max': 200, 'default': 120, 'description': 'Nível de colesterol LDL (colesterol "ruim").'},
    's3': {'label': 'Colesterol HDL', 'unit': 'mg/dL', 'min': 20, 'max': 100, 'default': 50, 'description': 'Nível de colesterol HDL (colesterol "bom").'},
    's4': {'label': 'Colesterol VLDL', 'unit': 'mg/dL', 'min': 1, 'max': 10, 'default': 5, 'description': 'Nível de colesterol VLDL.'},
    's5': {'label': 'Glicose', 'unit': 'mg/dL', 'min': 50, 'max': 200, 'default': 100, 'description': 'Nível de glicose no sangue.'},
    's6': {'label': 'Nível de Açúcar no Sangue', 'unit': 'mg/dL', 'min': 50, 'max': 200, 'default': 100, 'description': 'Outra medição de açúcar no sangue.'}
}

# --- Rotas da Aplicação Web ---
@app.route('/')
def home():
    """Renderiza a página inicial com o formulário."""
    return render_template('index.html', feature_info=FEATURE_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Recebe os dados do formulário, faz a previsão e mostra o resultado."""
    try:
        original_features = {}
        normalized_features = []

        feature_keys = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        
        for i, key in enumerate(feature_keys):
            val = request.form[key]
            original_features[key] = val # Store original value for display

            if key == 'sex':
                # Sex is already mapped to normalized values in FEATURE_INFO options
                normalized_features.append(FEATURE_INFO[key]['options'][val])
            else:
                # Manual normalization using mean and std from the original dataset
                # This maps real-world input to the approximate normalized range of load_diabetes
                mean_val = NORMALIZATION_PARAMS[key]['mean']
                std_val = NORMALIZATION_PARAMS[key]['std']
                
                normalized_val = (float(val) - mean_val) / std_val
                normalized_features.append(normalized_val)

        dados_para_prever = np.array(normalized_features).reshape(1, -1)

        prediction_raw = modelo.predict(dados_para_prever)[0]

        return render_template('index.html', prediction=prediction_raw,
                               original_features=original_features,
                               feature_info=FEATURE_INFO)

    except Exception as e:
        return render_template('index.html', prediction=f"Erro ao processar: {e}", feature_info=FEATURE_INFO)

# --- Ponto de Entrada da Aplicação ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
