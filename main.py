# Importa as bibliotecas necessárias
import pandas as pd  # manipulação de dados
import numpy as np  # operações numéricas
import matplotlib.pyplot as plt  # visualização de gráficos
import seaborn as sns  # visualização de dados
from sklearn.model_selection import train_test_split  # divisão dos dados
from sklearn.ensemble import RandomForestClassifier  # modelo de floresta aleatória
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # métricas

# Carrega os dados de treinamento e teste
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Preenche valores ausentes na idade com a mediana
train['Age'].fillna(train['Age'].median(), inplace=True)

# Preenche valores ausentes no embarque com o valor mais comum
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Cria mapas para converter sexo e embarque em números
sex_map = {'male': 0, 'female': 1}
embarked_map = {'S': 0, 'C': 1, 'Q': 2}

# Aplica os mapas nas colunas categóricas
train['Sex'] = train['Sex'].map(sex_map)
train['Embarked'] = train['Embarked'].map(embarked_map)

# Define as colunas que serão usadas como variáveis preditoras
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]  # variáveis independentes
y = train['Survived']  # variável alvo

# Divide os dados em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Cria e treina o modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faz predições nos dados de validação
y_pred = model.predict(X_val)

# Calcula a acurácia das predições
acc = accuracy_score(y_val, y_pred)
print(f"Acurácia na validação: {acc:.2f}")

# Gera a matriz de confusão
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matriz de Confusão")
plt.show()

# Pré-processamento do conjunto de teste
test['Age'].fillna(train['Age'].median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)
test['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Sex'] = test['Sex'].map(sex_map)
test['Embarked'] = test['Embarked'].map(embarked_map)

# Aplica o modelo aos dados de teste
X_final = test[features]
test_preds = model.predict(X_final)

# Mostra os 10 primeiros resultados da predição
print("Previsões para o test.csv (10 primeiros):", test_preds[:10])
