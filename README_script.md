# Projeto Titanic em Python (.py)

Este projeto aplica uma pipeline completa de aprendizado de máquina usando o dataset Titanic do Kaggle, executado diretamente como script Python, sem Jupyter Notebook.

## 🔧 Requisitos

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Instale os pacotes com:

```bash
pip install -r requirements.txt
```

## ▶️ Como Executar

1. Certifique-se de que os arquivos `train.csv` e `test.csv` estão na mesma pasta que `main.py`
2. No terminal, execute:

```bash
python main.py
```

## 📂 O que o script faz

- Carrega os dados do Titanic
- Preenche dados ausentes
- Converte colunas categóricas em numéricas
- Divide os dados em treino, validação e teste
- Treina um modelo Random Forest
- Avalia o desempenho com acurácia e matriz de confusão
- Faz previsões com os dados do conjunto `test.csv`

## 📊 Resultado Esperado

- Acurácia na validação: cerca de 84%
- Matriz de confusão exibida graficamente
- Previsões impressas no terminal

## 👩‍💻 Autora

Lílian Maria Damico Fonseca


https://youtu.be/PMgszLWM6Zk