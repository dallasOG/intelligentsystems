import pandas as pd
import pickle

# Dados para criar os dataframes
columns_names = ['sepal_length', 
                 'sepal_width', 
                 'petal_length', 
                 'petal_width',
                 'Iris-setosa', 
                 'Iris-versicolor', 
                 'Iris-virginica']

# Criar um dataframe vazio, com a estrutura desejada
flor_dataframe = pd.DataFrame(columns = columns_names)

# Nova flor
nova_flor = [(6.4, 2.8, 5.6, 2.1)]

# Abrir o normalizador
normalizador = pickle.load(open('normalizador_iris.pkl', 'rb'))

# Abrir o modelo salvo
cluster_iris = pickle.load(open('cluster_iris.pkl', 'rb'))

# Normalizar os dados de entrada
nova_flor_normalizada = normalizador.transform(nova_flor)

# Converter a nova instancia normalizada em dataframe
nova_flor_normalizada = pd.DataFrame(nova_flor_normalizada, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Concatenar o dataframe

flor_nova_instancia = pd.concat([nova_flor_normalizada, flor_dataframe]).fillna(0)
print(flor_nova_instancia)

cluster_flor = cluster_iris.predict(flor_nova_instancia)
print('Cluster da nova flor: ', cluster_flor)

67