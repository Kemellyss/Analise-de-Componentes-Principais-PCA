# Analise-de-Componentes-Principais-PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Carregar os dados do arquivo Excel
dados = pd.read_excel(r"C:/Users/breno/Desktop/python2/Travel_details_dataset.xlsx.xltx")

# Selecionar as colunas relevantes para a análise de PCA
dados_selecionados = dados[['Destination', 'Accommodation type', 'Accommodation cost', 'Transportation cost']]

# Codificar as variáveis categóricas usando one-hot encoding
dados_codificados = pd.get_dummies(dados_selecionados)

# Tratar valores ausentes
imputer = SimpleImputer(strategy='mean')
dados_imputados = imputer.fit_transform(dados_codificados)

# Normalizar os dados
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(dados_imputados)

# Criar o objeto PCA e ajustá-lo aos dados normalizados
pca = PCA()
dados_transformados = pca.fit_transform(dados_normalizados)

# Imprimir o tamanho da matriz de autovetores
print("Tamanho da Matriz de Autovetores:", pca.components_.shape)

# Imprimir o melhor autovalor
melhor_autovalor = pca.explained_variance_[0]
print("Melhor Autovalor:", melhor_autovalor)

# Identificar a linha com o melhor autovalor
indice_melhor_autovalor = np.argmax(pca.components_[0])
linha_melhor_autovalor = dados.iloc[indice_melhor_autovalor]

# Imprimir a linha com o melhor autovalor (apenas as colunas 'Destination' e 'Accommodation type')
print("Linha com o Melhor Autovalor:")
print(linha_melhor_autovalor[['Destination', 'Accommodation type']])

# Obter a variável categórica para colorir os pontos
categorias = pd.factorize(dados_selecionados['Destination'])[0]

# Plotar o gráfico de dispersão em 2D com cores
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
sc = ax.scatter(dados_transformados[:, 0], dados_transformados[:, 1], c=categorias)
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_title('Gráfico de Dispersão em 2D (Colorido)')
plt.colorbar(sc, label='Destino')
plt.show()

# Plotar o gráfico de dispersão em 3D com cores
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(dados_transformados[:, 0], dados_transformados[:, 1], dados_transformados[:, 2], c=categorias)
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('Gráfico de Dispersão em 3D (Colorido)')
plt.colorbar(sc, label='Destino')
plt.show()
