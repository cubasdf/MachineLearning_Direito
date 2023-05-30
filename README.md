import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_excel('D:\\OneDrive\\meus_programas_python\\TCU\\curso_machine_learning\\DF_sem_padrão_simplificado_muito_poucas_linhas_classificado_treinamento_A.xlsx')

with pd.option_context('display.width',None):
    print(df)

#gráfico da quantidade das classificações
#plt.plot(df['n_classificação'].value_counts())
#plt.show()

#TODO COM MAIS DE 5 CLASSIFICADORES, O sistema acusa o seguinte erro:
#ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
# Separa os dados de treino e teste (proporção 80/20)
X_train, X_test, y_train, y_test = train_test_split(df['Texto Completo'], df['classificação'], test_size=0.2, stratify=df['classificação'],random_state=42)
print(X_train[:10])

#proporção de classes nos dados de treino
print('TREINO : ',y_train.shape,'\n', y_train.value_counts(normalize=True),'\n')
print('TESTE  : ',y_test.shape,'\n', y_test.value_counts(normalize=True))

# instancia o transform CountVectorizer
tfidf_vectorizer = TfidfVectorizer()
# # tokeniza e cria o vocabulário
tfidf_vectorizer.fit(X_train)
# # mostra o vocabulário criado
print('Vocabulário: ')
print(tfidf_vectorizer.vocabulary_)

# encode document
tfidf_vector = tfidf_vectorizer.transform(X_train)
# mostra as dimensões da matrix de frequência
print('\nDimensões da matrix: ')
print(tfidf_vector.shape)

# Define o pipeline incluindo o extrator de 'features do texto e um classificador
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5, max_features=50000,ngram_range=(1,2))),
    ('clf', SGDClassifier(loss='log', penalty='elasticnet', alpha=1e-06)) ])

# Treine o modelo(fit)
classificador = pipeline.fit(X_train, y_train)

# Mostra a acurácia do modelo nos dados de teste
print(f'acurácia: {classificador.score(X_test,y_test)}')
#TODO acurácia de 0.64 é bem baixa. Nesse caso, o regex oferece um resultado melhor. Os valores variam até 0.72

# Predição nos dados de teste
predicted = classificador.predict(X_test)

# Mostra o f1 score do modelo nos dados de teste (usado quando as classes estão desbalanceadas)
from sklearn.metrics import f1_score
print(f"f1 score: {f1_score(y_test, predicted, average='weighted')}")
#TODO fi score foi de 0.63. Às vezes dá 0.70

#Avaliação do modelo
print(metrics.classification_report(y_test, predicted))

#matriz de confusão
display = ConfusionMatrixDisplay(classificador).plot()
plt.show()
