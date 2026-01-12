import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import RussianStemmer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from string import punctuation
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

def get_answer(request):

    # request = 'Как мне найти общежитие горняк'
    # request = input()

    vectorizer = TfidfVectorizer()
    stemmer = RussianStemmer()
    stop_word = set(stopwords.words('russian'))


    def preprocess(text):
        text = text.lower()
        text = ''.join([p for p in text if p not in punctuation])
        tokens = word_tokenize(text)
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_word]
        return ' '.join(tokens)


    with open('data.json', encoding='utf-8') as file:
        txt = json.load(file)
    data = tuple(t['Должность'] for t in txt)
    preprocess_data = list(preprocess(new_data) for new_data in data)

    tfidf_matrix = vectorizer.fit_transform([preprocess(request)] + preprocess_data)  # вычисление tfidf
    sparse_tfidf_matrix = csr_matrix(tfidf_matrix)  # преобразование в разреженную матрицу
    tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2', axis=1)  # нормализация матрицы по длине документа


    def jaccard_similarity():  # метрика Жаккарда
        vectorizer = TfidfVectorizer(binary=True)  # binary=True преобразует значения в бинарные
        tfidf_matrix = vectorizer.fit_transform([preprocess(request)] + preprocess_data)
        binary_matrix = tfidf_matrix.toarray()

        num_docs = binary_matrix.shape[0]
        jaccard_matrix = np.zeros((num_docs, num_docs))

        for i in range(num_docs):
            for j in range(i, num_docs):  # Симметричная матрица
                intersection = np.sum(np.minimum(binary_matrix[i], binary_matrix[j]))
                union = np.sum(np.maximum(binary_matrix[i], binary_matrix[j]))
                jaccard_matrix[i, j] = intersection / union
                jaccard_matrix[j, i] = jaccard_matrix[i, j]

        return jaccard_matrix


    def euclidean_distance(v1, v2):
        return np.linalg.norm(v1 - v2)


    def euclidean_distance_matrix():  # метрика Евклидова расстояния
        # Преобразуем разреженную матрицу в обычную для удобства вычислений
        tfidf_matrix = tfidf_matrix_normalized.toarray()

        num_docs = tfidf_matrix.shape[0]  # Количество документов (включая запрос)

        # Инициализируем матрицу для хранения Евклидовых расстояний
        euclidean_matrix = np.zeros((num_docs, num_docs))

        # Заполняем матрицу Евклидовых расстояний
        for i in range(num_docs):
            for j in range(i, num_docs):  # симметричная матрица
                # Вычисление Евклидова расстояния между векторами i и j
                distance = euclidean_distance(tfidf_matrix[i], tfidf_matrix[j])
                euclidean_matrix[i, j] = distance
                euclidean_matrix[j, i] = distance

        return euclidean_matrix


    def cos_similarity():
        cos_similatity_matrix = cosine_similarity(tfidf_matrix_normalized)
        return cos_similatity_matrix

    cosine_similarity_matrix = cos_similarity()
    jaccard_matrix = jaccard_similarity()
    euclid_matrix = euclidean_distance_matrix()

    result = []

    for ind, similarity in enumerate(zip(cosine_similarity_matrix[0], jaccard_matrix[0], euclid_matrix[0])):
        if 0 < similarity[0] < 1:
            similaritys = list(similarity)
            similaritys[2] = 1 / (1 + similarity[2])  # преобразование Евклидова расстояния в балл подобия
            result.append(
                (tuple(similaritys), data[ind - 1], tuple(t['ФИО'] for t in txt)[ind - 1],
                 tuple(t['Телефон'] for t in txt)[ind - 1]))

    sims = ['- косинусное сходство:', '- коэффициент сходства Жаккарда:', '- балл подобия по Евклидову расстоянию:']

    if not result:
        return 'Попробуй сформулировать вопрос по-другому'
    else:
        out = 'Возможно вам помогут:\n'
        for ans in result:
            if ans[0][0] > 0.20:
                out += f'👤 {ans[2]}\n'
                out += f'💼 {ans[1]}\n'
                out += f'📞 {ans[3].replace(" ", "").replace("-", "")}\n'
                out += '\nКоэффициенты сходства:\n'
                for i in zip(sims, ans[0]):
                    out += f'{i[0]} {i[1]}\n'
                out += '\n'

        return out