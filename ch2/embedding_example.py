import numpy as np

vocab = {'the':0, 'cat':1, 'sits':2, 'on': 3, 'mat': 4}
np.random.seed(42)
embedding_matrix = np.random.randn(5, 4)

print('Embedding matrix', embedding_matrix)
print()

cat_vec = embedding_matrix[vocab['cat']]
print('cat vec: ', cat_vec)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sits_vec = embedding_matrix[vocab['sits']]
print('cosine sim (cat, sits) ', cosine_sim(cat_vec, sits_vec))

the_vec = embedding_matrix[vocab['the']]
print('cosine sim (cat, the) ', cosine_sim(cat_vec, the_vec))