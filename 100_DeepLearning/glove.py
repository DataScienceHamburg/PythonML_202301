#%%
import torch
import torchtext.vocab as vocab
# %%
glove = vocab.GloVe(name='6B', dim=100)

glove.vectors.shape
# %% string to index
glove.stoi['king']

#%% index to string
glove.itos[691]
# %% embedding vector
def get_embedding_vector(word):
    word_index = glove.stoi[word]
    emb = glove.vectors[word_index]
    return emb

word_emb = get_embedding_vector('king')
# %% find closest words based on input word
def get_closest_words_from_embedding(word_emb, max_n=5):
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return dist_sort_filt

# %%
get_closest_words_from_embedding(word_emb)
# %% word analogies
# king is to queen like man is to woman
def get_word_analogy(word1, word2, word3, max_n=5):
    # logic: w1=king, w2=queen, w3=man --> w4=woman
    # w4 = w1- w2+w3
    word1_emb = get_embedding_vector(word1)
    word2_emb = get_embedding_vector(word2)
    word3_emb = get_embedding_vector(word3)
    word4_emb = word1_emb - word2_emb + word3_emb
    analogy = get_closest_words_from_embedding(word4_emb)
    return analogy
# %%
get_word_analogy(word1='king', word2='queen', word3='brother')
# %%
