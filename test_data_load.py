# test_data_load.py
from vocab import Vocabulary
import pickle

with open('pairs.pkl', 'rb') as f:
    tokenized_pairs = pickle.load(f)
print(f"Loaded {len(tokenized_pairs)} tokenized pairs.")

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
print(f"Vocabulary loaded. Sample words: {list(vocab.word2index.keys())[:10]}")
