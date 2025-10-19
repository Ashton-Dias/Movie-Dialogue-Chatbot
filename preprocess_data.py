import re
import unicodedata
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
from nltk.corpus import stopwords
import nltk
import pickle
from vocab import Vocabulary

nltk.download('punkt')         # For word_tokenize to work
nltk.download('stopwords')     # You may use stopwords later

from convokit import Corpus, download

# Load corpus
print("Downloading and loading corpus...")
corpus = Corpus(filename=download("movie-corpus"))

# --- Conversation loop using utterance ordering ---
print("Preparing conversations...")

convos = []
for convo in corpus.iter_conversations():
    utterances = [corpus.get_utterance(utt_id) for utt_id in convo._utterance_ids]

    # Skip conversations that are too short
    if len(utterances) < 2:
        continue
    lines = [{'text': utt.text} for utt in utterances]
    convos.append({'lines': lines})

print(f"Total conversations processed: {len(convos)}")

# --- Text cleaning ---
def clean_text(text):
    """Clean and normalize text data"""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r"([.!?])", r" \1", text.lower())
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r"\s+", r" ", text).strip()
    return text

# --- Create conversation pairs ---
def create_conversation_pairs(conversations):
    """Extract question-answer pairs from conversations"""
    print("Creating input-output text pairs...")
    pairs = []
    for idx, conversation in enumerate(conversations):
        lines = conversation['lines']
        for i in range(len(lines) - 1):
            input_text = clean_text(lines[i]['text'])
            target_text = clean_text(lines[i + 1]['text'])
            if input_text and target_text:
                pairs.append([input_text, target_text])
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} conversations...")
    print(f"Finished! Total pairs created: {len(pairs)}")
    if len(pairs) == 0:
        print("⚠️ Warning: No pairs were created. Check corpus parsing logic.")
    return pairs

pairs = create_conversation_pairs(convos)

# --- Greeting/Intro filtering ---
GREETINGS = [
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
    "good evening", "how are you", "how do you do", "what's up", "whats up", "how’s it going", "how is it going"
]

def is_greeting_pair(pair):
    input_text = pair[0]
    # Check if any greeting phrase exists in the input_text (simple substring match)
    return any(greet in input_text for greet in GREETINGS)

filtered_pairs = [pair for pair in pairs if is_greeting_pair(pair)]

print(f"Filtered greeting/intro pairs count: {len(filtered_pairs)}")

if filtered_pairs:
    print("\nSample filtered pair:")
    print("Q:", filtered_pairs[0][0])
    print("A:", filtered_pairs[0][1])
else:
    print("No greeting/intro pairs found.")

# --- Vocabulary builder ---
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3
        
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def trim_rare_words(self, min_count=3):
        keep_words = [word for word, count in self.word2count.items() if count >= min_count]
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3
        for word in keep_words:
            self.add_word(word)

print("Building vocabulary...")
vocab = Vocabulary("movie_dialogue")
for pair in filtered_pairs:
    vocab.add_sentence(pair[0])
    vocab.add_sentence(pair[1])
print(f"Vocabulary built. Size: {vocab.num_words}")
print("First 10 words:", list(vocab.word2index.keys())[:10])

# --- Tokenization and length filtering ---
def tokenize_and_process(text_pairs, max_length=10):
    """Tokenize and filter text pairs"""
    processed_pairs = []

    for pair in text_pairs:
        tokens1 = tokenizer.tokenize(pair[0])
        tokens2 = tokenizer.tokenize(pair[1])

        if len(tokens1) <= max_length and len(tokens2) <= max_length:
            processed_pairs.append([' '.join(tokens1), ' '.join(tokens2)])

    return processed_pairs

print("\nFiltering tokenized sentence pairs...")
tokenized_pairs = tokenize_and_process(filtered_pairs, max_length=10)
print(f"Total usable tokenized pairs: {len(tokenized_pairs)}")

if tokenized_pairs:
    print("Sample pair:")
    print("Q:", tokenized_pairs[0][0])
    print("A:", tokenized_pairs[0][1])

# Save filtered pairs and vocab for training
with open('pairs.pkl', 'wb') as f:
    pickle.dump(tokenized_pairs, f)

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
