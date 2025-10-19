from flask import Flask, render_template, request, jsonify
import torch
from vocab import Vocabulary
from seq2seq_models import EncoderLSTM, DecoderLSTM
from text_utils import clean_text
from chatbot import BeamSearchDecoder, load_vocabulary
import pickle
from train import load_checkpoint

app = Flask(__name__)

# Global variables for model, vocab, decoder
encoder = None
decoder = None
vocab = None
beam_decoder = None

def load_model():
    global encoder, decoder, vocab, beam_decoder
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load config dictionary
    with open('models/config.pkl', 'rb') as f:
        MODEL_CONFIG = pickle.load(f)
    
    # Load saved vocabulary object
    vocab = load_vocabulary('models/vocab.pkl')
    
    # Set vocab_size dynamically to avoid mismatch
    MODEL_CONFIG['vocab_size'] = vocab.num_words
    
    # Instantiate encoder and decoder with matching vocab_size
    encoder = EncoderLSTM(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout'],
    )
    decoder = DecoderLSTM(
        vocab_size=MODEL_CONFIG['vocab_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout'],
    )
    
    # Load weights from the checkpoint
    load_checkpoint('models/chatbot_model.pth', encoder, decoder)
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    beam_decoder = BeamSearchDecoder(encoder, decoder, vocab)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    print(f"USER: {user_message}")

    if not user_message:
        return jsonify({'response': 'Please enter a message.'})

    cleaned_input = clean_text(user_message)
    print(f"CLEANED: {cleaned_input}")

    tokens = [vocab.word2index.get(word, 0) for word in cleaned_input.split()]
    print(f"TOKENS: {tokens}")

    # Check if tokens is empty or only PAD tokens (0)
    if not tokens or all(t == 0 for t in tokens):
        print("No tokens in vocab for input.")
        return jsonify({'response': "I didn't understand that."})

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_tensor = torch.LongTensor([tokens]).to(device)
    input_length = torch.LongTensor([len(tokens)])

    with torch.no_grad():
        response_tokens = beam_decoder.decode(input_tensor, input_length)

    print(f"RESPONSE TOKENS: {response_tokens}")

    remove_tokens = ['EOS', 'PAD', 'SOS']
    filtered_tokens = [token for token in response_tokens if token.upper() not in remove_tokens]

    if not filtered_tokens:
        print("All output tokens were special tokens; returning fallback response.")
        response = "I am not sure how to respond to that yet."
    else:
        response = ' '.join(filtered_tokens)

    print(f"FINAL RESPONSE: {response}")

    return jsonify({'response': response})


if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
