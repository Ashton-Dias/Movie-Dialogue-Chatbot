import pickle
import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
import os
import time

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Project modules
from vocab import Vocabulary
from seq2seq_models import EncoderLSTM, DecoderLSTM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def load_checkpoint(checkpoint_path, encoder, decoder, encoder_optimizer=None, decoder_optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if encoder_optimizer and decoder_optimizer:
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('loss', None)
    print(f"Checkpoint loaded (epoch {epoch}, val_loss {val_loss}).")
    return epoch, val_loss

# Initial MODEL_CONFIG without vocab_size (set after vocab rebuild)
MODEL_CONFIG = {
    'hidden_size': 256,
    'embedding_dim': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'batch_size': 64,
    'learning_rate': 0.003,
    'max_sequence_length': 10,
    'teacher_forcing_ratio': 0.7,
    'epochs': 5
}

with open('models/config.pkl', 'wb') as f:
    pickle.dump(MODEL_CONFIG, f)

def prepare_training_data(pairs, vocab, test_size=0.2, val_size=0.1):
    train_pairs, test_pairs = train_test_split(pairs, test_size=test_size, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=val_size, random_state=42)

    def pairs_to_tensors(pair_list):
        input_seqs = []
        target_seqs = []
        for pair in pair_list:
            input_seq = [vocab.word2index.get(word, 0) for word in pair[0].split()]
            target_seq = [vocab.word2index.get(word, 0) for word in pair[1].split()]
            input_seqs.append(torch.tensor(input_seq + [2]))               # <EOS>
            target_seqs.append(torch.tensor([1] + target_seq + [2]))       # <SOS> + ... + <EOS>
        return input_seqs, target_seqs

    return map(pairs_to_tensors, [train_pairs, val_pairs, test_pairs])

def create_batches(inputs, targets, batch_size):
    data = list(zip(inputs, targets))
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        input_batch, target_batch = zip(*batch)
        input_batch = pad_sequence(input_batch, batch_first=True, padding_value=0)
        target_batch = pad_sequence(target_batch, batch_first=True, padding_value=0)
        input_lengths = torch.tensor([len(seq[seq != 0]) for seq in input_batch])
        yield input_batch, target_batch, input_lengths

def validate_model(encoder, decoder, val_data, criterion, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in create_batches(*val_data, MODEL_CONFIG['batch_size']):
            input_batch, target_batch, input_lengths = batch
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths)

            decoder_input = torch.LongTensor([[1]] * input_batch.size(0)).to(device)
            decoder_hidden = (encoder_hidden[0][:decoder.num_layers], encoder_hidden[1][:decoder.num_layers])

            loss = 0
            target_len = target_batch.size(1)
            for t in range(1, target_len):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_batch[:, t])
                decoder_input = decoder_output.argmax(1).unsqueeze(1)

            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)

def calculate_bleu_score(references, hypotheses):
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, hyp in zip(references, hypotheses):
        reference_tokens = [ref.split()]
        hypothesis_tokens = hyp.split()
        score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores)

def calculate_rouge_score(references, hypotheses):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores_1, scores_2, scores_L = [], [], []

    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        scores_1.append(score['rouge1'].fmeasure)
        scores_2.append(score['rouge2'].fmeasure)
        scores_L.append(score['rougeL'].fmeasure)

    return {
        'rouge-1': sum(scores_1) / len(scores_1),
        'rouge-2': sum(scores_2) / len(scores_2),
        'rouge-l': sum(scores_L) / len(scores_L)
    }

def evaluate_model_metrics(encoder, decoder, test_data, vocab, config):
    encoder.eval()
    decoder.eval()
    predicted, references = [], []
    with torch.no_grad():
        for inp_tensor, tgt_tensor in zip(*test_data):
            inp = inp_tensor.unsqueeze(0).to(device)
            tgt = tgt_tensor
            length = torch.tensor([inp.shape[1]])
            enc_out, enc_hidden = encoder(inp, length)
            dec_input = torch.LongTensor([[1]]).to(device)
            dec_hidden = (enc_hidden[0][:decoder.num_layers], enc_hidden[1][:decoder.num_layers])
            generated = []
            for _ in range(config['max_sequence_length']):
                out, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
                top1 = out.argmax(1)
                if top1.item() == 2: break  # <EOS>
                generated.append(top1.item())
                dec_input = top1.unsqueeze(1)
            gen_sentence = ' '.join(vocab.index2word.get(w, '<UNK>') for w in generated)
            ref_sentence = ' '.join(vocab.index2word.get(w, '<UNK>') for w in tgt.tolist() if w not in [0, 1, 2])
            predicted.append(gen_sentence)
            references.append(ref_sentence)
    bleu = calculate_bleu_score(references, predicted)
    rouge = calculate_rouge_score(references, predicted)

    print("\n📊 Evaluation Metrics:")
    print(f"🔵 BLEU: {bleu:.4f}")
    print(f"🟥 ROUGE-1 F1: {rouge['rouge-1']:.4f}")
    print(f"🟧 ROUGE-2 F1: {rouge['rouge-2']:.4f}")
    print(f"🟨 ROUGE-L F1: {rouge['rouge-l']:.4f}")

def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        'loss': loss,
        'model_config': {
            'vocab_size': encoder.embedding.num_embeddings,
            'hidden_size': encoder.hidden_size,
            'embedding_dim': encoder.embedding.embedding_dim,
            'num_layers': encoder.num_layers
        }
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved to {filepath}")

def train_model(encoder, decoder, train_data, val_data, vocab, config,
                encoder_optimizer, decoder_optimizer, resume=False, checkpoint_path='models/checkpoint.pth'):
    encoder.to(device)
    decoder.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=3)

    start_epoch = 0
    best_val_loss = float('inf')
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, encoder, decoder, encoder_optimizer, decoder_optimizer)
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")
    else:
        print("Starting training from scratch.")

    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        encoder.train()
        decoder.train()
        total_loss = 0
        batch_count = 0

        batches = list(create_batches(*train_data, config['batch_size']))
        for batch_idx, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch+1}")):
            input_batch, target_batch, input_lengths = batch
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths)
            decoder_input = torch.LongTensor([[1]] * input_batch.size(0)).to(device)
            decoder_hidden = (encoder_hidden[0][:decoder.num_layers], encoder_hidden[1][:decoder.num_layers])

            loss = 0
            target_len = target_batch.size(1)
            tf_ratio = config['teacher_forcing_ratio']

            for t in range(1, target_len):
                output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(output, target_batch[:, t])
                decoder_input = target_batch[:, t].unsqueeze(1) if random.random() < tf_ratio else output.argmax(1).unsqueeze(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        epoch_end = time.time()
        print(f"Epoch {epoch + 1} duration: {epoch_end - epoch_start:.2f} seconds")
        print(f"Batches/sec: {batch_count / (epoch_end - epoch_start):.2f}")

        avg_train_loss = total_loss / batch_count
        val_loss = validate_model(encoder, decoder, val_data, criterion, device)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch + 1, val_loss, checkpoint_path)

        if early_stopping(val_loss):
            print("✅ Early stopping triggered.")
            break

    return train_losses, val_losses



if __name__ == "__main__":
    with open("models/pairs.pkl", "rb") as f:
        tokenized_pairs = pickle.load(f)

    tokenized_pairs = tokenized_pairs[:2000]  # Subset for faster training

    # Rebuild vocabulary on subset
    vocab = Vocabulary("greetings_subset")
    for pair in tokenized_pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])

    # Set vocab_size dynamically for the model config
    MODEL_CONFIG['vocab_size'] = vocab.num_words

    print("MODEL_CONFIG:", MODEL_CONFIG)
    print(f"Loaded {len(tokenized_pairs)} pairs.")
    print(f"Vocabulary size: {vocab.num_words}")

    # Save updated config and vocab
    with open('models/config.pkl', 'wb') as f:
        pickle.dump(MODEL_CONFIG, f)

    with open('models/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = prepare_training_data(tokenized_pairs, vocab)

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

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=MODEL_CONFIG['learning_rate'], weight_decay=1e-5)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=MODEL_CONFIG['learning_rate'], weight_decay=1e-5)

    train_losses, val_losses = train_model(
        encoder, decoder,
        train_data=(train_inputs, train_targets),
        val_data=(val_inputs, val_targets),
        vocab=vocab,
        config=MODEL_CONFIG,
        encoder_optimizer=encoder_optimizer,
        decoder_optimizer=decoder_optimizer,
        resume=False,
        checkpoint_path='models/checkpoint.pth'
    )

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, 'models/chatbot_model.pth')

    final_checkpoint_path = 'models/final_model.pth'
    last_epoch = len(train_losses)
    last_val_loss = val_losses[-1]
    save_checkpoint(
        encoder, decoder,
        encoder_optimizer, decoder_optimizer,
        last_epoch, last_val_loss,
        final_checkpoint_path
    )

    evaluate_model_metrics(
        encoder, decoder,
        (test_inputs, test_targets),
        vocab,
        MODEL_CONFIG
    )
