import torch
import torch.nn as nn

def get_decoder_init_hidden(encoder_hidden, num_layers):
    # encoder_hidden: tuple (h, c) of shape [(num_layers*2, batch, hidden_size)]
    # Returns only the first num_layers (the "forward" layers)
    return (
        encoder_hidden[0][:num_layers].contiguous(),   # h
        encoder_hidden[1][:num_layers].contiguous()    # c
    )


# ---------------------------
# Attention Layer
# ---------------------------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, method='general'):
        super(AttentionLayer, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, 1, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        if self.method == 'general':
            energy = self.attn(encoder_outputs)  # [batch_size, seq_len, hidden_size]
            attention_weights = torch.sum(hidden * energy, dim=2)  # [batch_size, seq_len]
        elif self.method == 'dot':
            attention_weights = torch.sum(hidden * encoder_outputs, dim=2)
        elif self.method == 'concat':
            hidden_expanded = hidden.expand(encoder_outputs.size(0), -1, -1)
            energy = self.attn(torch.cat((hidden_expanded, encoder_outputs), 2)).tanh()  # [B, T, H]
            attention_weights = torch.sum(self.v * energy, dim=2)

        return torch.softmax(attention_weights, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]

# ---------------------------
# Encoder LSTM
# ---------------------------
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers=2, dropout=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)

    def forward(self, input_seq, input_lengths):
        embedded = self.dropout(self.embedding(input_seq))  # [B, T, E]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, 
                                                   batch_first=True, enforce_sorted=False)
        outputs, hidden = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # [B, T, H*2]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # [B, T, H]
        return outputs, hidden  # encoder_outputs, (h_n, c_n)

# ---------------------------
# Decoder LSTM with Attention (Standard, no genre-embedding)
# ---------------------------
class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_layers=2, dropout=0.1, attn_method='general'):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        self.attention = AttentionLayer(hidden_size, method=attn_method)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Embed the current input word
        embedded = self.dropout(self.embedding(input_step))  # [B, 1, E]

        # Pass through the decoder LSTM
        lstm_output, hidden = self.lstm(embedded, last_hidden)  # [B, 1, H]

        # Compute attention weights and context vector
        attn_weights = self.attention(lstm_output, encoder_outputs)  # [B, 1, T]
        context = attn_weights.bmm(encoder_outputs)  # [B, 1, H]

        # Concatenate context with LSTM output
        concat_input = torch.cat((lstm_output.squeeze(1), context.squeeze(1)), dim=1)  # [B, 2H]
        concat_output = torch.tanh(self.concat(concat_input))  # [B, H]

        # Final output prediction
        output = self.out(concat_output)  # [B, vocab_size]

        return output, hidden, attn_weights



# ---------------------------
# Decoder LSTM with Attention
# ---------------------------
class GenreAwareDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, num_genres, num_layers=2, dropout=0.1, attn_method='general'):
        super(GenreAwareDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, 50)
        self.lstm = nn.LSTM(
            embedding_dim + 50, hidden_size, num_layers,
            dropout=dropout, batch_first=True
        )
        self.attention = AttentionLayer(hidden_size, method=attn_method)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, last_hidden, encoder_outputs, genre_id):
        embedded = self.dropout(self.embedding(input_step))
        genre_embedded = self.genre_embedding(genre_id).unsqueeze(1)
        combined_input = torch.cat([embedded, genre_embedded.expand(-1, embedded.size(1), -1)], dim=2)

        lstm_output, hidden = self.lstm(combined_input, last_hidden)
        attn_weights = self.attention(lstm_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs)
        
        concat_input = torch.cat((lstm_output.squeeze(1), context.squeeze(1)), dim=1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)

        return output, hidden, attn_weights


    def forward(self, input_step, last_hidden, encoder_outputs):
        # Embed the current input word
        embedded = self.dropout(self.embedding(input_step))  # [B, 1, E]

        # Pass through the decoder LSTM
        lstm_output, hidden = self.lstm(embedded, last_hidden)  # [B, 1, H]

        # Compute attention weights and context vector
        attn_weights = self.attention(lstm_output, encoder_outputs)  # [B, 1, T]
        context = attn_weights.bmm(encoder_outputs)  # [B, 1, H]

        # Concatenate context with LSTM output
        concat_input = torch.cat((lstm_output.squeeze(1), context.squeeze(1)), dim=1)  # [B, 2H]
        concat_output = torch.tanh(self.concat(concat_input))  # [B, H]

        # Final output prediction
        output = self.out(concat_output)  # [B, vocab_size]

        return output, hidden, attn_weights  # Include attention weights for optional use




# ---------------------------
# Seq2Seq Model
# ---------------------------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, input_lengths, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        vocab_size = self.decoder.vocab_size

        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)

        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)

        # Encode input sequence
        encoder_outputs, hidden = self.encoder(input_seq, input_lengths)

        # Use encoder's final forward hidden state to initialize decoder
        decoder_hidden = (hidden[0][:self.decoder.num_layers],
                          hidden[1][:self.decoder.num_layers])

        decoder_input = target_seq[:, 0].unsqueeze(1)  # <SOS>

        for t in range(1, target_len):
            output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)  # [B, 1]
            decoder_input = target_seq[:, t].unsqueeze(1) if teacher_force else top1

        return outputs
