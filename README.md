# Python Chatbot Project

- End-to-end deep learning chatbot built with a sequence-to-sequence LSTM architecture and attention; trained on movie corpus dialogues for conversational modeling.

- Supports both command-line interface (CLI) and Flask-powered web front-end for flexible, interactive conversations.

- Features robust data preprocessing—text cleaning, tokenization, vocabulary building, and length filtering—for high-quality training data.

- Implements bidirectional encoder/decoder models, contextual attention layers, and optional genre-aware decoding for enhanced dialogue quality.

- Training pipeline includes dynamic batching, teacher forcing, early stopping, checkpointing, and model evaluation via BLEU/ROUGE scores.

- Beam search decoding produces diverse and relevant chatbot responses, outperforming standard greedy approaches.

- Includes support modules for Apple Silicon (MPS) compatibility, data integrity checks, and modular utility functions.
