# Contains hugging face implementation of BERT to get started
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Input text
text = "Replace this with your own text."

# Tokenize input text
input_ids = tokenizer.encode(text, add_special_tokens=True)

# Convert tokenized input to tensor
input_tensor = torch.tensor([input_ids])

# Set model to evaluation mode
model.eval()

# Generate embeddings
with torch.no_grad():
    outputs = model(input_tensor)

# Extract embeddings from the last layer
last_hidden_states = outputs.last_hidden_state

# For sentence embeddings, you can take the mean of all token embeddings
sentence_embedding = torch.mean(last_hidden_states, dim=1)

# Print the embedding tensor
print(sentence_embedding)