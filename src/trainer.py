import sys
sys.path.append(r'E:\\IREP Project\\src')


import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from parallelDataProcessor import ParallelDataProcessor, ParallelDataIterator
'''
trainer classes used in the pipeline. Initilises the training and saves the model. 


parameters is a list that contains the parameters in the following order: 
[d_model, 
ffn_hidden,
num_heads, 
drop_prob, 
num_layers, 
max_sequence_length,
lang1_vocab_size,
english_to_index,
lang1_to_index,
START_TOKEN, 
END_TOKEN, 
PADDING_TOKEN
index_to_lang1
num_epochs]
'''

NEG_INFTY = -1e9
class Trainer(): 
    def __init__(self, parameters : list, model, dataset : ParallelDataProcessor, model_name : str):
        self.model_name = model_name
        self.model = model
        self.d_model = parameters[0]
        self.ffn_hidden = parameters[1]
        self.num_heads = parameters[2]
        self.drop_prob = parameters[3]
        self.num_layers = parameters[4]
        self.max_sequence_length = parameters[5]
        self.lang1_vocab_size = parameters[6]
        self.english_to_index = parameters[7]
        self.lang1_to_index = parameters[8]
        self.START_TOKEN = parameters[9]
        self.END_TOKEN = parameters[10]
        self.PADDING_TOKEN = parameters[11]
        self.index_to_lang1 = parameters[12]
        self.num_epochs = parameters[13]
        self.batch_size = 32
        self.criterian = nn.CrossEntropyLoss(ignore_index=self.lang1_to_index[self.PADDING_TOKEN],
                                reduction='none')
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = torch.self.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.train_loader = DataLoader(dataset, self.batch_size)
        self.iterator = ParallelDataIterator(self.max_sequence_length, dataset)
        pad_idx = dataset.english_to_index[self.PADDING_TOKEN]
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.processor = dataset
        self.num_epochs = 10
    
    #trains the model 
    def train(self):
        for params in self.model.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)
        self.model.train()
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            epoch_loss = 0

            for batch in self.iterator:
                english_sentences, arabic_sentences = batch
                self.optim.zero_grad()  # Reset gradients
                # Forward pass
                output = self.model(english_sentences, arabic_sentences[:-1, :])
                # Reshape output and target for calculating loss
                output = output.view(-1, output.size(-1))  # Flatten the first two dimensions
                target = arabic_sentences[1:].view(-1)  

                # Calculate loss
                loss = self.criterion(output, target)
                
                # Backward pass and optimization
                loss.backward()
                self.optim.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.processor)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        self.save_model()
        return self.model_name + '.bin'

    #saves the model in the current working directory
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_name + '.bin')
        print(f"Model saved as {self.model_name}.bin")
    
    #loads the model from the path needed. 
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    #uses the model to generate the numbers representing vocabulary items from the model. 
    def generate_translation(self, input_tensor, decoder_input, src_padding_mask):
        output_sequence = []

        for _ in range(self.max_sequence_length):
            # Generate mask for the decoder input
            trg_mask = self.model.transformer.generate_square_subsequent_mask(decoder_input.size(0)).to(self.device)
            
            # Forward pass through the model
            output = self.model(input_tensor, input_tensor)
            
            # Get the predicted token
            next_token = output.argmax(2)[-1, -1]
            print(output.argmax(2)[-1])
            output_sequence.append(next_token)

            # Break if the end token is predicted
            if next_token == self.lang1_to_index[self.END_TOKEN]:
                break
            
            # Append the predicted token to the decoder input
            decoder_input = torch.cat([decoder_input, torch.LongTensor([[next_token]]).to(self.device)], dim=0)
        
        return output_sequence
    
    #turns the numbers generated by generate_translations into vocabulary items and returns the vocabulary items in one string. 
    def translate(self, input_sentence):
        self.model.eval()
        with torch.no_grad():
            # Tokenize input sentence letter by letter
            input_tokens = [self.english_to_index[char] for char in input_sentence]
            input_tensor = torch.LongTensor(input_tokens).unsqueeze(0).to(self.device)

            # Initialize decoder input with start token
            decoder_input = torch.LongTensor([self.lang1_to_index[self.START_TOKEN]]).unsqueeze(0).to(self.device)

            # Initialize attention masks
            src_padding_mask = self.model.make_src_mask(input_tensor)

            # Generate translation
            output_indices = self.generate_translation(input_tensor, decoder_input, src_padding_mask)

            # Convert indices to characters
            translated_sentence = [self.index_to_lang1[idx.item()] for idx in output_indices if idx.item() != self.lang1_to_index[self.END_TOKEN]]
        
        return ''.join(translated_sentence)



    
        

