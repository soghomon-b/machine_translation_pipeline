import torch 
from torch import nn, Dataset
from transformer import Transformer
from parallelDataProcessor import ParallelDataProcessor
from trainer import Trainer
from validator import Validator




'''
manages the training process for one Transformer model for machine translation purposes 
Tasks that are managed: 
1. Data Ingestion: simple variable for two lines
    adding the data to the model 
2. Data preprocessing: ParallelDataProcessor
    Remove noise 
    Tokenization 
    Encoding  
3. Model Training:  Trainer
    Inject the first model 
    Start the training process 
4. Model Validation : Validator

(5.) fine tuner : tbd

Parameters: 
l1 = data for the first langauge translated 

l2 = english data

initial_parameters are in this order: 
    [d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length]


position_markers are in this order: 
    [START_TOKEN, PADDING_TOKEN, END_TOKEN]
'''
class Pipeline():   
    def __init__(self, l1 : str, l2: str, model, initial_parameters : list, position_markers : list) -> None:
        #1
        self.model = model

        #2
        self.data_files = [l1, l2]
        self.data_processor = ParallelDataProcessor(self.data_files, position_markers) 

        #3
        self.parameters = self.get_parameters(initial_parameters[0], initial_parameters[1], initial_parameters[2], 
                                         initial_parameters[3], initial_parameters[4], initial_parameters[5], 
                                          position_markers[0], position_markers[1], position_markers[2] )
        self.trainer = Trainer(self.parameters, self.model, self.data_processor) 

        #4
        self.validator = Validator() 
        self.trained_model_path = ''

     #combines the initial_parameters and the position_markers along with some others into one parameter
     #explanation of parameters can be found above   
     #returns the parameters list needed for training class
    def get_parameters(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, start, padding, end): 
        lang1_vocab_size = self.data_processor.lang1_vocab 
        english_to_index = self.data_processor.english_to_index
        lang1_to_index = self.data_processor.lang1_to_index
        
        index_to_lang1 = self.data_processor.index_to_lang1

        parameters = [d_model, 
                        ffn_hidden,
                        num_heads, 
                        drop_prob, 
                        num_layers, 
                        max_sequence_length,
                        lang1_vocab_size,
                        english_to_index,
                        lang1_to_index,
                        start, 
                        end, 
                        padding,
                        index_to_lang1]
        return parameters
    
    #starts the training process
    def train(self):
        self.saved_path = self.trainer.train(self.model)


#exceptions 
