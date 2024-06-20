# README for Transformer Model Training Pipeline

## Overview
This pipeline manages the training process of a Transformer model for machine translation purposes. The tasks managed by the pipeline include data ingestion, data preprocessing, model training, and model validation. Future updates may include a fine-tuning module.

## Tasks Managed
1. **Data Ingestion**: Simple variable for two lines of data. Adds the data to the model.
2. **Data Preprocessing**: Handled by `ParallelDataProcessor`. This includes:
   - Removing noise
   - Tokenization
   - Encoding
3. **Model Training**: Handled by `Trainer`. This includes:
   - Injecting the first model
   - Starting the training process
4. **Model Validation**: Handled by `Validator`.
5. **Fine Tuning**: (To be determined)

## Parameters
- **l1**: Data for the first language (translated).
- **l2**: English data.

### Initial Parameters
These parameters are required in the following order:
- `d_model`
- `ffn_hidden`
- `num_heads`
- `drop_prob`
- `num_layers`
- `max_sequence_length`

### Position Markers
These markers are required in the following order:
- `START_TOKEN`
- `PADDING_TOKEN`
- `END_TOKEN`

## Installation
Ensure you have the necessary dependencies installed. You can install them via pip:
```bash
pip install torch
```

## Usage
Here is an example of how to use the `Pipeline` class.

```python
import torch
from torch import nn
from transformer import Transformer
from parallelDataProcessor import ParallelDataProcessor
from trainer import Trainer
from validator import Validator

class Pipeline:   
    def __init__(self, l1: str, l2: str, model: nn.Module, initial_parameters: list, position_markers: list) -> None:
        # Step 1: Model
        self.model = model

        # Step 2: Data Ingestion and Preprocessing
        self.data_files = [l1, l2]
        self.data_processor = ParallelDataProcessor(self.data_files, position_markers) 

        # Step 3: Model Training
        self.parameters = self.get_parameters(
            initial_parameters[0], initial_parameters[1], initial_parameters[2], 
            initial_parameters[3], initial_parameters[4], initial_parameters[5], 
            position_markers[0], position_markers[1], position_markers[2]
        )
        self.trainer = Trainer(self.parameters, self.model, self.data_processor) 

        # Step 4: Model Validation
        self.validator = Validator() 
        self.trained_model_path = ''

    def get_parameters(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, start, padding, end): 
        lang1_vocab_size = self.data_processor.lang1_vocab 
        english_to_index = self.data_processor.english_to_index
        lang1_to_index = self.data_processor.lang1_to_index
        index_to_lang1 = self.data_processor.index_to_lang1

        parameters = [
            d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length,
            lang1_vocab_size, english_to_index, lang1_to_index, start, end, padding, index_to_lang1
        ]
        return parameters
    
    def train(self):
        self.saved_path = self.trainer.train(self.model)

# Example usage
l1 = 'path/to/lang1_data.txt'
l2 = 'path/to/english_data.txt'
initial_parameters = [512, 2048, 8, 0.1, 6, 100]  # Example parameters
position_markers = ['<s>', '<pad>', '</s>']
model = Transformer()

pipeline = Pipeline(l1, l2, model, initial_parameters, position_markers)
pipeline.train()
```

## Classes and Their Responsibilities
### `Pipeline`
- **Attributes**:
  - `model`: The Transformer model to be trained.
  - `data_files`: List of data files for the two languages.
  - `data_processor`: Instance of `ParallelDataProcessor` for data preprocessing.
  - `parameters`: Training parameters.
  - `trainer`: Instance of `Trainer` to handle model training.
  - `validator`: Instance of `Validator` to handle model validation.
  - `trained_model_path`: Path where the trained model is saved.
  
- **Methods**:
  - `__init__(self, l1, l2, model, initial_parameters, position_markers)`: Initializes the pipeline.
  - `get_parameters(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, start, padding, end)`: Combines initial parameters and position markers into a list of parameters needed for the `Trainer`.
  - `train(self)`: Starts the training process.

## Future Work
- **Fine Tuning Module**: To be determined (TBD).
- **Evaluation**: While the code exists, it has not yet been used or tested.

## Contact
For any questions or issues, please contact [soghmon5@gmail.com].

