import sys
sys.path.append(r'E:\\IREP Project\\Pipeline')

import unittest
from src import parallelDataProcessor, trainer
from parallelDataProcessor import ParallelDataProcessor
from transformer import Transformer
from trainer import Trainer
import torch 
import torch.nn.functional as F

# Hyperparameters
d_model = 128  # Dimensionality of the model
ffn_hidden = 256  # Hidden size of the feedforward network
num_heads = 8  # Number of attention heads
drop_prob = 0.3  # Dropout probability
num_layers = 6  # Number of encoder/decoder layers
max_sequence_length = 100  # Maximum sequence length
embedding_dim = 512
num_epochs = 10
START_TOKEN = '<start>'  # Start token
END_TOKEN = '<end>'  # End token
PADDING_TOKEN = '<pad>'  # Padding token
files = ["E:\\IREP Project\\Data\\Arabic\\short_la_la.txt", "E:\\IREP Project\\Data\\Arabic\\short_la_eng.txt"]
arabic_characters = [
                'ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 
                'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ى', 'ة', 'ـ', 'ً', 'ٌ', 'ٍ', 
                'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', 'ٱ', 'ٲ', 'ٳ', 'ٴ', 'ٵ', 'ٶ', 'ٷ', 'ٸ', 'ٹ', 'ٺ', 'ٻ', 'ټ', 'ٽ', 
                'پ', 'ٿ', 'ڀ', 'ځ', 'ڂ', 'ڃ', 'ڄ', 'څ', 'چ', 'ڇ', 'ڈ', 'ډ', 'ڊ', 'ڋ', 'ڌ', 'ڍ', 'ڎ', 'ڏ', 'ڐ', 
                'ڑ', 'ڒ', 'ړ', 'ڔ', 'ڕ', 'ږ', 'ڗ', 'ژ', 'ڙ', 'ښ', 'ڛ', 'ڜ', 'ڝ', 'ڞ', 'ڟ', 'ڠ', 'ڡ', 'ڢ', 'ڣ', 
                'ڤ', 'ڥ', 'ڦ', 'ڧ', 'ڨ', 'ک', 'ڪ', 'ګ', 'ڬ', 'ڭ', 'ڮ', 'گ', 'ڰ', 'ڱ', 'ڲ', 'ڳ', 'ڴ', 'ڵ', 'ڶ', 
                'ڷ', 'ڸ', 'ڹ', 'ں', 'ڻ', 'ڼ', 'ڽ', 'ھ', 'ڿ', 'ۀ', 'ہ', 'ۂ', 'ۃ', 'ۄ', 'ۅ', 'ۆ', 'ۇ', 'ۈ', 'ۉ', 
                'ۊ', 'ۋ', 'ی', 'ۍ', 'ێ', 'ۏ', 'ې', 'ۑ', 'ے', 'ۓ', '۔', 'ە', 'ۖ', 'ۗ', 'ۘ', 'ۙ', 'ۚ', 'ۛ', 'ۜ', 
                '۝', '۞', '۟', '۠', 'ۡ', 'ۢ', 'ۣ', 'ۤ', 'ۥ', 'ۦ', 'ۧ', 'ۨ', '۩', '۪', '۫', '۬', 'ۭ', 'ۮ', 'ۯ', 
                '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', 'ۺ', 'ۻ', 'ۼ', '۽', '۾', 'ۿ'
            ]

position_markers = [START_TOKEN, PADDING_TOKEN, END_TOKEN]
sequence_length = 100
batch_size = 32
processor = ParallelDataProcessor(files, arabic_characters, position_markers, sequence_length, batch_size) 
parameters = [d_model, 
                ffn_hidden,
                num_heads, 
                drop_prob, 
                num_layers, 
                max_sequence_length,
                len(processor.lang1_vocab)-1,
                processor.english_to_index,
                processor.lang1_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN,
                processor.index_to_lang1, 
                num_epochs]

saved_model_path = "E:\\IREP Project\\Model\\Arabic\\pytorch_model.bin"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(
                        embedding_dim,
                        len(processor.english_vocabulary),
                        len(processor.lang1_vocab),
                        processor.english_to_index[PADDING_TOKEN],
                        num_heads,
                        num_layers,
                        num_layers,
                        drop_prob,
                        max_sequence_length,
                        device=device
                    )
PATH = "E:\\IREP Project\\Pipeline\\test\\small_model.bin"
model.load_state_dict(torch.load(PATH))
eng, arab = processor[21]
dataset = ParallelDataProcessor(files, arabic_characters, position_markers, sequence_length, batch_size)
trainer = Trainer(parameters, model, dataset, 'test_translation')


print(trainer.translate("I speak English"))