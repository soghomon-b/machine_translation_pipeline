import sys
sys.path.append(r'E:\\IREP Project\\Pipeline')

import unittest
from src import parallelDataProcessor, trainer
from parallelDataProcessor import ParallelDataProcessor
from transformer import Transformer
from trainer import Trainer
import torch 
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
#model.load_state_dict(torch.load(saved_model_path), map_location=torch.device('cpu'))

# If your model was trained with CUDA, you might need to move it to GPU
model.to(device)
class TestTrainer(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.trainer = Trainer(parameters, model, processor, "small_model")
    

    def testInit(self): 
        self.assertEqual(self.trainer.d_model, d_model)
        self.assertEqual(self.trainer.ffn_hidden, ffn_hidden)
        self.assertEqual(self.trainer.num_heads, num_heads)
        self.assertEqual(self.trainer.drop_prob, drop_prob)
        self.assertEqual(self.trainer.num_layers, num_layers)
        self.assertEqual(self.trainer.max_sequence_length, max_sequence_length)
        self.assertEqual(self.trainer.lang1_vocab_size, len(processor.lang1_vocab)-1)
        self.assertEqual(self.trainer.english_to_index['a'], processor.english_to_index['a'])
        self.assertEqual(self.trainer.index_to_lang1[1], processor.index_to_lang1[1])
        self.assertEqual(self.trainer.lang1_to_index['ا'], processor.lang1_to_index['ا'])
        self.assertEqual(self.trainer.START_TOKEN, START_TOKEN)
        self.assertEqual(self.trainer.PADDING_TOKEN, PADDING_TOKEN)
        self.assertEqual(self.trainer.END_TOKEN, END_TOKEN)

    def testTrainer(self): 
        try: 
            self.trainer.train()
        except Exception: 
            self.fail("error thrown")





if __name__ == '__main__':
    unittest.main()