import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'
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


class ParallelDataProcessor(Dataset):
    def __init__(self, file_directories: list, lang1_vocab, position_markers, sequence_length, batch_size):
        if len(file_directories) != 2:
            raise ValueError("file_directories should be exactly two")
        super().__init__()
        self.english_file = file_directories[1]
        self.lang1_file = file_directories[0]
        self.english_sentences = []
        self.lang1_sentences = []
        self.index_to_lang1 = {}
        self.lang1_to_index = {}
        self.index_to_english = {}
        self.english_to_index = {}
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.build_data()
        self.english_vocabulary = self.get_english_vocab(position_markers)
        self.lang1_vocab = self.add_position_markers(lang1_vocab, position_markers)
        self.files_to_dict()

    def add_position_markers(self, lang1_vocab, position_markers):
        lang1_vocab.append(position_markers[1])
        lang1_vocab.append(position_markers[2])
        lang1_vocab.insert(0, position_markers[0])
        lang1_vocab.append('<UNK>')
        return lang1_vocab

    def get_english_vocab(self, position_markers):
        english_vocabulary = [position_markers[0], ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                              ':', '<', '=', '>', '?', '@',
                              '[', '\\', ']', '^', '_', '`', 
                              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                              'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                              'y', 'z', 
                              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                              'U', 'V', 'W', 'X', 'Y', 'Z', '\n',  ' ',
                              '{', '|', '}', '~', position_markers[1], position_markers[2], '<UNK>']
        return english_vocabulary

    def build_data(self):
        with open(self.lang1_file, 'r', encoding='utf8') as file:
            self.lang1_sentences = file.readlines()

        with open(self.english_file, 'r', encoding='utf8') as file:
            self.english_sentences = file.readlines()

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        eng_indices = [self.english_to_index.get(char, self.english_to_index['<UNK>']) for char in self.english_sentences[idx]]
        eng_tensor = torch.tensor(eng_indices, dtype=torch.long).unsqueeze(1)

        lang1_indices = [self.lang1_to_index.get(char, self.lang1_to_index['<UNK>']) for char in self.lang1_sentences[idx]]
        lang1_tensor = torch.tensor(lang1_indices, dtype=torch.long).unsqueeze(1)
        
        return eng_tensor, lang1_tensor
    
    def collate_fn(self, batch):
        # Separate the batch into source and target sentences
        src_batch, trg_batch = zip(*batch)
        
        # Pad the sequences to the maximum length within the batch
        src_padded = rnn_utils.pad_sequence(src_batch, batch_first=True, padding_value=self.english_to_index['<PAD>'])
        trg_padded = rnn_utils.pad_sequence(trg_batch, batch_first=True, padding_value=self.lang1_to_index['<PAD>'])
        
        # Transpose to match the shape (sequence length, batch size)
        src_padded = src_padded.transpose(0, 1)
        trg_padded = trg_padded.transpose(0, 1)
        
        return src_padded, trg_padded

    def files_to_dict(self):
        self.index_to_lang1 = {k: v for k, v in enumerate(self.lang1_vocab)}
        self.lang1_to_index = {v: k for k, v in enumerate(self.lang1_vocab)}
        self.index_to_english = {k: v for k, v in enumerate(self.english_vocabulary)}
        self.english_to_index = {v: k for k, v in enumerate(self.english_vocabulary)}
        return {}

# Example usage
file_directories = ['E:\\IREP Project\\Pipeline\\test\\toy_lang1.txt', 'E:\\IREP Project\\Pipeline\\test\\toy_eng.txt']
sequence_length = 100
batch_size = 32
class DummyIterator:
        def __init__(self, max_len, iteratee) -> None:
            self.max_len = max_len
            self.iteratee = iteratee
        def __iter__(self):
            target_shape = (self.max_len, 32)
            for i in range(100):
                src, trg = self.iteratee[i]
                src = self.resize_tensor(src, target_shape)  # (sequence length, batch size)
                trg = self.resize_tensor(trg, target_shape) # (sequence length, batch size)
                yield src, trg


        def resize_tensor(self, tensor, target_shape, pad_value=0):
            """
            Resize a tensor to match the target shape by padding or truncating.
            
            Args:
                tensor (torch.Tensor): The input tensor to be resized.
                target_shape (tuple): The desired shape for the tensor (sequence length, batch size).
                pad_value (int): The value to use for padding if the tensor is shorter than the target shape.
            
            Returns:
                torch.Tensor: The resized tensor.
            """
            target_len, target_batch_size = target_shape
            
            # Get the current shape of the tensor
            current_len, current_batch_size = tensor.size()
            
            # If the tensor is longer than the target length, truncate it
            if current_len > target_len:
                tensor = tensor[:target_len, :]
            # If the tensor is shorter than the target length, pad it
            elif current_len < target_len:
                pad_size = target_len - current_len
                pad_tensor = torch.full((pad_size, current_batch_size), pad_value, dtype=tensor.dtype)
                tensor = torch.cat((tensor, pad_tensor), dim=0)
            
            # If the tensor has more batches than the target, truncate it
            if current_batch_size > target_batch_size:
                tensor = tensor[:, :target_batch_size]
            # If the tensor has fewer batches than the target, pad it
            elif current_batch_size < target_batch_size:
                pad_size = target_batch_size - current_batch_size
                pad_tensor = torch.full((target_len, pad_size), pad_value, dtype=tensor.dtype)
                tensor = torch.cat((tensor, pad_tensor), dim=1)
            
            return tensor

