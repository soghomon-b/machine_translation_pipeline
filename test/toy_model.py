import sys
sys.path.append(r'E:\\IREP Project\\Pipeline')

from transformer import Transformer
from other_data_processor import ParallelDataProcessor, DummyIterator
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
num_epochs = 10
learning_rate = 0.0005
embedding_size = 128  # Adjusted for smaller vocabulary size
src_vocab_size = 100  # Smaller source vocabulary size
trg_vocab_size = 200  # Different target vocabulary size
src_pad_idx = 0       # Example padding index in the source vocabulary
num_heads = 4         # Reduced number of heads
num_encoder_layers = 3  # Reduced number of layers
num_decoder_layers = 3  # Reduced number of layers
dropout = 0.1
max_len = 100         # Example maximum sequence length
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the Transformer model
model = Transformer(
    embedding_size=embedding_size,
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dropout=dropout,
    max_len=max_len,
    device=device
)

# Move the model to the appropriate device
model.to(device)

# Example criterion and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy data iterator for illustration purposes


file_directories = ['E:\\IREP Project\\Pipeline\\test\\toy_lang1.txt', 'E:\\IREP Project\\Pipeline\\test\\toy_eng.txt']
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
sequence_length = 100
batch_size = 32
dataset = ParallelDataProcessor(file_directories, arabic_characters, position_markers, sequence_length, batch_size)
data_iterator = DummyIterator(dataset.sequence_length, dataset)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0

    for batch in data_iterator:
        english_sentences, arabic_sentences = batch
        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        output = model(english_sentences, arabic_sentences[:-1, :])
        
        # Reshape output and target for calculating loss
        output = output.reshape(-1, trg_vocab_size)
        target = arabic_sentences[1:].reshape(-1)

        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(data_iterator)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

print('Training complete.')
