#untested and unused class, still work in progress

from transformers import BertTokenizer, BertModel
import torch


class Validator():
    def __init__(self, func, *args, eval_lang1, eval_eng, model) -> None:
        with open(eval_lang1, 'r', encoding='utf8') as file: 
            self.lang1_sentences = file.readlines()
        with open(eval_eng, 'r', encoding='utf8') as file: 
            self.eng1_gold_sentences = file.readlines()
        self.eval_func = func 
        self.args = args
        self.model = model
        self.vec_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def evaluate(self):
        results = []
        for i in range(len(self.lang1_sentences)):
            output = self.eval_func(self.lang1_sentences[i], self.args)
            gold = self.eng1_gold_sentences[i]
            result = self.compare(gold, output)
            results.append(result)
        
        average = 0 

        for result in results: 
            average += result 

        average /= len(results)

        return average

    def comapre(self, sentence1, sentence2): 
        sentence1_vec = self.get_sentence_vector(sentence1)
        sentence2_vec = self.get_sentence_vector(sentence2)
        distance =  torch.norm(sentence1_vec - sentence2_vec)

        return distance
    

    def get_word_vector(self,word):
        # Tokenize the input word
        inputs = self.tokenizer(word, return_tensors='pt')
        
        # Get the hidden states from BERT
        with torch.no_grad():
            outputs = self.vec_model(**inputs)
        
        # The hidden states are the embeddings, we use the embeddings of the first token
        embeddings = outputs.last_hidden_state[0][0]
        
        return embeddings
    
    def get_sentence_vector(self, sentence):
        sentence_vec = torch.zeros(768)  # Initialize a zero vector of the same dimension as BERT embeddings
        num_words = 0
        for word in sentence.split():
            word_vec = self.get_word_vector(word)
            sentence_vec += word_vec
            num_words += 1
        if num_words > 0:
            sentence_vec /= num_words  # Normalize by the number of words
        return sentence_vec