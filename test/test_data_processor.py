import sys
sys.path.append(r'E:\\IREP Project\\Pipeline')


import unittest
from src import parallelDataProcessor
from parallelDataProcessor import ParallelDataProcessor, NotAppropriateData


START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'


class test_parallelDataProcessor(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.files = ['E:\\IREP Project\\Pipeline\\test\\toy_lang1.txt', 'E:\\IREP Project\\Pipeline\\test\\toy_eng.txt']
        self.arabic_characters = [
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
        self.position_markers = [START_TOKEN, PADDING_TOKEN, END_TOKEN]

        self.processor = ParallelDataProcessor(self.files, self.arabic_characters, self.position_markers) 

    def testInitFile(self):
        self.assertEqual(self.processor.english_file, 'E:\\IREP Project\\Pipeline\\test\\toy_eng.txt') 
        self.assertEqual(self.processor.lang1_file, 'E:\\IREP Project\\Pipeline\\test\\toy_lang1.txt')
        with self.assertRaises(NotAppropriateData): 
            ParallelDataProcessor([""], self.arabic_characters, self.position_markers)
    
    def testInitData(self): 
        eng_sentence = "Let's talk about the time Moldova made Romania a birthday cake and Romania said it tasted good even though it didn't.\n"
        arabic_sentence = "خلونا نحكي عن الوقت اللي قدمت فيه مالدوفا لرومانيا، كعكة عيد ميلاد ورومانيا قالت إنو طيبة حتى لو ما كانت طيبة" + "\n"
        self.assertEqual(self.processor.english_sentences[0], eng_sentence )
        self.assertEqual(self.processor.lang1_sentences[0],  arabic_sentence)

        

    def testInitDicts(self):
        eng_vocab_length = len(self.processor.english_vocabulary)
        lang1_vocab_length = len(self.processor.lang1_vocab)
        self.assertEqual(self.processor.index_to_lang1[1], 'ا')
        self.assertEqual(self.processor.lang1_to_index['ب'], 2)
        self.assertEqual(self.processor.index_to_english[0], START_TOKEN)
        self.assertEqual(self.processor.index_to_english[eng_vocab_length -2], END_TOKEN)
        self.assertEqual(self.processor.index_to_english[eng_vocab_length -3], PADDING_TOKEN)
        self.assertEqual(self.processor.index_to_lang1[0], START_TOKEN)
        self.assertEqual(self.processor.index_to_lang1[lang1_vocab_length -2], END_TOKEN)
        self.assertEqual(self.processor.index_to_lang1[lang1_vocab_length-3], PADDING_TOKEN)
        self.assertEqual(self.processor.index_to_english[4], '#')
        self.assertEqual(self.processor.english_to_index['a'], 39)

    def testLen(self):
        self.assertEqual(len(self.processor.english_sentences), 1000)
        self.assertEqual(len(self.processor.english_sentences), len(self.processor.lang1_sentences))
        self.assertEqual(len(self.processor), 1000)

    def testGetItem(self): #passed
        eng_sentence, arab_sentence = self.processor[321]
        print(self.processor.english_to_index['a'])
        print(self.processor.english_sentences[321])
        print(eng_sentence)
        print(arab_sentence)



if __name__ == '__main__':
    unittest.main()

