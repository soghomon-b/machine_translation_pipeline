


eng_file = "E:\\IREP Project\\Data\\Arabic\\short_la_eng.txt"

with open(eng_file, 'r', encoding='utf8') as file:
    eng_sentences = file.readlines()

with open('toy_eng.txt', 'w', encoding='utf-8') as file:
    file.write(''.join(eng_sentences[:1000]))
   

