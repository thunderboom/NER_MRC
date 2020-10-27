import re
import os
import jieba
# import spacy
# nlp = spacy.load("zh_core_web_sm")python -m spacy download zh_core_web_sm

def merge_dataset(data_dir, gen_path=None):
    fw = open(gen_path, 'w', encoding='utf-8')
    file_list = sorted(os.listdir(test_dir), key=lambda x: int(x[:-4]))
    for file in file_list:
        f = open(os.path.join(data_dir, file), encoding='utf-8')
        text = f.readlines()
        if len(text) > 1:
            text = [''.join([sen.strip() for sen in text])]
        fw.write(text[0].strip() + '\n')
        f.close()
    fw.close()
    print("finished")
       
pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def gen_pretrain(data_path, gen_path):
    fw = open(gen_path, 'w', encoding='utf-8')
    with open(data_path, 'r', encoding='utf-8') as fr:
        max_length = 0
        max_token_nums = 0
        for text in fr.readlines():
            # doc = nlp(text)
            # print(doc)
            # exit()
            text = re.sub(pattern, '', text.strip())
            text_len = len(text)
            if max_length < text_len:
                max_length = text_len
            if text_len < 3:
                continue
            sentence_tokens = list(jieba.cut(text))
            token_nums= len(sentence_tokens)
            if max_token_nums < token_nums:
                max_token_nums = token_nums
            if token_nums < 3:
                continue
            context_left  = sentence_tokens[: token_nums // 2]
            context_right = sentence_tokens[token_nums // 2 :]
            fw.write(''.join(context_left) + '\n')
            fw.write(''.join(context_right) + '\n')
            fw.write('\n')
    fw.close()
    print("max_length={}".format(max_length))   # * 1090
    print("max_token_nums={}".format(max_token_nums)) # * 637
            

if __name__ == "__main__":

#======================merge===================================                                            
    # test_dir = '../data/test'
    # gen_path = '../data/test_merge.txt'
    # merge_dataset(test_dir, gen_path)
#======================get_pretrain============================
   
    data_path =  '../data/test_merge.txt'
    gen_path = '../pretrain/pretrain_data/ie_pretrain.txt'
    gen_pretrain(data_path, gen_path)