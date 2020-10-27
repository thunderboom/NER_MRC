import logging
import os
import re
import json
import copy
import csv
import numpy as np
import torch.utils.data as Data
from transformers import BertTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None,
                 file_id=None):
        self.file_id = file_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




class InputFeaturesMRC(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, type_start_labels=None,
                 type_end_labels=None, start_label_mask=None, end_label_mask=None, match_labels=None,
                 query_type=None, file_id=None):
        self.file_id = file_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.type_start_labels = type_start_labels
        self.type_end_labels = type_end_labels
        self.start_label_mask = start_label_mask
        self.end_label_mask = end_label_mask
        self.match_labels = match_labels
        self.query_type = query_type


    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"




class DataProcessor:
    """Processor for the chinese ner data set."""
    def __init__(self, data_dir, max_len=512, split=(0.8, 0.2)):
        self.data_dir = data_dir
        self.train_data_dir = os.path.join(self.data_dir, 'train')
        self.test_data_dir = os.path.join(self.data_dir, 'test')
        self.query_dir = os.path.join(data_dir, 'query.json')
        self.max_len = max_len
        self.category_list = ['O', 'position', 'name', 'organization', 'movie', 'company', 'game', 'book', 'address', 'scene', 'government', 'email', 'mobile', 'qq', 'vx']
        self.total_data = self._read_combined_data(self.train_data_dir, 'train')
        np.random.seed(1024)
        np.random.shuffle(self.total_data)
        self.split = split
        self.train_len = int(len(self.total_data)*split[0])+1
        self.label_category = ['']

    def get_query(self):
        with open(self.query_dir, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
        return data

    def get_train_examples(self):
        """See base class."""
        return self.total_data[:self.train_len]

    def get_dev_examples(self):     # check
        """See base class."""
        return self.total_data[self.train_len:]

    def get_test_examples(self):    # check
        """See base class."""
        return self._read_combined_data(self.test_data_dir, "test")

    def get_origin_test_data(self):
        origin_data = {}
        text_file = os.listdir(self.test_data_dir)
        text_path = [os.path.join(self.test_data_dir, file) for file in text_file]
        for file_name, data_path in zip(text_file, text_path):
            file_idx = int(file_name.split('.')[0])
            text = self._read_text_data(data_path)
            origin_data[file_idx] = text
        return origin_data

    def _labeled_data(self, text, label_list=None, data_type='train'): # 以chunk方式打标
        text_chunk = []
        label_chunk = []
        temp_pos = 0     # 记录子句的起始位置
        if data_type == 'test':
            text_chunk = [text]
            label_chunk = ['O']
        else:  # 训练和验证
            label_list = sorted(label_list, key=lambda x: x[1], reverse=False)  # 根据Label的位置进行排序，无重叠
            for label in label_list:
                (category, former_pos, latter_pos, word) = label
                latter_pos += 1
                # print(text[former_pos: latter_pos], word)
                assert text[former_pos: latter_pos] == word    # 检查位置是否错误
                if former_pos != temp_pos:   # 存在0类子句
                    text_chunk.append(text[temp_pos: former_pos])
                    label_chunk.append('O')
                text_chunk.append(text[former_pos: latter_pos])
                label_chunk.append(category.lower())
                temp_pos = latter_pos
            if temp_pos != len(text) and len(text[temp_pos:])>0:   # 句子结束边界
                text_chunk.append(text[temp_pos:])
                label_chunk.append('O')
        return text_chunk, label_chunk

    def chunk_sentence(self, text: list, label: list):  # 去噪声  # 去重（未做）
        new_text, new_label = [], []
        for sentence_, label_ in zip(text, label):
            if label_ == 'O':
                sentence_ = self._text_clean(sentence_)
            if len(sentence_) > 0:
                new_text.append(sentence_)
                new_label.append(label_)
        return new_text, new_label

    def sentence_combined(self, sentence_list, label_list):    # 输入为一条数据,根据句子长度切分成几部分
        process_text = []
        process_label = []
        combined_sentence = []
        combined_label = []
        sentence_len = 0
        for sentence_, label_ in zip(sentence_list, label_list):
            sentence_len += len(sentence_)      # 如果句子长度即超过子模块相加的长度
            if sentence_len > self.max_len:   # 如果句子长度即超过子模块相加的长度
                temp_sentence_list =  re.split('。|？|！',  sentence_)
                for i in range(len(temp_sentence_list)-1, -1, -1):
                    sentence_len -= len(temp_sentence_list[i])
                    if sentence_len < self.max_len:
                        break
                if i > 0:
                    combined_sentence.append('。'.join(temp_sentence_list[:i]))
                    combined_label.append(label_)
                process_text.append(combined_sentence)
                process_label.append(combined_label)
                combined_sentence = ['。'.join(temp_sentence_list[i:])]
                combined_label = [label_]
                sentence_len = len('。'.join(temp_sentence_list[i:]))
            else:
                combined_sentence.append(sentence_)
                combined_label.append(label_)
        if len(combined_sentence) > 0:  # 在最后位置加入
            process_text.append(combined_sentence)
            process_label.append(combined_label)
        return process_text, process_label

    def pre_processing(self, text: list, label: list): # [text:]
        """clean data & length handle"""
        sentence_list, label_list = self.chunk_sentence(text, label)       # 清洗+去重
        process_text, process_label = self.sentence_combined(sentence_list, label_list)  # 处理长度
        return process_text, process_label

    @classmethod
    def _read_text_data(self, input_file):  # get unlabeled text
        text = ""
        with open(input_file, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                if len(line.strip()) > 0:
                    # line = line.replace('\n', '|')
                    text += str(line)
        return text.strip('\n')

    @classmethod
    def _read_label_data(self, input_file):  # get label
        label = []
        with open(input_file, 'r', encoding='utf-8') as fr:
            csv_list = list(csv.reader(fr, delimiter=','))
            for line in csv_list[1:]:
                category = line[1].strip()
                pos_start = int(line[2].strip())
                pos_end = int(line[3].strip())
                privacy = line[4].strip()
                label.append((category, pos_start, pos_end, privacy))
        return label

    def _read_combined_data(self, data_dir, data_type):  # tagging, process and cut_sentence
        
        examples = []
        if data_type == 'train':
            text_data_dir = os.path.join(data_dir, 'data')
            label_data_dir = os.path.join(data_dir, 'label')
        else:
            text_data_dir = data_dir
            label_data_dir = None
        text_file = os.listdir(text_data_dir)
        text_path = [os.path.join(text_data_dir, file) for file in text_file]
        for file_name, data_path in zip(text_file, text_path):
            # print(file_name)
            text = self._read_text_data(data_path)
            if data_type == 'train':
                label_file = file_name.replace('txt', 'csv')
                label_path = os.path.join(label_data_dir, label_file)
                label_list = self._read_label_data(label_path)
                text_chunk, label_chunk = self._labeled_data(text, label_list)    # 根据字句进行切分
                for category in label_chunk:
                    if category not in self.category_list:
                        self.category_list.append(category.lower())
            else:  # 生成fake label
                text_chunk, label_chunk = self._labeled_data(text, data_type='test')
            process_text, process_label = self.pre_processing(text_chunk, label_chunk)  # 数据预处理
            for sentence_, label_ in zip(process_text, process_label):
                examples.append(InputExample(guid=data_type + '-' + str(file_name).split('.')[0], text_a=sentence_, label=label_))
        return examples

    def get_labels(self):
        label_list = ['O']
        for category in self.category_list:
            if category != 'O':
                label_list.extend(['B-'+category, 'I-'+category])
        return label_list

    def _text_clean(self, text):
        additional_chars = ['\ue404', '６', '"', '★', '＠', '∩', '┃', '※', '「', '\u202a', '｣', 'ค', '\u202c', 
                    '｜', '\ue253', '﹕', 'の', '세', '－', 'た', '］', '﹐', 'に', ' 『', 'ช', '＋', '≫',
                        'ะ', 'ｅ', 'ら', 'ス', '＜', '＂', '＞', '①', '｢', '\n', '』', 'ト', 'れ', '〉', '↑', 
                        '～', 'は', 'Ｔ', 'ｌ', 'で', '」', '﹔', '②', 'ち', '＼', 'い', 'る', '０', '\ue449', 
                        'な', '→', '・', '□', '◆', '●', 'く', '안', '×', 'ｍ', '＄', '〃', '녕', 'あ', 'を', 
                        'み', 'タ', '\ue316', '〜', 'ω', '≪', '〈', '•', '﹗', 'ต', '\u2006', "'", '하', 'า', 
                        'て', 'ん', '\ue056', '♪', '요', 'ニ', 'し', '⋯', 'โ', '↓', 'や', '\uf87d', '⊙', 'っ', 
                        'が', '\u2029', 'ー', 'コ', '\ue41d', '☆', 'ｓ', 'と', '—', '☏', 'ま', '＆', '／', '［', 'え',
                        'す', '∶', '﹣', '─', 'こ', '■', '…', '\t', '\u3000']
        # extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】")  
        text = text.strip()
        text = re.sub('(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text)  # 过滤网址
        text = re.sub('www.[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', text).replace("()", "")  
        text = re.sub('\s', '', text) 
        for wbad in additional_chars:  
            text = text.replace(wbad, '') 
        # pattern = re.compile(r'[\u3000\s]')
        # text = re.sub(pattern, '', text)
        return text


def convert_examples_to_features_for_ner(
    examples,
    tokenizer,
    label_list,
    max_seq_length=256,
    special_tokens_count=0,
    pad_token=0,
    sequence_a_segment_id=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100
):
    label_map = {label: i for i, label in enumerate(label_list)}
    print("label_map",label_map)
    pad_token_label_id = label_map['O']
    features = []
    for ex_index, example in enumerate(examples):
        tokens = []
        label_ids = []
        for sentence, label in zip(example.text_a, example.label):
            print("label", label)
            sentence_tokens = tokenizer.tokenize(sentence)
            if len(sentence_tokens) == 0:
                continue
            if label in ['O']:
                token_labels = ['O'] * len(sentence_tokens)
            else:
                token_labels = ['B-' + label.lower()] + ['I-' + label.lower()] * (len(sentence_tokens)-1)
            tokens.extend(sentence_tokens)
            label_ids.extend([label_map[ls] for ls in token_labels])

        if len(tokens) == 0:
            continue

        if len(tokens) > max_seq_length - special_tokens_count:  # 进行截断
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # tokens = [cls_token] + tokens + [sep_token]
        # label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
        # segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        segment_ids = [sequence_a_segment_id] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # padding
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask,
                          token_type_ids=segment_ids, label=label_ids, file_id=[int(str(example.guid).split('-')[1])])
        )
    return features

def convert_examples_to_features_for_MRC(
    examples,
    tokenizer,
    max_seq_length=512,
    special_tokens_count=0,
    pad_token=0,
    sequence_a_segment_id=0,
    pad_token_segment_id=0,
    pad_token_label_id=0,
    query_type=None
):
    features = []
    for ex_index, example in tqdm(enumerate(examples)):
        tokens = []
        example_label = []
        current_pos = 0
        for sentence, label in zip(example.text_a, example.label):
            sentence_tokens = tokenizer.tokenize(sentence)
            if len(sentence_tokens) == 0:
                continue
            if label != 'O':
                start_position = current_pos
                end_position = current_pos + len(sentence_tokens) - 1     # 真实的position位置
                example_label.append({'start_position': start_position, 'end_position': end_position, 'label': label.lower()})
            tokens.extend(sentence_tokens)
            current_pos += len(sentence_tokens)
        for type_ in query_type.keys():      # 每个类别构造query & label
            query = query_type[type_]
            query_tokens = tokenizer.tokenize(query)
            combined_tokens = ['[CLS]'] + tokens + ['[SEP]'] + query_tokens + ['[SEP]']
            combined_len = len(combined_tokens)
            if combined_len > max_seq_length:  # 对content的token进行截断
                tokens = tokens[: max_seq_length-combined_len]
                combined_tokens = ['[CLS]'] + tokens + ['[SEP]'] + query_tokens + ['[SEP]']
            type_example_start, type_example_end = [], []
            for item in example_label:
                if item['label'] == type_:
                    if item['end_position'] + 1 < len(tokens):    # 对label截断
                        type_example_start.append(item['start_position']+1)   #加入了CLS标签
                        type_example_end.append(item['end_position']+1)     # #加入了CLS标签
            type_start_labels = [(1 if idx in type_example_start else 0)
                                 for idx in range(len(combined_tokens))]
            type_end_labels = [(1 if idx in type_example_end else 0)
                                 for idx in range(len(combined_tokens))]
            segment_ids = [sequence_a_segment_id] * (len(tokens)+1) + [1] * (len(query_tokens) + 2)
            assert len(segment_ids) == len(combined_tokens)
            label_mask = [     # label mask: content为1， query和特殊标签为0
                (1 if segment_ids[token_idx] == 0 and token_idx != 0 else 0)    # 起始位置
                for token_idx in range(len(combined_tokens))
            ]
            start_label_mask = label_mask.copy()
            end_label_mask = label_mask.copy()
            assert all(start_label_mask[p] != 0 for p in type_example_start)
            assert all(end_label_mask[p] != 0 for p in type_example_end)

            # convert token to id
            input_ids = tokenizer.convert_tokens_to_ids(combined_tokens)
            input_mask = [1] * len(input_ids)
            # padding
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            type_start_labels += [pad_token_label_id] * padding_length
            type_end_labels += [pad_token_label_id] * padding_length
            start_label_mask += [pad_token_label_id] * padding_length
            end_label_mask += [pad_token_label_id] * padding_length

            # 构造match_labels 标签， [seq_len, seq_len]
            seq_len = len(tokens)
            match_labels = np.zeros([seq_len, seq_len], dtype=np.long)
            for start, end in zip(type_example_start, type_example_end):
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(type_start_labels) == max_seq_length
            assert len(type_end_labels) == max_seq_length
            assert len(start_label_mask) == max_seq_length
            assert len(end_label_mask) == max_seq_length

            if ex_index < 1:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("type_start_labels: %s", " ".join([str(x) for x in type_start_labels]))
                logger.info("type_end_labels: %s", " ".join([str(x) for x in type_end_labels]))
                logger.info("start_label_mask: %s", " ".join([str(x) for x in start_label_mask]))
                logger.info("end_label_mask: %s", " ".join([str(x) for x in end_label_mask]))
                logger.info("label type:{}".format(type_))
                logger.info("match_labels: {}".format(match_labels))

            features.append(
                InputFeaturesMRC(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                                 type_start_labels=type_start_labels, type_end_labels=type_end_labels,
                                 start_label_mask=start_label_mask, end_label_mask=end_label_mask,
                                 match_labels=match_labels, query_type=type_,
                                 file_id=None)
                )
    return features


class BuildDataSet(Data.Dataset):
    """
    将经过convert_examples_to_features的数据 包装成 Dataset
    """
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        feature = self.features[index]
        file_ids = np.array(feature.file_id)
        input_ids = np.array(feature.input_ids)
        attention_mask = np.array(feature.attention_mask)
        token_type_ids = np.array(feature.token_type_ids)
        type_start_labels = np.array(feature.type_start_labels)
        type_end_labels = np.array(feature.type_end_labels)
        start_label_mask = np.array(feature.start_label_mask)
        end_label_mask = np.array(feature.end_label_mask)
        match_labels = np.array(feature.match_labels)
        type_ = feature.query_type
        return file_ids, input_ids, attention_mask, token_type_ids, \
               type_start_labels, type_end_labels, start_label_mask, end_label_mask, match_labels, type_

    def __len__(self):
        return len(self.features)

# test
if __name__ == "__main__":
    # query_type = {'game': '游戏，娱乐的方式',
    #  'company': '公司，企业的组织形式',
    #  'government': ' 政府；政体；管辖',
    #  'book': '书籍',
    #  'address': '地点、地址、所在地',
    #  'email': '邮箱',
    #  'movie': '电影',
    #  'position': '职位、位置',
    #  'name': '姓名、中英文名',
    #  'scene': '景点、自然景观、建筑物等',
    #  'mobile': '手机号',
    #  'organization': '组织、机构、体制、团体 ',
    #  'wx': '微信号',
    #  'QQ': 'qq号',
    # }
    # query_type = json.dumps(query_type)
    # with open('./data/query.json', 'w') as fw:
    #     fw.write(query_type)
    # raise ValueError()
    processor = DataProcessor('./data/', max_len=110)
    processor.get_test_examples()
    # test_data = processor.get_origin_test_data()
    # print(test_data)
    dev_data = processor.get_train_examples()
    for example in dev_data[:2]:
        print(example.text_a)
        print(example.label)
    print(processor.category_list)
    query_type = processor.get_query()
    print(query_type)
    tokenizer_path = './former/pretrained_model/ERNIE/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    examples = convert_examples_to_features_for_MRC(dev_data[:2], tokenizer=tokenizer, query_type=query_type, max_seq_length=128)
    print(len(examples))
    for example in examples:
        print(example.input_ids)
        print(example.end_label_mask)
        print(sum(example.type_start_labels))
        print(example.type_start_labels)
        print(example.type_end_labels)
        print(sum(example.type_end_labels))
        print(example.query_type)

