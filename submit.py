import torch
import re
import csv
from tqdm import tqdm


def model_predict(config, model, tokenizer, test_iter):
    """
    映射到文件上
    :param config:
    :param model:
    :param tokenizer:
    :param test_iter:
    :return:{ file_id:{
                        'token': [[],[],...],
                        'label': [[],[],...]
                        }
            }
    """
    model.eval()
    file_map = {}   # {id:{'token': [], 'label': []} }
    label_map = {i: label for i, label in enumerate(config.class_list)}

    with torch.no_grad():
        for (file_ids, input_ids, attention_mask, token_type_ids, labels) in tqdm(test_iter):
            file_ids = file_ids.view(-1).numpy()
            input_tensor = torch.tensor(input_ids).type(torch.LongTensor).to(config.device)
            attention_mask_tensor = torch.tensor(attention_mask).type(torch.LongTensor).to(config.device)
            token_type_ids_tensor = torch.tensor(token_type_ids).type(torch.LongTensor).to(config.device)
            labels_tensor = torch.tensor(labels).type(torch.LongTensor).to(config.device)

            outputs, loss = model(input_tensor, attention_mask_tensor, token_type_ids_tensor, labels_tensor)
            predict_out = model.crf.decode(outputs, mask=attention_mask_tensor).squeeze(0)
            assert len(file_ids) == len(input_ids), 'error lens predict'
            assert len(input_ids) == len(predict_out), 'error lens predict'
            for file, ids, out in zip(file_ids, input_ids, predict_out):
                tokens = tokenizer.convert_ids_to_tokens(ids)
                tokens = [tk for tk in tokens if tk != '[PAD]']
                predict_label = [label_map[key] for key in out.cpu().numpy()][:len(tokens)]
                if file not in file_map:
                    file_map[file] = {'token': [tokens], 'label': [predict_label]}
                else:
                    file_map[file]['token'].append(tokens)
                    file_map[file]['label'].append(predict_label)
    return file_map


def extract_entity_mapping_file(file_map, tokenizer):
    """

    :param file_map: {idx: {'token': [[],[]...], 'label': [[],[]...]}, ...}
    :return:
    """
    file_entity = {}
    for file_idx, file_info in file_map.items():
        file_token = []
        file_category = []
        for token_list, label_list in zip(file_info['token'], file_info['label']):
            category = ''
            tokens = []
            for token, label in zip(token_list, label_list):
                if label == 'O':
                    if tokens:
                        tokens = tokenizer.convert_tokens_to_string(tokens)
                        tokens = tokens.split()
                        file_token.append(tokens)
                        file_category.append(category)
                        tokens = []
                        category = ""
                else:
                    pos, ct = label.split('-')
                    if pos == 'B':
                        if tokens:
                            tokens = tokenizer.convert_tokens_to_string(tokens)
                            tokens = tokens.split()
                            file_token.append(tokens)
                            file_category.append(category)
                            tokens = []
                        category = ct
                        tokens.append(token)
                    elif pos == 'I':
                        if ct != category:
                            if tokens:
                                tokens = tokenizer.convert_tokens_to_string(tokens)
                                tokens = tokens.split()
                                file_token.append(tokens)
                                file_category.append(category)
                                tokens = []
                        category = ct
                        tokens.append(token)
            if tokens:
                tokens = tokenizer.convert_tokens_to_string(tokens)
                tokens = tokens.split()
                file_token.append(tokens)
                file_category.append(category)
        file_entity[file_idx] = {
            'token': file_token,
            'category': file_category
        }
    return file_entity


def find_origin_text_entity_pos(origin_data, token_info):
    """

    :param origin_data: dict
    :param token_info: dict, {
                        'file_id': {
                            'token': [],
                            'category': []
                        },
                    }
    :return:
    """
    text_entity_pos = {}
    for key, entity_info in token_info.items():
        origin_text = origin_data[key]
        file_token = entity_info['token']
        file_category = entity_info['category']
        cur_entity_info = []
        pos = 0
        for tokens, category in zip(file_token, file_category):
            if category == 'O':
                continue
            strs = "".join([tk if tk != '[UNK]' else '.' for tk in tokens])
            pattern = re.compile(strs)
            res = pattern.search(origin_text, pos)
            # print(pos, res, origin_data[res[0]: res[1]])
            # res = re.search(strs, origin_text)
            if res:
                start, end = res.span()
                cur_entity_info.append({
                    'category': category,
                    'start_pos': start,
                    'end_pos': end - 1,
                    'privacy': origin_data[start: end]
                })
                pos = end
        text_entity_pos[key] = cur_entity_info
    return text_entity_pos


def source_entity_pos(file_entity, origin_test_data, route):
    """
    :param file_entity {idx: {'token': [[],[]...], 'category': ['', '', ...]}, ...}
    :return predict_data {idx:[(Category, pos_b, pos_e, Privacy),(...), ...], ...}
    """
    csvfile = open(route, "w", encoding='utf-8', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(["ID", "Category", "Pos_b", "Pos_e", "Privacy"])
    entity_list = sorted(file_entity.items(), key=lambda x: x[0])
    for idx, data_dict in entity_list:
        pos = 0
        for token_list, category in zip(data_dict['token'], data_dict['category']):
            if category != 'O':
                origin_data = origin_test_data[idx]
                strs = "".join([tk if tk != '[UNK]' else '.' for tk in token_list])
                strs = re.escape(strs).replace('\\.', '.')
                pattern = re.compile(strs)
                res = pattern.search(origin_data, pos)
                # res = re.search(strs, origin_data)
                if res:
                    start, end = res.span()
                    pos = end
                    writer.writerow([idx, category, start, end-1, origin_data[start:end]])
    csvfile.close()


if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./bert_pretrain_models/ERNIE/vocab.txt')
    file_map = {
        0: {
            'token': [
                ['我', 'va', '##ll', '[UNK]', '国', '是', '1', '4'],
                ['104', '联', '合', '国']
            ],
            'label': [
                ['O', 'O', 'O', 'B-country', 'I-country', 'O', 'O', 'I-count'],
                ['B-age', 'B-company', 'I-company', 'I-person']
            ]
        }
    }
    file_entity = extract_entity_mapping_file(file_map, tokenizer)
    print(file_entity)
