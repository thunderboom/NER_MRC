import logging
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from DataProcessor import DataProcessor, convert_examples_to_features_for_MRC, BuildDataSet
from utils import random_seed, config_to_json_string, model_save, model_load
from submit import model_predict
from transformers import BertTokenizer
from submit import extract_entity_mapping_file, source_entity_pos
from models.bertmrc import BertQueryNER
from train_eval_for_mrc import model_train, model_test
import time

logger = logging.getLogger(__name__)


class Config:

    def __init__(self):
        absdir = os.path.dirname(os.path.abspath(__file__))
        _pretrain_path = '/bert_pretrain_models/ERNIE'
        _config_file = 'config.json'
        _model_file = 'pytorch_model.bin'
        _tokenizer_file = 'vocab.txt'
        _data_path = '/data/'

        self.task = 'bert_mrc'
        self.config_file = os.path.join(absdir + _pretrain_path, _config_file)
        self.model_name_or_path = os.path.join(absdir + _pretrain_path, _model_file)
        self.tokenizer_file = os.path.join(absdir + _pretrain_path, _tokenizer_file)
        self.data_dir = absdir + _data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')              # 设备
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.device_id = 2
        self.do_lower_case = True
        self.label_on_test_set = True
        self.requires_grad = False
        self.class_list = []
        self.num_labels = 0
        self.train_num_examples = 0
        self.dev_num_examples = 0
        self.test_num_examples = 0
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.early_stop = False
        self.require_improvement = 2000                                                         # 若超过1000batch效果还没提升，则提前结束训练
        self.num_train_epochs = 10                                                               # epoch数
        self.batch_size = 16                                                                     # mini-batch大小
        self.pad_size = 256                                                       # 每句话处理成的长度
        self.learning_rate = 3e-5                                                               # 学习率
        self.crf_learning_rate = 0.01                                                            # crf学习率
        self.weight_decay = 0.01                                                                # 权重衰减因子
        self.warmup_proportion = 0.1                                                            # Proportion of training to perform linear learning rate warmup for.
        self.output = 50
        # logging
        self.is_logging2file = True
        self.logging_dir = absdir + '/logging' + '/' + self.task
        # save
        self.save_path = absdir + '/model_saved'
        self.do_train = True
        self.do_predict = False

        # loss
        self.weight_start = 1.0
        self.weight_end = 1.0
        self.weight_span = 1.0   # 0.1
        self.loss_type == 'bce'
        self.span_loss_candidates = 'pred_and_gold'
        self.dice_smooth = 1e-8
        self.query_length = 15

def Task(config):

    if config.device.type == 'cuda':
        torch.cuda.set_device(config.device_id)

    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file,
                                              do_lower_case=config.do_lower_case)
    processor = DataProcessor(config.data_dir, max_len=config.pad_size-config.query_length)
    config.class_list = processor.get_labels()
    config.num_labels = len(config.class_list)

    train_examples = processor.get_train_examples()
    config.train_num_examples = len(train_examples)

    dev_examples = processor.get_dev_examples()
    config.dev_num_examples = len(dev_examples)

    test_examples = processor.get_test_examples()
    config.test_num_examples = len(test_examples)

    bert_model = BertQueryNER(config).to(config.device)
    logger.info("self config: {}\n".format(config_to_json_string(config)))
    if config.do_train:
        train_features = convert_examples_to_features_for_MRC(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=config.pad_size,
            pad_token_label_id=config.pad_token_label_id,
            query_type=processor.get_query()
        )
        dev_features = convert_examples_to_features_for_MRC(
            examples=dev_examples,
            tokenizer=tokenizer,
            max_seq_length=config.pad_size,
            pad_token_label_id=config.pad_token_label_id,
            query_type=processor.get_query()
        )

        train_dataset = BuildDataSet(train_features)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        dev_dataset = BuildDataSet(dev_features)
        dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
        model_train(config, bert_model, train_loader, dev_loader)
        model_test(config, bert_model, dev_loader)
        model_save(bert_model, config.save_path, config.task)

    if config.do_predict:
        model_load(bert_model, config.save_path, config.task, config.device.type, config.device_id)
        test_features = convert_examples_to_features_for_MRC(
             examples=test_examples,
             tokenizer=tokenizer,
             max_seq_length=config.pad_size,
             pad_token_label_id=config.pad_token_label_id
        )

        test_dataset = BuildDataSet(test_features)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        origin_test_data = processor.get_origin_test_data()
        predict_dict = model_predict(config, bert_model, tokenizer, test_loader)
        file_entity = extract_entity_mapping_file(predict_dict, tokenizer)
        source_entity_pos(file_entity, origin_test_data, config.predict_dir) / ';;;;;;;;;;;/    '

if __name__ == '__main__':

    config = Config()
    logging_filename = None
    if config.is_logging2file is True:
        file = time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        logging_filename = os.path.join(config.logging_dir, file)
        if not os.path.exists(config.logging_dir):
            os.makedirs(config.logging_dir)

    logging.basicConfig(filename=logging_filename, format='%(levelname)s: %(message)s', level=logging.INFO)
    random_seed(2020)
    Task(config)


