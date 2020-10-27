# coding: UTF-8
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import time
from utils import extract_flat_spans_batch
from models.loss import DiceLoss
from torch.nn.modules import BCEWithLogitsLoss

from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def compute_loss(config, start_logits, end_logits, span_logits,
                 start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
    batch_size, seq_len = start_logits.size()

    start_float_label_mask = start_label_mask.view(-1).float()
    end_float_label_mask = end_label_mask.view(-1).float()
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    match_label_mask = match_label_row_mask & match_label_col_mask
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

    if config.span_loss_candidates == "all":
        # naive mask
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    else:
        # use only pred or golden start/end to compute match loss
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        if config.span_loss_candidates == "gold":
            match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
        else:
            match_candidates = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                 & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )
        match_label_mask = match_label_mask & match_candidates
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    if config.loss_type == "bce":
        bce_loss = BCEWithLogitsLoss(reduction="none")
        start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
    else:
        dice_loss = DiceLoss(with_logits=True, smooth=config.dice_smooth)
        start_loss = dice_loss(start_logits, start_labels.float(), start_float_label_mask)
        end_loss = dice_loss(end_logits, end_labels.float(), end_float_label_mask)
        match_loss = dice_loss(span_logits, match_labels.float(), float_match_label_mask)

    return start_loss, end_loss, match_loss

def model_train(config, model, train_iter, dev_iter):
    start_time = time.time()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         },
    ]
    t_total = len(train_iter) * config.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    )
    #FocalLoss(gamma =2, alpha = 1) #调整gamma=0,1,2,3
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", len(train_iter))
    logger.info("  Dev Num examples = %d", len(dev_iter))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s, id:%d", config.device, config.device_id)

    global_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    total_loss = 0.

    for epoch in range(config.num_train_epochs):
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_train_epochs))
        scheduler.step()  # 学习率衰减
        for i, (_, input_ids, attention_mask, token_type_ids, type_start_labels, type_end_labels,
                start_label_mask, end_label_mask, match_labels, type_) in enumerate(train_iter):
            global_batch += 1
            model.train()

            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(config.device)
            attention_mask = torch.tensor(attention_mask).type(torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(torch.LongTensor).to(config.device)
            type_start_labels = torch.tensor(type_start_labels).type(torch.LongTensor).to(config.device)
            type_end_labels = torch.tensor(type_end_labels).type(torch.LongTensor).to(config.device)
            start_label_mask = torch.tensor(start_label_mask).type(torch.LongTensor).to(config.device)
            end_label_mask = torch.tensor(end_label_mask).type(torch.LongTensor).to(config.device)
            match_labels = torch.tensor(match_labels).type(torch.LongTensor).to(config.device)
            # model output
            start_logits, end_logits, span_logits = model(input_ids, attention_mask, token_type_ids)
            start_loss, end_loss, match_loss = compute_loss(
                config=config,
                start_logits=start_logits,
                end_logits=end_logits,
                span_logits=span_logits,
                start_labels=type_start_labels,
                end_labels=type_end_labels,
                match_labels=match_labels,
                start_label_mask=start_label_mask,
                end_label_mask=end_label_mask
                )
            loss = config.weight_start * start_loss + config.weight_end * end_loss + config.weight_span * match_loss
            model.zero_grad()
            total_loss += loss
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            # [B, length], [B, length], [B, length, length]
            start_preds, end_preds, span_pred = start_logits > 0, end_logits > 0, span_logits>0
            active_labels = extract_flat_spans_batch(start_pred=type_start_labels,
                                                     end_pred=type_end_labels,
                                                     match_pred=match_labels,
                                                     label_mask=start_label_mask,
                                                     pseudo_tag=type_
                                                     )
            predic = extract_flat_spans_batch(start_pred=start_preds,
                                              end_pred=end_preds,
                                              match_pred=span_pred,
                                              label_mask=start_label_mask,
                                              pseudo_tag=type_
                                            )
            labels_all = np.append(labels_all, active_labels)
            predict_all = np.append(predict_all, predic)

            if global_batch % config.output == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true_label = labels_all
                predict_label = predict_all

                train_acc = metrics.accuracy_score(labels_all, predict_all)
                train_precision = precision_score(true_label, predict_label)
                train_recall = recall_score(true_label, predict_label)
                train_f1 = f1_score(true_label, predict_label)
                predict_all = np.array([], dtype=int)
                labels_all = np.array([], dtype=int)

                acc, precision, recall, f1, dev_loss = model_evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = global_batch
                else:
                    improve = ''
                time_dif = time.time() - start_time
                msg = '{0:>6}, Train Loss: {1:>.4f}, train_acc: {2:>.2%}, precision: {3:>.2%}, recall: {4:>.2%}, f1: {5:>.2%}' \
                      ' Val Loss: {6:>5.6f}, acc: {7:>.2%}, precision: {8:>.2%}, recall: {9:>.2%}, f1: {10:>.2%}, ' \
                      ' Time: {11} - {12}'
                logger.info(msg.format(global_batch, total_loss / config.output, train_acc, train_precision, train_recall, train_f1,
                                       dev_loss, acc, precision, recall, f1, time_dif, improve))
                total_loss = 0.

            if config.early_stop and global_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def model_evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, (_, input_ids, attention_mask, token_type_ids, type_start_labels, type_end_labels,
                start_label_mask, end_label_mask, match_labels, type_) in enumerate(data_iter):
            input_ids = torch.tensor(input_ids).type(torch.LongTensor).to(config.device)
            attention_mask = torch.tensor(attention_mask).type(torch.LongTensor).to(config.device)
            token_type_ids = torch.tensor(token_type_ids).type(torch.LongTensor).to(config.device)
            type_start_labels = torch.tensor(type_start_labels).type(torch.LongTensor).to(config.device)
            type_end_labels = torch.tensor(type_end_labels).type(torch.LongTensor).to(config.device)
            start_label_mask = torch.tensor(start_label_mask).type(torch.LongTensor).to(config.device)
            end_label_mask = torch.tensor(end_label_mask).type(torch.LongTensor).to(config.device)
            match_labels = torch.tensor(match_labels).type(torch.LongTensor).to(config.device)
            # model output
            start_logits, end_logits, span_logits = model(input_ids, attention_mask, token_type_ids)
            start_loss, end_loss, match_loss = compute_loss(
                config=config,
                start_logits=start_logits,
                end_logits=end_logits,
                span_logits=span_logits,
                start_labels=type_start_labels,
                end_labels=type_end_labels,
                match_labels=match_labels,
                start_label_mask=start_label_mask,
                end_label_mask=end_label_mask
            )
            loss = config.weight_start * start_loss + config.weight_end * end_loss + config.weight_span * match_loss
            loss_total += loss
            # [B, length], [B, length], [B, length, length]
            start_preds, end_preds, span_pred = start_logits > 0, end_logits > 0, span_logits>0
            active_labels = extract_flat_spans_batch(start_pred=type_start_labels,
                                                     end_pred=type_end_labels,
                                                     match_pred=match_labels,
                                                     label_mask=start_label_mask,
                                                     pseudo_tag=type_
                                                     )
            predic = extract_flat_spans_batch(start_pred=start_preds,
                                              end_pred=end_preds,
                                              match_pred=span_pred,
                                              label_mask=start_label_mask,
                                              pseudo_tag=type_
                                            )
            labels_all = np.append(labels_all, active_labels)
            predict_all = np.append(predict_all, predic)

    true_label = labels_all
    predict_label = predict_all
    acc = metrics.accuracy_score(labels_all, predict_all)
    precision = precision_score(true_label, predict_label)
    recall = recall_score(true_label, predict_label)
    f1 = f1_score(true_label, predict_label)
    if test:
        report = classification_report(true_label, predict_label, digits=4)
        confusion = metrics.confusion_matrix(true_label, predict_label)
        return acc, precision, recall, f1, loss_total / len(data_iter), report, confusion
    return acc, precision, recall, f1, loss_total / len(data_iter)


def model_test(config, model, test_iter):
    # test!
    logger.info("***** Running testing *****")
    logger.info("  Test Num examples = %d", len(test_iter))
    start_time = time.time()
    acc, precision, recall, f1, test_loss, test_report, test_confusion = model_evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.4f},  Test acc: {1:>.2%}, precision: {2:>.2%} recall: {3:>.2%}, f1: {4:>.2%}'
    logger.info(msg.format(test_loss, acc, precision, recall, f1))
    logger.info("Precision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...")
    logger.info(test_confusion)
    time_dif = time.time() - start_time
    logger.info("Time usage:%.6fs", time_dif)
