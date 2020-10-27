import copy
import json
import logging
import numpy as np
import torch.utils.data as Data
import random
import torch
import os


logger = logging.getLogger(__name__)


def config_to_dict(config):

    output = copy.deepcopy(config.__dict__)
    if hasattr(config.__class__, "model_type"):
        output["model_type"] = config.__class__.model_type
    output['device'] = config.device.type
    return output


def config_to_json_string(config):
    """Serializes this instance to a JSON string."""
    return json.dumps(config_to_dict(config), indent=2, sort_keys=True) + '\n'


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_save(model, path, file_name):

    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, file_name+'.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved.")


def model_load(model, path, file_name, device='cpu', device_id=0):
    file_name = os.path.join(path, file_name+'.pkl')
    model.load_state_dict(torch.load(file_name,
                                     map_location=device if device == 'cpu' else "{}:{}".format(device, device_id)))


def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, pseudo_tag):
    """
    Extract flat-ner spans from start/end/match logits
    Args:
        start_pred: [seq_len], 1/True for start, 0/False for non-start
        end_pred: [seq_len, 2], 1/True for end, 0/False for non-end
        match_pred: [seq_len, seq_len], 1/True for match, 0/False for non-match
        label_mask: [seq_len], 1 for valid boundary.
    Returns:
        tags: list of BIO label
    Examples:
        >>> start_pred = [0, 1]
        >>> end_pred = [0, 1]
        >>> match_pred = [[0, 0], [0, 1]]
        >>> label_mask = [1, 1]
        >>> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [0, B-X]
    """
    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"I-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"I-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"B-{pseudo_tag}"
    return bmes_labels

def extract_flat_spans_batch(start_pred, end_pred, match_pred, label_mask, pseudo_tag):
    batch_label = []
    B, length = start_pred.size()
    for i in range(B):
        temp_start_pred, temp_end_pred, temp_match_pred, temp_label_mask, temp_pseudo_tag = \
        start_pred[i, :], end_pred[i, :], match_pred[i, :, :], label_mask[i, :], pseudo_tag[i]
        temp_bio_label = extract_flat_spans(
            temp_start_pred,
            temp_end_pred,
            temp_match_pred,
            temp_label_mask,
            temp_pseudo_tag
        )
        batch_label.append(temp_bio_label)
    return batch_label