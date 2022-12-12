import random
import sys

import pandas as pd
import h5py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BertTokenizer

from text_processor import read_examples, convert_examples_to_features

MAX_PASSAGE_LEN = 400
MAX_TEXT_LEN = 50


def save_hdf5(all_chars, all_words, all_char_masks, all_word_masks, all_annotation, idx, save_name):
    file = h5py.File(save_name, "w")
    chars, char_masks = all_chars[idx], all_char_masks[idx]
    words, word_masks = all_words[idx], all_word_masks[idx]
    file.create_dataset("char_id", data=np.array(chars).astype(np.int32))
    file.create_dataset("word_id", data=np.array(words).astype(np.int32))
    file.create_dataset("char_mask", data=np.array(char_masks).astype(np.int32))
    file.create_dataset("word_mask", data=np.array(word_masks).astype(np.int32))
    for key, value in all_annotation.items():
        file.create_dataset(key, data=np.array(value[idx]).astype(np.int32))
    file.create_dataset("idx", data=np.array(idx).astype(np.int))
    file.attrs['size'] = len(idx)
    file.close()


def generate_char_tokenizer(tokenizer):
    def char_tokenizer(text):
        examples = read_examples(text, 0)
        features = convert_examples_to_features(examples=examples, seq_length=MAX_TEXT_LEN,
                                                tokenizer=tokenizer)
        char_id = np.array(features[0].input_ids).astype(np.int32)
        char_mask = np.array(features[0].input_mask).astype(np.int32)
        char_len = char_mask.sum()
        return char_id, char_mask, char_len
    return char_tokenizer

def generate_word_tokenizer(path):
    model = SentenceTransformer(path, device='cuda:0')
    model.eval()
    def word_tokenizer(text):
        result = model.tokenize([text])
        word_id = zero_pad(result["input_ids"].squeeze(0).long().cpu().numpy(), MAX_TEXT_LEN)
        mask = zero_pad(result["attention_mask"].squeeze(0).long().cpu().numpy(), MAX_TEXT_LEN)
        word_len = mask.sum()
        return word_id, mask, word_len
    return word_tokenizer


def transform_font_size(font_size):
    if font_size == -1:
        return 0
    if font_size == "large":
        return 1
    real_size = int(font_size[:-2])
    if real_size == 18:
        return 0
    elif real_size > 18:
        return 1
    else:
        return 2

def transform_styles(tag):
    color, bg_color, supertalk = tag["color"], tag["background-color"], tag["supertalk"]
    strong, small_title, h4 = tag["strong"], tag["sns-small-title"], tag["h4"]
    blockquote = tag["blockquote"]
    return [int((color != -1) or (bg_color != -1)), int(supertalk != 0),
            int(strong != 0), int((small_title != 0) or (h4 != 0)), int(blockquote != 0)]
    # [color (fg or bg), supertalk, strong, title, blockquote]

def transform_tags(tags):
    para_order = []
    sent_order = []
    style_mark = []  # [Color, BG-Color, Supertalk, Strong, Blockquote, sns-small-title, H4]
    font_size = []
    for tag in tags:
        para_order.append(tag["po"])
        sent_order.append(tag["pi"])
        font_size.append(transform_font_size(tag["font-size"]))
        style_mark.append(transform_styles(tag))
    return [np.array(para_order), np.array(sent_order),
            np.array(font_size), np.array(style_mark)]

def preprocess_claims(major, claims, premises):
    """
    数据处理逻辑：
        主论点被视为一个独立论点，如果主论点和子论点完全一致，则把主论点和子论点合并；如果主论点和子论点不完全一致，新建一个新的论点
    """
    major_idx = -1
    for idx, claim in enumerate(claims):
        if claim == major:
            major_idx = idx
            break
    if major_idx == -1:
        claims = [major, *claims]
        major_idx = 0
        premises = [[[], [], [], []], *premises]
    placeholder = np.zeros(MAX_PASSAGE_LEN)
    sorted_indices = np.array([len(claim) for claim in claims]).argsort()
    unique_claims = [[] for _ in range(len(claims))]
    valid_claims = []
    valid_premises = []
    # generate unique results
    for sorted_idx in sorted_indices:
        current = []
        for sentence_idx in claims[sorted_idx]:
            if placeholder[sentence_idx] == 1:
                continue
            placeholder[sentence_idx] = 1
            current.append(sentence_idx)
        current.sort()
        unique_claims[sorted_idx] = current
    for claim_idx, unique_claim in enumerate(unique_claims):
        if len(unique_claim) == 0:
            assert major_idx != claim_idx
        else:
            valid_claims.append(unique_claim)
            valid_premises.append(premises[claim_idx])
    return valid_claims, valid_premises, major_idx


def transform_trgs(trgs, max_len):
    trg_dict = trgs['results']
    claim_order = [-1] * max_len
    premise_order = [-1] * max_len
    premise_relation = [-1] * max_len
    max_claim_num, max_premise_num = 8, 4
    major_claim = trg_dict["MajorClaim"]
    assert len(major_claim) > 0, "Invalid Passage!"
    claims = [trg_dict["Claim_{}".format(i)] for i in range(1, max_claim_num + 1)]
    premises = [[trg_dict["Premise_{}_{}".format(i, j)]
                 for j in range(1, max_premise_num + 1)] for i in range(1, max_claim_num + 1)]
    # Scan for claims
    claims, premises, major_idx = preprocess_claims(major_claim, claims, premises)
    for claim_idx in range(len(claims)):
        claim = claims[claim_idx]
        for sent_idx in claim:
            claim_order[sent_idx] = claim_idx

    # Check Claim Sanity
    # cnt_max = 0
    # for sent_idx in range(len(claim_order)):
    #     assert claim_order[sent_idx] == -1 or cnt_max <= claim_order[sent_idx]
    #     cnt_max = max(cnt_max, claim_order[sent_idx])

    # Scan for Premise
    for claim_idx in range(len(claims)):
        for premise_idx in range(max_premise_num):
            premise = premises[claim_idx]
            premise = premise[premise_idx]
            for sent_idx in premise:
                if claim_order[sent_idx] != -1:
                    continue
                if premise_order[sent_idx] != -1 or premise_order[sent_idx] != -1:
                    continue
                premise_relation[sent_idx] = claim_idx
                premise_order[sent_idx] = premise_idx

    # Check Premise Sanity
    # cnt_max = 0
    # for sent_idx in range(len(premise_order)):
    #     assert premise_order[sent_idx] == -1 or cnt_max <= premise_order[sent_idx]
    #     cnt_max = max(cnt_max, premise_order[sent_idx])

    return major_idx, np.array(claim_order), np.array(premise_order), np.array(premise_relation)
    # -1: Not a premise, > 0: the corresponding claim id

def raw2sentence(raw_dict):
    sentence = []
    for key, val in raw_dict.items():
        sentence.append(val)
    return sentence

def zero_pad(data, max_len):
    if len(data.shape) == 1:
        content = np.zeros(max_len).astype(data.dtype)
    elif len(data.shape) == 2:
        content = np.zeros((max_len, data.shape[-1])).astype(data.dtype)
    else:
        raise NotImplementedError
    length = min(len(data), max_len)
    content[:length] = data[:length]
    return content

def annotation_transform(csv_paths, model):
    result = {
        "para_order": [], "sent_order": [], "font_size": [], "style_mark": [],
        "major_idx": [], "claim_order": [], "premise_order": [], "target_relation": [],
        "reflection": [], "passage_mask": []
    }
    deserted_idx = []
    all_idx = -1
    for csv_path in csv_paths:
        csv_file = pd.read_csv(csv_path)
        # srcs, sents, tags, trgs
        for row_idx in tqdm(range(len(csv_file))):
            all_idx += 1
            row = csv_file.loc[row_idx]
            sentences, tags, trgs = raw2sentence(eval(row[1])), eval(row[2]), eval(row[3])
            if len(tags) > MAX_PASSAGE_LEN:
                print(f"Too Long Passage for Row {row_idx}")
                continue
            try:
                para_order, sent_order, font_size, style_mark = transform_tags(tags)
                major_idx, claim_order, premise_order, premise_relation = transform_trgs(trgs, len(tags))
            except AssertionError as e:
                print(f"Major claim gets destroyed for Row {row_idx}")
                continue
            labels = 1 * (claim_order != -1) + 2 * (premise_order != -1)  # 0: others, 1: claim, 2: premise
            for sentence in sentences:


            passage_mask = np.zeros((MAX_PASSAGE_LEN))
            passage_mask[:len(para_order)] = 1
            result["passage_mask"].append(passage_mask)
            result["para_order"].append(zero_pad(para_order, MAX_PASSAGE_LEN))
            result["sent_order"].append(zero_pad(sent_order, MAX_PASSAGE_LEN))
            result["font_size"].append(zero_pad(font_size, MAX_PASSAGE_LEN))
            result["style_mark"].append(zero_pad(style_mark, MAX_PASSAGE_LEN))
            result["major_idx"].append(major_idx)
            result["reflection"].append(all_idx)
            result["claim_order"].append(zero_pad(claim_order, MAX_PASSAGE_LEN))
            result["premise_order"].append(zero_pad(premise_order, MAX_PASSAGE_LEN))
            result["target_relation"].append(zero_pad(premise_relation, MAX_PASSAGE_LEN))
    for key, value in result.items():
        result[key] = np.array(value)
    return result, deserted_idx


# deserted_idx = []
results = annotation_transform(["./raw_data/train.collect.csv",
                                "./raw_data/dev.collect.csv",
                                "./raw_data/test.collect.csv"])

char_path = "pretrained_model/FinBERT_L-12_H-768_A-12_pytorch/vocab.txt"
word_path = 'pretrained_model/paraphrase-xlm-r-multilingual-v1'
char_base = BertTokenizer.from_pretrained(char_path, do_lower_case=True)

char_tokenizer = generate_char_tokenizer(char_base)
word_tokenizer = generate_word_tokenizer(word_path)


# train: 8000, 991, 1000
save_hdf5(char_id, word_id, char_mask, word_mask, annotations, train_idx, "train_data.hdf5")
save_hdf5(char_id, word_id, char_mask, word_mask, annotations, val_idx, "val_data.hdf5")
save_hdf5(char_id, word_id, char_mask, word_mask, annotations, test_idx, "test_data.hdf5")