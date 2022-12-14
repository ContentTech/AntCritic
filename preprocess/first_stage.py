import h5py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BertTokenizer

from utils.text_processor import read_examples, convert_examples_to_features

MAX_PASSAGE_LEN = 400
MAX_TEXT_LEN = 50

char_path = "pretrained_model/FinBERT_L-12_H-768_A-12_pytorch/vocab.txt"
word_path = 'pretrained_model/paraphrase-xlm-r-multilingual-v1'
char_base = BertTokenizer.from_pretrained(char_path, do_lower_case=True)


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


char_tokenizer = generate_char_tokenizer(char_base)
word_tokenizer = generate_word_tokenizer(word_path)

print("0:", word_tokenizer("mask"))
print("1:", word_tokenizer("<mask>"))
print("2:", word_tokenizer("mask_token"))
print("3:", word_tokenizer("ashaihdsufg"))
exit(0)

def raw2sentence(raw_dict):
    sentence = []
    for key, val in raw_dict.items():
        sentence.append(val)
    return sentence

def preprocess_claims(major, claims, premises):
    """
    ?????????????????????
        ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
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

    # Scan for Premise
    for claim_idx in range(len(claims)):
        for premise_idx in range(max_premise_num):
            premise = premises[claim_idx]
            premise = premise[premise_idx]
            for sent_idx in premise:
                if claim_order[sent_idx] != -1 or premise_order[sent_idx] != -1:
                    continue
                premise_order[sent_idx] = premise_idx

    return major_idx, np.array(claim_order), np.array(premise_order)
    # -1: Not a premise, > 0: the corresponding claim id

def idx2set(idx):
    if idx < 8000:
        cnt = "train"
    elif 8000 <= idx < 9000:
        cnt = "test"
    else:
        cnt = "val"
    return cnt

def annotation_transform(csv_paths):
    results = {
        "train": {"char_id": [], "char_mask": [], "word_id": [], "word_mask": [], "label": []},
        "val": {"char_id": [], "char_mask": [], "word_id": [], "word_mask": [], "label": []},
        "test": {"char_id": [], "char_mask": [], "word_id": [], "word_mask": [], "label": []}
    }
    lengths = {"train": 0, "val": 0, "test": 0}
    all_idx = -1
    for csv_path in csv_paths:
        csv_file = pd.read_csv(csv_path)
        for row_idx in tqdm(range(len(csv_file))):
            all_idx += 1
            row = csv_file.loc[row_idx]
            sentences, tags, trgs = raw2sentence(eval(row[1])), eval(row[2]), eval(row[3])
            if len(tags) > MAX_PASSAGE_LEN:
                print(f"Too Long Passage for Row {row_idx}")
                continue
            try:
                major_idx, claim_order, premise_order = transform_trgs(trgs, len(tags))
            except AssertionError as e:
                print(f"Major claim gets destroyed for Row {row_idx}")
                continue
            labels = 1 * (claim_order != -1) + 2 * (premise_order != -1)  # 0: others, 1: claim, 2: premise
            for idx, sentence in enumerate(sentences):
                word_id, word_mask, word_len = word_tokenizer(sentence)
                char_id, char_mask, char_len = char_tokenizer(sentence)
                set_name = idx2set(all_idx)
                lengths[set_name] += 1
                results[set_name]["char_id"].append(char_id)
                results[set_name]["char_mask"].append(char_mask)
                results[set_name]["word_id"].append(word_id)
                results[set_name]["word_mask"].append(word_mask)
                results[set_name]["label"].append(labels[idx])
    return results, lengths


data, lengths = annotation_transform(["./raw_data/train.collect.csv",
                                    "./raw_data/dev.collect.csv",
                                    "./raw_data/test.collect.csv"])

for set_key in data.keys():
    file = h5py.File(set_key + "_1.hdf5", "w")
    for input_key in data[set_key].keys():
        file.create_dataset(input_key, data=np.array(data[set_key][input_key]).astype(np.int32))
    file.attrs["size"] = lengths[set_key]
    file.close()

