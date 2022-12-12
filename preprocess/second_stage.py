import copy
import random
import sys
import os
BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(BASEDIR)
import pandas as pd
import h5py
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BertTokenizer
import os
from models import FirstStageModel
from utils.text_processor import read_examples, convert_examples_to_features
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'torch_device: {torch_device}')

MAX_PASSAGE_LEN = 400
MAX_TEXT_LEN = 50

# 1. TODO: Component Load*
def generate_char_tokenizer(tokenizer):
    def char_tokenizer(text):
        examples = read_examples(text, 0)
        features = convert_examples_to_features(examples=examples, seq_length=MAX_TEXT_LEN,
                                                tokenizer=tokenizer)
        char_id = torch.from_numpy(np.array(features[0].input_ids).astype(np.int64)).to(torch_device)
        char_mask = torch.from_numpy(np.array(features[0].input_mask).astype(np.int64)).to(torch_device)
        char_len = char_mask.sum()
        return char_id, char_mask, char_len
    return char_tokenizer

def generate_word_tokenizer(path):
    model = SentenceTransformer(path, device=torch_device)
    model.eval()
    def word_tokenizer(text):
        result = model.tokenize([text])
        word_id = torch.from_numpy(zero_pad(result["input_ids"].squeeze(0).long().cpu().numpy(), MAX_TEXT_LEN)).to(torch_device)
        mask = torch.from_numpy(zero_pad(result["attention_mask"].squeeze(0).long().cpu().numpy(), MAX_TEXT_LEN)).to(torch_device)
        word_len = mask.sum()
        return word_id, mask, word_len
    return word_tokenizer

def load_model(pretrained_path, checkpoint_path, use_word):
    if not use_word:
        model = FirstStageModel({"char_path": pretrained_path}, use_word=False).to(torch_device)
    else:
        model = FirstStageModel({"word_path": pretrained_path}, use_word=True).to(torch_device)
    state_dict = torch.load(checkpoint_path, map_location=torch.device(torch_device))
    parameters = state_dict['model_parameters']
    model.load_state_dict(parameters)
    model.eval()
    return model

def generate_models(char_path, word_path, char_pretrained, word_pretrained):
    char_model = load_model(char_pretrained, char_path, use_word=False)
    word_model = load_model(word_pretrained, word_path, use_word=True)
    return char_model, word_model


# 2. TODO: Transform Tags and Targets

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
    return [np.array(para_order).astype(np.int32), np.array(sent_order).astype(np.int32),
            np.array(font_size).astype(np.int32), np.array(style_mark).astype(np.int32)]

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


def order2grid(major_idx, claim_order, premise_order, premise_relation, max_len):

    # def is_major(sent_idx):
    #     return claim_order[sent_idx] == major_idx

    def is_claim(sent_idx):
        # return claim_order[sent_idx] != major_idx and claim_order[sent_idx] != -1
        return claim_order[sent_idx] != -1

    def is_premise(sent_idx):
        return claim_order[sent_idx] == -1 and premise_order[sent_idx] != -1

    def co_reference(sent_a, sent_b):  # sent_a 和 sent_b 属于同一个论点或者属于同一个论据
        # if is_major(sent_a) and is_major(sent_b):
        #     return True
        if is_claim(sent_a) and is_claim(sent_b) and claim_order[sent_a] == claim_order[sent_b]:
            return True
        if is_premise(sent_a) and is_premise(sent_b) and premise_order[sent_a] == premise_order[sent_b]:
            return True
        return False

    def affiliation(sent_a, sent_b):  # sent_a 属于 sent_b 上一层的节点
        # if is_major(sent_a) and is_claim(sent_b):
        #     return True
        if is_claim(sent_a) and is_premise(sent_b) and premise_relation[sent_b] == claim_order[sent_a]:
            return True
        return False

    def co_occurrence(sent_a, sent_b):  # sent_a 和 sent_b 不属于同一个论点/论据，但是其所在的论点/论据共同支撑某个主论点/论点
        if is_claim(sent_a) and is_claim(sent_b) and not co_reference(sent_a, sent_b):
            return True
        if (is_premise(sent_a) and is_premise(sent_b) and not co_reference(sent_a, sent_b) and
                premise_relation[sent_a] == premise_relation[sent_b]):
            return True
        return False

    # global_idx, stop_idx = MAX_PASSAGE_LEN - 2, MAX_PASSAGE_LEN - 1
    grid = np.ones((MAX_PASSAGE_LEN, MAX_PASSAGE_LEN)).astype(np.int32) * -1
    # 0: No Relation, 1: Co-occurence, 2: Co-reference, 3: Affiliation
    # 1. Link Global with Major Claim, Link Major Claim with Other Claims
    for sent_a in range(max_len):
        # if is_major(sent_a):
        #     grid[global_idx, sent_a] = 3
        for sent_b in range(max_len):
            if affiliation(sent_a, sent_b):
                grid[sent_a, sent_b] = 3
            elif co_reference(sent_a, sent_b):
                grid[sent_a, sent_b] = 2
            elif co_occurrence(sent_a, sent_b):
                grid[sent_a, sent_b] = 1
            else:
                grid[sent_a, sent_b] = 0
    return grid



def target2order(trgs, max_len):
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
            premise = premises[claim_idx][premise_idx]
            for sent_idx in premise:
                if claim_order[sent_idx] != -1:
                    continue
                if premise_order[sent_idx] != -1 or premise_order[sent_idx] != -1:
                    continue
                premise_relation[sent_idx] = claim_idx
                premise_order[sent_idx] = claim_idx * max_premise_num + premise_idx   # important!

    # Check Premise Sanity
    # cnt_max = 0
    # for sent_idx in range(len(premise_order)):
    #     assert premise_order[sent_idx] == -1 or cnt_max <= premise_order[sent_idx]
    #     cnt_max = max(cnt_max, premise_order[sent_idx])

    return major_idx, np.array(claim_order), np.array(premise_order), np.array(premise_relation)
    # -1: Not a premise, > 0: the corresponding claim id

# 3. TODO: Some processing
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

def idx2set(idx):
    if idx < 8000:
        cnt = "train"
    elif 8000 <= idx < 9000:
        cnt = "test"
    else:
        cnt = "val"
    return cnt

def get_sentence_results(sentences, char_tokenizer, char_model, word_tokenizer, word_model):
    char_ids, char_masks, word_ids, word_masks = [], [], [], []
    for sentence in sentences:
        char_id, char_mask, char_len = char_tokenizer(sentence)
        word_id, word_mask, word_len = word_tokenizer(sentence)
        char_ids.append(char_id)
        char_masks.append(char_mask)
        word_ids.append(word_id)
        word_masks.append(word_mask)
        # char_id, char_mask = char_id.unsqueeze(0), char_mask.unsqueeze(0)
        # word_id, word_mask = word_id.unsqueeze(0), word_mask.unsqueeze(0)
        # print("CI: ", char_id.size())
        # print("CM: ", char_mask.size())
        # print("WI: ", word_id.size())
        # print("WM: ", word_mask.size())
    char_id, char_mask = torch.stack(char_ids, dim=0), torch.stack(char_masks, dim=0)
    word_id, word_mask = torch.stack(word_ids, dim=0), torch.stack(word_masks, dim=0)
    with torch.no_grad():
        char_logit, char_embedding = char_model(char_id, char_mask, word_id, word_mask)
        word_logit, word_embedding = word_model(char_id, char_mask, word_id, word_mask)
    # [L, 3] & [L, D]
    overall_logit = ((char_logit.softmax(-1) + word_logit.softmax(-1))).log_softmax(-1)
    # overall_logits.append(overall_logit.squeeze().cpu())
    overall_embedding = torch.cat((char_embedding, word_embedding), dim=-1)
    # overall_embeddings.append(overall_embedding.squeeze().cpu())
    # overall_logits = torch.stack(overall_logits, dim=0).numpy()  # (L, 3)
    # overall_embeddings = torch.stack(overall_embeddings, dim=0).numpy()  # (L, D)
    return overall_logit.cpu().numpy(), overall_embedding.cpu().numpy()
    # return overall_logits, overall_embeddings

def annotation_transform(csv_paths, modules, has_target=True):
    annotation = {"embedding": [],  "coarse_logit": [], "is_major": [],
                  "sentence_mask": [],  "paragraph_order": [], "sentence_order": [],
                  "reflection": [], "font_size": [], "style_mark": [], "label": [], "grid": []}
    if isinstance(csv_paths, str):
        result = {
            "test": copy.deepcopy(annotation)
        }
        lengths = {"test": 0}
        csv_paths = [csv_paths]
    else:
        result = {
            "train": copy.deepcopy(annotation), "val": copy.deepcopy(annotation), "test": copy.deepcopy(annotation)
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
                para_order, sent_order, font_size, style_mark = transform_tags(tags)
                if has_target:
                    major_idx, claim_order, premise_order, premise_relation = target2order(trgs, len(tags))
                    is_major = claim_order == major_idx
                    grid = order2grid(major_idx, claim_order, premise_order, premise_relation, len(tags))
            except AssertionError:
                print(f"Major claim gets destroyed for Row {row_idx}")
                continue
            if has_target:
                labels = (1 * (claim_order != -1) + 2 * (premise_order != -1)).astype(np.int32)
            # 0: others, 1: claim, 2: premise
            logits, embedding = get_sentence_results(sentences, *modules)
            if has_target:
                set_name = idx2set(all_idx)
            else:
                set_name = 'test'
            lengths[set_name] += 1
            sentence_mask = np.zeros((MAX_PASSAGE_LEN)).astype(np.int32)
            sentence_mask[:len(para_order)] = 1
            result[set_name]["embedding"].append(zero_pad(embedding, MAX_PASSAGE_LEN))
            result[set_name]["coarse_logit"].append(zero_pad(logits, MAX_PASSAGE_LEN))

            result[set_name]["sentence_mask"].append(sentence_mask)
            result[set_name]["paragraph_order"].append(zero_pad(para_order, MAX_PASSAGE_LEN))
            result[set_name]["sentence_order"].append(zero_pad(sent_order, MAX_PASSAGE_LEN))
            result[set_name]["font_size"].append(zero_pad(font_size, MAX_PASSAGE_LEN))
            result[set_name]["style_mark"].append(zero_pad(style_mark, MAX_PASSAGE_LEN))
            if has_target:
                result[set_name]["label"].append(zero_pad(labels, MAX_PASSAGE_LEN))
                result[set_name]["grid"].append(grid)
                result[set_name]["is_major"].append(zero_pad(is_major, MAX_PASSAGE_LEN))
            # result[set_name]["major_idx"].append(major_idx)
            result[set_name]["reflection"].append(all_idx)
            # result[set_name]["claim_order"].append(zero_pad(claim_order, MAX_PASSAGE_LEN))
            # result[set_name]["premise_order"].append(zero_pad(premise_order, MAX_PASSAGE_LEN))
            # result[set_name]["target_relation"].append(zero_pad(premise_relation, MAX_PASSAGE_LEN))

    for key, value in result.items():
        result[key] = value
    return result, lengths


char_path = os.path.join(BASEDIR, "pretrained_model/FinBERT_L-12_H-768_A-12_pytorch")
word_path = os.path.join(BASEDIR, 'pretrained_model/paraphrase-xlm-r-multilingual-v1')
char_base = BertTokenizer.from_pretrained(char_path, do_lowcheckpointser_case=True)
char_model_path = os.path.join(BASEDIR, "checkpoints/char/models-9.pt")
word_model_path = os.path.join(BASEDIR, "checkpoints/word/models-12.pt")
char_pretrained = os.path.join(BASEDIR, "pretrained_model/FinBERT_L-12_H-768_A-12_pytorch")
word_pretrained = os.path.join(BASEDIR, "pretrained_model/paraphrase-xlm-r-multilingual-v1")
char_tokenizer = generate_char_tokenizer(char_base)
word_tokenizer = generate_word_tokenizer(word_path)
char_model, word_model = generate_models(char_model_path, word_model_path, char_pretrained, word_pretrained)


import argparse
def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-i', '--input', type=str, default=None, help='ref file')
    parser.add_argument('-p', '--pred', type=str, default=None, help='pred file')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


if __name__ == "__main__":
    """
    python /mnt/fengyao.hjj/argument_mining/preprocess/second_stage.py \
    -i /mnt/fengyao.hjj/transformers/data/topic_pgc/0428_2022042702000000348001.csv \
    -p /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.0428_2022042702000000348001.hdf5
    """
    args = cmd()
    file = args.input
    pred = args.pred

    if file is None or pred is None:
        results, lengths = annotation_transform([os.path.join(BASEDIR, "antcritic/train.1.csv"),
                                                os.path.join(BASEDIR, "antcritic/dev.1.csv"),
                                                os.path.join(BASEDIR, "antcritic/test.1.csv")],
                                                 (char_tokenizer, char_model, word_tokenizer, word_model))
        for set_key in results.keys():
            file = h5py.File(os.path.join(BASEDIR, "antcritic/{}_2.hdf5".format(set_key)), "w")
            for input_key in results[set_key].keys():
                try:
                    file.create_dataset(input_key, data=np.array(results[set_key][input_key]))
                except Exception as e:
                    print(input_key)
                    raise e
            file.attrs["size"] = lengths[set_key]
            file.close()

    else:
        results, lengths = annotation_transform(file, (char_tokenizer, char_model, word_tokenizer, word_model), has_target=False)
        for set_key in results.keys():
            file = h5py.File(pred, "w")
            for input_key in results[set_key].keys():
                try:
                    file.create_dataset(input_key, data=np.array(results[set_key][input_key]))
                except Exception as e:
                    print(input_key)
                    raise e
            file.attrs["size"] = lengths[set_key]
            file.close()
