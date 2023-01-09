"""
filter training data to each class only has 1500 examples
"""

import os
import h5py
import pickle
import random
from collections import defaultdict
from operator import itemgetter
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import normalize
import pandas as pd
import glob, json, torch, time
from torch.utils.data import Dataset, DataLoader


def main():
    print(os.getcwd())
    train_file = '/home/xinyue/VQA_ReGat/data/mimic/mimic_dataset_train_full.pkl'
    with open(train_file, 'rb') as f:
        qa = pickle.load(f)  # [508543]
        max_count = 1000
        per_class_dict = defaultdict(list) # {class: list of indices}
        # 1. stat each class's count
        # all_answsers = []  # [772395]
        for i, example in enumerate(qa):
            ans = example['answer']['labels']
            for label in ans:
                if label not in per_class_dict:
                    per_class_dict[label].append(i)
                elif label in per_class_dict and len(per_class_dict[label]) < max_count:  # 
                    per_class_dict[label].append(i)                    

        selected_indices = []  # [16259]
        for k, v in per_class_dict.items():
            selected_indices.extend(v)
        selected_indices = list(set(selected_indices))
        save_path = os.path.join(os.getcwd(), f'datasets/filtered_qa_indices.pkl')
        with open(save_path, 'wb') as f2:
            pickle.dump(selected_indices, f2)


def main2():
    """
    randomly select examples from each class
    """
    print(os.getcwd())
    train_file = '/home/xinyue/VQA_ReGat/data/mimic/mimic_dataset_train_full.pkl'
    with open(train_file, 'rb') as f:
        qa = pickle.load(f)  # [508543]
        max_count = 1000
        per_class_dict = defaultdict(list) # {class: list of indices}
        
        for i, example in enumerate(qa):
            ans = example['answer']['labels']
            for label in ans:
                if label not in per_class_dict:
                    per_class_dict[label].append(i)
                else:
                    per_class_dict[label].append(i)                    

        selected_indices = []  # [16259]
        for k, v in per_class_dict.items():
            selected_indices.extend(np.random.choice(v, size=max_count, replace=False))
        selected_indices = list(set(selected_indices))
        save_path = os.path.join(os.getcwd(), f'datasets/filtered_qa_indices.pkl')
        with open(save_path, 'wb') as f2:
            pickle.dump(selected_indices, f2)

def load_filtered_indices():
    train_file = '/home/xinyue/VQA_ReGat/data/mimic/mimic_dataset_train_full.pkl'
    save_path = os.path.join(os.getcwd(), f'datasets/filtered_qa_indices.pkl')
    with open(save_path, 'rb') as f1:
        indices = pickle.load(f1)
        with open(train_file, 'rb') as f2:
            qa = pickle.load(f2)
            qa = np.array(qa)
            # op = itemgetter(indices)
            # sublist = list(op(qa))
            sublist = qa[indices]
            print(sublist)


if __name__ == "__main__":
    # main()
    # load_filtered_indices()
    main2()
