from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from realign.realignretriever.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        
        if self.data_args.dataset_path and (os.path.exists(self.data_args.dataset_path) or 
                                            os.path.exists(os.path.dirname(self.data_args.dataset_path))):
            if os.path.isdir(self.data_args.dataset_path):
                data_files = os.path.join(self.data_args.dataset_path, '*.parquet')
                if not any(f.endswith('.parquet') for f in os.listdir(self.data_args.dataset_path)):
                    data_files = os.path.join(self.data_args.dataset_path, '*.json')
                self.train_data = load_dataset(
                    'parquet' if '*.parquet' in data_files else 'json',
                    data_files=data_files,
                    split=self.data_args.dataset_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
            else:
                self.train_data = load_dataset(
                    'parquet' if self.data_args.dataset_path.endswith('.parquet') else 'json',
                    data_files=self.data_args.dataset_path,
                    split=self.data_args.dataset_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
        else:
            self.train_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
            
        if not self.data_args.pretrain:
            corpus_dir = self.data_args.corpus_path.replace('/*.parquet', '').replace('*.parquet', '').replace('/*', '').replace('*', '')
            if self.data_args.corpus_path and (os.path.exists(corpus_dir) or os.path.isdir(os.path.dirname(self.data_args.corpus_path))):
                if os.path.isdir(corpus_dir):
                    corpus_files = os.path.join(corpus_dir, '*.parquet')
                else:
                    corpus_files = self.data_args.corpus_path
                    
                self.corpus  = load_dataset(
                    'parquet',
                    data_files=corpus_files,
                    split=self.data_args.corpus_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )
            else:
                self.corpus  = load_dataset(
                    self.data_args.corpus_name,
                    self.data_args.corpus_config,
                    data_files=self.data_args.corpus_path,
                    split=self.data_args.corpus_split,
                    cache_dir=self.data_args.dataset_cache_dir,
                )

            from datasets.features import Image as ImageFeature
            if 'image' in self.corpus.features:
                new_features = self.corpus.features.copy()
                new_features['image'] = ImageFeature(decode=True)
                self.corpus = self.corpus.cast(new_features)


            self.docid2idx = {}
            if 'doc_id' in self.corpus.features:
                for idx, docid in enumerate(self.corpus['doc_id']):
                    self.docid2idx[str(docid)] = idx
            else:
                for idx in range(len(self.corpus)):
                    self.docid2idx[str(idx)] = idx

        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        if 'query_type' in group:
            query_text = group['query_text']
            query_type = group['query_type']
            describe = group.get('description', '') if query_type == 'image' else ''
        else:
            query_text = group['query']
            query_type = 'text'
            describe = ''
        
        if self.data_args.pretrain:
            image = group['image']
            return query_text, image
        else:
            relevant_docids = group['relevant_doc_ids']

            if not relevant_docids or len(relevant_docids) == 0:
                relevant_doc_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
            else:
                if self.data_args.positive_document_no_shuffle or self.data_args.image_sample_strategy == 'first':
                    docid = relevant_docids[0]
                else:
                    docid = relevant_docids[(_hashed_seed + epoch) % len(relevant_docids)]

                try:
                    relevant_doc_image = self.corpus[self.docid2idx[docid]]['image']
                except KeyError:
                    relevant_doc_image = Image.new('RGB', (224, 224), color=(255, 255, 255))

        return query_text, relevant_doc_image, query_type, describe


class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        if self.data_args.encode_is_query:
            self.encode_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )
        else:    
            self.encode_data = load_dataset(
                self.data_args.corpus_name,
                self.data_args.corpus_config,
                data_files=self.data_args.corpus_path,
                split=self.data_args.corpus_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        
    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        data = self.encode_data[item]
        text, image = None, None
        if self.data_args.encode_is_query:
            id = data['query_id']
            text = data['query']
        else:
            id = data['doc_id']
            image = data['image']
        return id, text, image
