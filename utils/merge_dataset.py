# Last modified: 2024-04-18
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: GitHub - prs-eth/Marigold: [CVPR 2024 - Oral, Best Paper Award Candidate] Marigold: Repurposing Diff
# If you use or adapt this code, please attribute to GitHub - prs-eth/Marigold: [CVPR 2024 - Oral, Best Paper Award Candidate] Marigold: Repurposing Diff.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------



import torch
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)


class MixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self, 
        src_dataset_ls, 
        batch_size, 
        rank, 
        seed, 
        num_replicas, 
        drop_last=True, 
        shuffle=True, 
        prob=None, 
        generator=None
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.prob_generator = torch.Generator().manual_seed(seed+rank*seed)

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)
        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # cumulative dataset length
        # for dist
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        # end dist

        # BatchSamplers for each source dataset
        if self.shuffle:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=RandomSampler(
                        ds, replacement=False, generator=self.generator
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        else:
            self.src_batch_samplers = [
                BatchSampler(
                    sampler=SequentialSampler(ds),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for ds in self.src_dataset_ls
            ]
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # index in original dataset

        # start dist
        self.raw_batches_split = []
        for i in range(len(self.raw_batches)):
            batch = self.raw_batches[i]
            self.raw_batches_split.append(list(batch[self.rank:self.dataset_length[i]:self.num_replicas]))

        self.n_batches = [len(b) for b in self.raw_batches_split]
        self.n_total_batch = sum(self.n_batches)
        # end dist

        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)

    def __iter__(self):
        """_summary_

        Yields:
            list(int): a batch of indics, corresponding to ConcatDataset of src_dataset_ls
        """
        for i in range(self.n_total_batch):
            idx_ds = torch.multinomial(
                self.prob, 1, replacement=True, generator=self.prob_generator,
            ).item()

            # if batch list is empty, generate new list
            if 0 == len(self.raw_batches_split[idx_ds]):
                # self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])
                self.raw_batches_split[idx_ds] = list(self.raw_batches[idx_ds][self.rank:self.dataset_length[idx_ds]:self.num_replicas])
            # get a batch from list
            batch_raw = self.raw_batches_split[idx_ds].pop()
            # shift by cumulative dataset length
            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]

            yield batch

    def __len__(self):
        return self.n_total_batch 




