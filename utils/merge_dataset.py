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

        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed



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
        ]  


        self.raw_batches_split = []
        for i in range(len(self.raw_batches)):
            batch = self.raw_batches[i]
            self.raw_batches_split.append(list(batch[self.rank:self.dataset_length[i]:self.num_replicas]))

        self.n_batches = [len(b) for b in self.raw_batches_split]
        self.n_total_batch = sum(self.n_batches)



        if prob is None:

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


            if 0 == len(self.raw_batches_split[idx_ds]):

                self.raw_batches_split[idx_ds] = list(self.raw_batches[idx_ds][self.rank:self.dataset_length[idx_ds]:self.num_replicas])

            batch_raw = self.raw_batches_split[idx_ds].pop()

            shift = self.cum_dataset_length[idx_ds]
            batch = [n + shift for n in batch_raw]

            yield batch

    def __len__(self):
        return self.n_total_batch 



if "__main__" == __name__:
    from torch.utils.data import ConcatDataset, DataLoader, Dataset

    class SimpleDataset(Dataset):
        def __init__(self, start, len) -> None:
            super().__init__()
            self.start = start
            self.len = len

        def __len__(self):
            return self.len

        def __getitem__(self, index):
            return self.start + index

    dataset_1 = SimpleDataset(0, 10)
    dataset_2 = SimpleDataset(200, 20)
    dataset_3 = SimpleDataset(1000, 50)

    concat_dataset = ConcatDataset(
        [dataset_1, dataset_2, dataset_3]
    )  

    mixed_sampler = MixedBatchSampler(
        src_dataset_ls=[dataset_1, dataset_2, dataset_3],
        batch_size=1,
        drop_last=True,
        shuffle=False,
        prob=[0.6, 0.3, 0.1],
        generator=torch.Generator().manual_seed(0),
        num_replicas=4,
        rank=0,
        seed=0,
    )

    loader = DataLoader(concat_dataset, batch_sampler=mixed_sampler)

    for d in loader:
        print(d)




