"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        num_samples = len(self.dataset)
        num_batches = num_samples // self.batch_size
        remainder = num_samples % self.batch_size

        if remainder != 0 and not self.drop_last:
            num_batches += 1

        indices = np.arange(num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch = {}

            for key in self.dataset[0].keys():
                data_list = [self.dataset[idx][key] for idx in batch_indices]
                batch[key] = np.array(data_list)

            yield batch  

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last!                                 #
        ########################################################################
        num_samples = len(self.dataset)
        length = num_samples // self.batch_size

        if not self.drop_last and num_samples % self.batch_size != 0:
            length += 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
