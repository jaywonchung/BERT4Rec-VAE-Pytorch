from abc import *
from pathlib import Path
import pickle


class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.save_folder = save_folder

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print('Negatives samples exist. Loading.')
            negative_samples = pickle.load(savefile_path.open('rb'))
            return negative_samples
        print("Negative samples don't exist. Generating.")
        negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump(negative_samples, f)
        return negative_samples

    def _get_save_path(self):
        folder = Path(self.save_folder)
        filename = '{}-sample_size{}-seed{}.pkl'.format(self.code(), self.sample_size, self.seed)
        return folder.joinpath(filename)
