import torchvision.datasets as datasets
import pickle


class ImageNetWithScale(datasets.ImageFolder):
    def __init__(self, root, searched_scale_path, transform=None):
        super(ImageNetWithScale, self).__init__(root, transform)
        f = open(searched_scale_path, 'rb')
        self.gt_scale = pickle.load(f)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, gt_scale) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.gt_scale[index, :]

