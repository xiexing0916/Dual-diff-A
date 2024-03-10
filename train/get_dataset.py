from dataset import LIDCDataset, DEFAULTDataset, SingleDataGenerator
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == 'LIDC':
        # train_dataset = LIDCDataset(augmentation=False)
        # val_dataset = LIDCDataset(augmentation=False)
        train_dataset = SingleDataGenerator(mode="train")
        val_dataset = SingleDataGenerator(mode="val")
        visual_dataset = SingleDataGenerator(mode="visual")
        sampler = None
        return train_dataset, val_dataset, visual_dataset
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
