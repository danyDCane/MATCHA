import os

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class PACSDataset(Dataset):
    def __init__(self, root, dataset_name, transform=None):
        self.dataset_name = dataset_name
        dataset_path = os.path.join(root, 'PACS')
        self.dataset_path = os.path.join(dataset_path, f'{dataset_name}')
        
        # Check if path exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f'PACS dataset path not found: {self.dataset_path}')
        
        print(f'[PACSDataset] Loading domain "{dataset_name}" from: {self.dataset_path}')
        image_folder_dataset = ImageFolder(root=self.dataset_path)
        self.dataset = image_folder_dataset
        # expose common attributes used by downstream wrappers
        self.targets = image_folder_dataset.targets

        self.num_class = 7
        self.transform = transform
        
        print(f'[PACSDataset] Successfully loaded {len(self.dataset)} samples from domain "{dataset_name}"')
        print(f'[PACSDataset] Number of classes: {self.num_class}')

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.dataset)

