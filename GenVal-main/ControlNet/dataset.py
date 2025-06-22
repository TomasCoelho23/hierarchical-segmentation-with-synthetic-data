import json
import cv2
import numpy as np
import random
import os
from torch.utils.data import Dataset

class PhenoBenchDataset(Dataset):
    def __init__(self, root, split='train', prompt_json=None):
        self.root = root
        self.split = split
        self.image_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "semantics")
        self.filenames = sorted(os.listdir(self.image_dir))

        # Load prompts from JSON file (expects {"image_name.png": "prompt text", ...})
        if prompt_json is not None:
            with open(prompt_json, 'r') as f:
                self.prompts = json.load(f)
        else:
            self.prompts = {}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Resize/crop if needed (optional)
        #This is most likely needed unless you have a very good GPU. Basically you are forcing the resizing because it is dificult to handle 1024x1024 images
        #image = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
        #mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_NEAREST)

        # Normalize image to [-1, 1]
        image = (image / 127.5) - 1.0

        # Add channel axis to mask
        mask = mask[..., np.newaxis]

        # Get prompt
        prompt = self.prompts.get(img_name, "")

        return {
            "jpg": image,      # shape [H, W, 3], float32
            "txt": prompt,     # string
            "hint": mask,       # shape [H, W, 1], float32
            "filename": img_name  # string
        }