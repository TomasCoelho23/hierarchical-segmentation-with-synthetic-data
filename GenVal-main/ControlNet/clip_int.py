import glob
import json
import argparse
import os

from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm


def main():
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='phenobench', type=str, help='dataset to process in phenobench')
    parser.add_argument('--device', default=0, type=int, help='device to run on')
    args = parser.parse_args()

    # Get arguments
    dataset = args.dataset
    device = int(args.device)

    # Get files to read from
    #Change this to phenobench dataset
    #Here be careful
    #When you are cliping the original folder Phenobench use the config.json
    #When you are cliping the folder Phenobench_GenImg use the config_GenImg.json
    config = json.load(open('../GenVal-main/ControlNet/config.json'))
    dpath = config[dataset]
    files = {dataset: glob.glob(dpath, recursive=True)}

    # Initializing captions
    captions = {}
    os.makedirs('captions', exist_ok=True)

    # Initializing interrogator
    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai", cache_path='cache', device=device, quiet=True))

    # Looping over datasets and files to get both captions and embeddings
    for dset in files:
        captions[dset] = {}

        for file in tqdm(files[dset]):
            # Read image
            image = Image.open(file).convert('RGB')

            # Get captions and embeddings
            caption = ci.interrogate(image)
            captions[dset][file] = caption

    # Save files
    #Here be careful
    #As stated before you need two config json : One that corresponds to the clipping of the original Phenobench dataset, which can be called captions_phenobench.json
    # And another json file corresponding to the folder containing the images that will be used for generation (in the folder Phenobench_GenImg), which can be called  captions_generation.json
    json.dump(captions, open(f'captions/captions_{dset}.json', 'w'), indent=4)


if __name__ == '__main__':
    main()