<div align="center">

# From Benchmarks to Synthetic Data: Improving Hierarchical Panoptic Segmentation for Agricultural Perception Systems

<strong>Francisco Murta</strong>
·
<strong>Tomás Coelho</strong>

</div>

<div align="center">
    <a href="https://arxiv.org/abs/2312.09231" class="button"><b>[Paper]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://drive.google.com/drive/folders/1c3HthfWYrw_PEbf0eD2CRYp-xwYmxbLV?usp=sharing" class="button"><b>[Data]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://drive.google.com/drive/folders/1BK1-I1uys0PN6U8KEVDjkLwMAQIbKJho?usp=sharing" class="button"><b>[Checkpoints]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

<br/>

![Teaser](./assets/teaser.png)

# Getting started

This repository integrates two key components, each hosted in separate GitHub repositories:

1. **Baseline Model**: The first repository contains the official codebase used by the [third-place winner](https://github.com/DTU-PAS/phenobench_challenge.git) of the [CVPPA@ICCV'23 Challenge](https://codalab.lisn.upsaclay.fr/competitions/14153), which focuses on hierarchical panoptic segmentation of crops and weeds. It serves both as a benchmark architecture and as the evaluation framework for all generated predictions.

2. **Synthetic Data Generation**: The second repository leverages the [**ControlNet** architecture to generate synthetic data](https://github.com/valeoai/GenVal.git). It extends the ControlNet model to agricultural imagery by conditioning style transfer on segmentation masks from the PhenoBench dataset.

### 🔁 General Pipeline Overview

The proposed workflow is structured as follows:

1. **Train ControlNet on PhenoBench**: Fine-tune the ControlNet model using real semantic masks and paired RGB imagery from the PhenoBench dataset.

2. **Generate Synthetic Samples**: Use the trained ControlNet to produce new, realistic images that replicate the style and structure of crop scenes.

3. **Augment Training Data**: Add these synthetic images (with corresponding segmentation masks) to the original training set to increase diversity and volume.

4. **Evaluate Improvement**: Retrain or fine-tune the baseline segmentation model using the augmented dataset and evaluate performance on the PhenoBench validation set using metrics such as **PQ**, **IoU**, and **PQ+**.

This pipeline aims to assess whether synthetic data can improve segmentation accuracy in agricultural settings, particularly under the constraints of limited annotated data.
# 1. Download PhenoBench dataset

Download into a folder the [PhenoBench Dataset](https://www.phenobench.org/data/PhenoBench-v110.zip) from the official link. 


# 2. Style-transfer data generation

A good practice that makes it easier to track which original images are used during the final step of the style-transfer pipeline is to create an auxiliary folder named `PhenoBench_GenImg`.

This folder should mirror the structure of the original PhenoBench dataset:

PhenoBench_GenImg/
├── images/
├── plant_instances/
├── leaf_instances/
└── semantics/

Inside each of these subfolders, copy a selected subset (`n` images) from the original `train` set of the PhenoBench dataset. Ensure that for each selected image, its corresponding annotations are also copied into the respective folders.

⚠️Rename Images to Avoid Conflicts

Once copied, run a script to rename all selected images consistently using a naming convention like:

Z0001.png, Z0002.png, ..., Z0500.png

⚠️ Important: 
All four folders (`images`, `plant_instances`, `leaf_instances`, `semantics`) must use exactly the same filenames to ensure alignment during training.

This step is critical because when you later copy the generated synthetic images back into the main PhenoBench training set, this renaming avoids filename conflicts or accidental overwrites of original data.


## Clone Repository:

    git clone https://gitlab.isae-supaero.fr/alice/nanostar/alice-group/students/mae2_2025-2027-francisco-tomas.git


## Create the Environment:

    cd ThirdPlace/phenobench_challenge-main/
    pip install -r requirements.txt



## Control Net Image Generation Pipeline

The style-transfer pipeline is based on the original repo of [ControlNet](https://github.com/lllyasviel/ControlNet).

To be able to train the ControlNet and then generate new images using the semantic masks of PhenoBench, follow the following steps:


<details><summary><strong>Captioning New Dataset</strong></summary>
&emsp;

To train ControlNet on the PhenoBench dataset, you first need to generate descriptive captions for the original training images. This can be done using the [CLIP-interrogator](https://github.com/pharmapsychotic/clip-interrogator) tool.

Begin by installing the required package and running the captioning script:

<pre><code>pip install clip-interrogator==0.5.4
python clip_int.py --dataset {dataset}</code></pre>

This will produce caption descriptions for each image in the dataset and save them into a `.json` file located in the `captions/` folder.

Next, repeat the process for the `PhenoBench_GenImg` dataset — the subset of training images you've selected and renamed for style transfer generation. This ensures that each of the `n` selected images also has a matching caption.

At the end of this step, you should have two `.json` files:

1. `captions_PhenoBench.json` — Captions for the full original PhenoBench training set.
2. `captions_PhenoBench_GenImg.json` — Captions for the subset of images that will be used in the synthetic data generation process.

These caption files are essential for guiding the ControlNet during both training and inference.
</details>

<br/>   

<details><summary><strong>Train the ControlNet</strong></summary>
&emsp;

- Now you will train the ControlNet yourself. You will first need to create the trainable copy of the encoder of the denoising U-Net of Stable Diffusion. 

First download the pretrained Stable Diffusion model (7.7 Gb):

    wget -P models/ https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt

Then create the trainable copy doing:

    python tool_add_control.py models/v1-5-pruned.ckpt models/control_seg.ckpt

Finally, launch training (you will need 1 GPU with 40 Gb VRAM, or you can decrease the batch size and adapt the gradient accumulation). Here you will use for training the json file "captions_Phenobench":

    python train.py
    
The checkpoints during training will be stored in ``logs/{run_num}/lightning_logs/version_0/checkpoints/`` folder. You can also visualize the training advancement in ``logs/{run_num}/image_log/``

</details>

<br/>

<details><summary><strong>Generate images with transferred style</strong></summary>
&emsp;

The checkpoint that is used to generate new samples is the one obtained on the step before.

To generate 512 samples with fog, you can launch (don't forget that now your dataset is the PhenoBench_GenImg so you will need to change the json file to "captions_Phenobench_GenImg"):

    python style_transfer.py --num_samples=512 --domain=fog

You can choose whatever domain you want by changing the ``--domain`` option above when launching the command (for our case we disabled the option of domain because it isn't particurally benefitial for our goal).

This will create new samples in ``../images/style_transfer/`` folder, based on random examples from the train set of PhenoBench, with the corresponding original images and ground truths in different subfolders.

</details>

<br/>


<details><summary><strong>Copy Generated Images to PhenoBench Original Dataset</strong></summary>
&emsp;

To evaluate the effect of the generated images using the competition's baseline model, you need to incorporate the synthetic samples into the official PhenoBench training set.

Start by copying the generated images (located in `../images/style_transfer/generated`) into the `images` folder of the PhenoBench training dataset:

<pre><code>cp ../images/style_transfer/generated/*.png ../PhenoBench/train/images/</code></pre>

Next, navigate to the `PhenoBench_GenImg` folder and copy the corresponding ground truth annotations — from the `semantics`, `plant_instances`, and `leaf_instances` subfolders — into the respective directories of the original PhenoBench dataset. Make sure you copy the renamed files (e.g., Z0001.png to Z0nnn.png) to ensure consistency:

<pre><code>cp ../PhenoBench_GenImg/train/semantics/*.png ../PhenoBench/train/semantics/
cp ../PhenoBench_GenImg/train/plant_instances/*.png ../PhenoBench/train/plant_instances/
cp ../PhenoBench_GenImg/train/leaf_instances/*.png ../PhenoBench/train/leaf_instances/</code></pre>

At this point, your original PhenoBench training dataset should include both the original labeled images and the newly generated synthetic samples (with annotations), ready for training and evaluation.
</details>

<br/>







# 2. Mask2Former At Phenobench

Please download the weights for each mask2former model if you want to just perform inference on the dataset.These are the base parameters given by [third place](https://github.com/DTU-PAS/phenobench_challenge.git).

1. [Panoptic plant segmention](https://data.dtu.dk/ndownloader/files/42444264)
2. [Small Plant Refinement](https://data.dtu.dk/ndownloader/files/42444267)
3. [Leaf Instance Crop Segmentation](https://data.dtu.dk/ndownloader/files/42444273)

<p><strong>Note:</strong> Please update the paths in all <code>configs/*.yaml</code> files before training or inference:</p>

- `dataset_root`: This should point to the **PhenoBench dataset** that now includes both the original labeled images and the newly generated synthetic samples.
- `output_folder`: This path will be used during inference to store prediction outputs (e.g., segmentation masks, visualizations).
- `output_train`: (Recommended) A separate path for storing training logs and model checkpoints to avoid cluttering the prediction output directory.

<details><summary><strong>Training</strong></summary>
&emsp;

To begin the training process, follow the steps below:

1. **Filter and Index Valid Training Instances**  
   These scripts will process and filter the dataset, generating `.txt` files used during training:

<pre><code>python filter_crop_instances.py
python filter_small_instances.py</code></pre>

2. **Train Each Stage of the Hierarchical Segmentation**  
   Launch the training scripts for each Mask2Former module:

<pre><code>python train_panoptic_plant_instance.py
python train_plant_refinement.py
python train_leaf_instance_crop.py</code></pre>

Each script will generate model checkpoints (`.ckpt`) in the `output_train` folder configured inside its corresponding `.yaml` configuration file.

You can now replace the default pre-trained checkpoints with these newly trained models for evaluation.

</details>

<details><summary><strong>Inference</strong></summary>
&emsp;

After training, you can run the full inference pipeline to generate predictions using the updated checkpoints:

<pre><code>python mask2former_at_phenobench/inference_all.py</code></pre>

Ensure that the `.yaml` file used here points to the newly trained checkpoints and the correct `dataset_root`. The predictions will be saved in the `output_folder` path specified in the same configuration file.

</details>



