
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

from datasets.phenobench_panoptic import PhenoBenchPanopticsPlants
from utils.helper import collate_fn, get_transforms
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig
from transformers import Mask2FormerImageProcessor
import os

from lightning.pytorch.loggers import CSVLogger
from models.mask2former_phenobench_panoptic import Mask2FormerPhenobenchPanopticLightningModule

import torch

            
@hydra.main(config_path="../configs", config_name="panoptic_plant_segmentation.yaml")
def main(cfg : DictConfig):
    pl.seed_everything(cfg.seed)
    no_devices = torch.cuda.device_count()
    gpus = list(range(no_devices)) 
    # Preparing Dataloaders
    # 1. Datasets
    if cfg.model.mode == "panoptic":
        ignore_index = 255
    elif cfg.model.mode == "instance":
        ignore_index = 0
    processor = Mask2FormerImageProcessor(size=(cfg.data.img_size, cfg.data.img_size), ignore_index=ignore_index, do_resize=False, do_rescale=False, do_normalize=False)
    train_dataset = PhenoBenchPanopticsPlants(cfg.data.root, "train", processor, target_type=cfg.data.target_type, transform=get_transforms(cfg, True), blackout=cfg.data.blackout, overfit=cfg.overfit)
    val_dataset = PhenoBenchPanopticsPlants(cfg.data.root, "val", processor, target_type=cfg.data.target_type, transform=get_transforms(cfg, False), blackout=cfg.data.blackout, overfit=cfg.overfit)
    if cfg.overfit:
        train_dataset = val_dataset
    # 2. Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.data.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=cfg.data.num_workers)


    # Lightning Module
    pl_module = Mask2FormerPhenobenchPanopticLightningModule(cfg, processor)


    # Callbacks
    log_folder = f"{cfg.logging.output_folder}/{cfg.project_name}/{cfg.experiment_name}"
    os.makedirs(log_folder, exist_ok=True)
    logger = CSVLogger(save_dir=log_folder)

    checkpoint_callback = ModelCheckpoint(
                monitor='val/loss',
                dirpath=log_folder,
                filename='epoch{epoch:02d}-val_loss{val/loss:.2f}',
                auto_insert_metric_name=False,
                save_top_k=3,
                save_last=True,
                mode='min',
                )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')


    trainer = pl.Trainer(
        accelerator='gpu',
        devices=gpus,
        strategy='ddp',
        max_steps=cfg.training.max_steps,
        log_every_n_steps=cfg.training.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        default_root_dir=log_folder,
        logger=logger,   
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )
    if os.path.exists(cfg.model.ckpt_path):
        trainer.fit(
            pl_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.model.ckpt_path,
        )
    else:
        trainer.fit(
            pl_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

if __name__ == '__main__':
    main()
           





