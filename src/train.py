from typing import Any, Dict

import yaml
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DetectionDataset, collate_fn
from model import DetectionModel


class DetectionTrainer:
    """ 
        Trainer class to handle training and validation loops.
        1. Initialize with model, data loaders, optimizer, scheduler, device (cpu or gpu).
        2. train_epoch: Train for one epoch, return average loss.
        3. validate_epoch: Validate for one epoch, return average loss.
        4. train: Full training loop for num_epochs, save best model at end.
    """
    def __init__(self, model, train_loader, val_loader, lr, num_epochs, device):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.num_epochs = num_epochs

        params = [p for p in self.model.parameters() if p.requires_grad]
        # used SGD optimizer with momentum and weight decay for regularization
        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        self.train_losses = []
        self.val_losses = []


    def train_epoch(self):
        """ Train for one epoch and return average loss"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(self.train_loader, desc="Training")

        for imgs, targets in progress_bar:
            # move to device if available
            imgs = [img.to(self.device) for img in imgs]
            targets = [{key: val.to(self.device) for key, val in t.items()} for t in targets]
            self.optimizer.zero_grad()
            # forward pass
            loss_dict = self.model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            # backward pass
            losses.backward()
            self.optimizer.step()
            epoch_loss += losses.item()
            num_batches += 1

            progress_bar.set_postfix({"train_loss": losses.item()})
        
        return epoch_loss / num_batches
    

    def validate_epoch(self):
        """ Validate for one epoch and return average loss"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for imgs, targets in progress_bar:
                # move to device if available
                imgs = [img.to(self.device) for img in imgs]
                targets = [{key: val.to(self.device) for key, val in t.items()} for t in targets]
                # forward pass
                loss_dict = self.model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()
                num_batches += 1

                progress_bar.set_postfix({"val_loss": losses.item()})
        
        return epoch_loss / num_batches

    def train(self, save_path):
        """ Full training loop for num_epochs, save best model at end """
        print("Starting training...")
        best_val_loss = float("inf")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            # train for one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            # validate for one epoch
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            # update learning rate 
            self.lr_scheduler.step()
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_val_loss': best_val_loss
                }, save_path)
                print(f"Best model saved with val loss: {val_loss:.4f}")
        
        return self.train_losses, self.val_losses



def train(config: Dict[str, Any]):
    """ 
        Main training function to set up data loaders, model, and trainer, then start training.
        Saves the best model and training history to specified paths at end.
    """
    # config
    train_dir: str = config["train_path"]
    val_dir: str = config["val_path"]
    model_output_path: str = config["model_output_path"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    """
    Define the model class and instantiate it

    """
    model = DetectionModel(classes=4)  # 3 classes + background
    
    # datasets
    train_dataset = DetectionDataset(base_dir=train_dir, split="train",
                                      transforms=DetectionDataset.get_transforms(train=True))
    val_dataset = DetectionDataset(base_dir=val_dir, split="val", 
                                   transforms=DetectionDataset.get_transforms(train=False))
    
    print(f"Number of training samples: {len(train_dataset)}")  # sanity check, should be 1800
    print(f"Number of validation samples: {len(val_dataset)}")  # sanity check, should be 200

    train_dataloader = DataLoader(
        train_dataset,
        config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    val_dataloader = DataLoader(
        val_dataset,
        config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    print("Testing data loaders...")
    # simple loop to check data loader
    for i, batch in enumerate(train_dataloader):
        imgs, targets = batch
        print(f"Batch {i}: {len(imgs)} images, {len(targets)} targets")
        if i == 0:
            break

    """
    Create a trainer class that will be used to train the model
    
    """
    trainer = DetectionTrainer(
        model=model, 
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        lr=config['learning_rate'],
        num_epochs=config['num_epochs'],
        device=device
    )
    train_losses, val_losses = trainer.train(save_path=model_output_path)

    """
    Checkpoint the model during or after training and store the results in the model_output_path
    
    """
    # save the training history as well, for later visualization of losses
    history_path = model_output_path.replace(".pth", "_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f, indent=2)
    # sanity check, check paths to make sure files are saved
    print(f"Training complete. Model saved to {model_output_path} and training history saved to {history_path}.")

    """
    Either in this file or in a separate file, run the evaluation loop and store the results

    """
    # in evaluate.py



if __name__ == "__main__":
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    train(config)