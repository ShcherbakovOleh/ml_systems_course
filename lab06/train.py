import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow

# ==========================================
# Load configuration from config.yaml (Task 1)
# ==========================================
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Simple MLP model (adapted from Lab 3 / Lab 5 structures)
class LabDatasetMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=64, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train():
    config = load_config()
    
    # Set random seed for reproducibility (Task 1)
    torch.manual_seed(config["seed"])
    
    # Initialize MLflow tracking (Task 1)
    mlflow.set_experiment(config["mlflow_experiment_name"])
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(config)
        
        # Dummy dataset representing Lab 3 data (e.g., Fashion-MNIST shape)
        X = torch.randn(1000, 1, 28, 28)
        y = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(X, y)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            num_workers=config["num_workers"]
        )
        
        model = LabDatasetMLP()
        criterion = nn.CrossEntropyLoss()
        
        if config["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
            
        if config["scheduler"] == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
            
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
        start_epoch = 0
        
        # Checkpoint resume logic (Task 7)
        latest_ckpt = os.path.join(config["checkpoint_dir"], "latest.pt")
        if os.path.exists(latest_ckpt):
            checkpoint = torch.load(latest_ckpt)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed training from epoch {start_epoch}")

        # PyTorch Profiler setup (Task 2)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            
            # Training loop with Gradient Accumulation (Task 6)
            accumulation_steps = 4  # Emulate larger batch size
            
            for epoch in range(start_epoch, config["epochs"]):
                model.train()
                running_loss = 0.0
                optimizer.zero_grad()
                
                for i, (inputs, targets) in enumerate(dataloader):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        
                    running_loss += loss.item() * accumulation_steps
                    prof.step() # Profiler step
                
                scheduler.step()
                epoch_loss = running_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {epoch_loss:.4f}")
                
                # Log metrics to MLflow (Task 1)
                mlflow.log_metric("loss", epoch_loss, step=epoch)
                
                # Save checkpoint (Task 1 & 7)
                ckpt_path = os.path.join(config["checkpoint_dir"], f"epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }, ckpt_path)
                # Keep latest checkpoint for easy resume
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }, latest_ckpt)

        # Log artifacts (Task 1)
        mlflow.log_artifacts(config["checkpoint_dir"])
        print("Training pipeline finished successfully!")

if __name__ == "__main__":
    train()