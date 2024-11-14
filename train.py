import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import json
import os
from typing import Tuple, Dict, Optional

from data_collection import HistoricalDataCollector
from preprocessing import FeatureExtractor, LogarithmicBuffer, prepare_data
from models import CryptoSTST

class CryptoDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self) -> int:
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=50,
            steps_per_epoch=len(train_loader)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Pour le suivi des métriques
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
        val_loss = total_loss / len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        
        return val_loss, accuracy
    
    def train(self, epochs: int = 50, save_path: str = 'models') -> Dict:
        """
        Entraîne le modèle et sauvegarde le meilleur
        Returns: historique d'entraînement
        """
        os.makedirs(save_path, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, accuracy = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.accuracies.append(accuracy)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
            
            # Sauvegarde le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(save_path, 'best_model.pth'))
                
                # Sauvegarde les méta-données
                metadata = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'accuracies': self.accuracies
        }

async def prepare_training_data(
    days: int = 30,
    symbol: str = 'DOGE/USDT'
) -> Tuple[np.ndarray, np.ndarray]:
    """Récupère et prépare les données d'entraînement"""
    collector = HistoricalDataCollector()
    start_date = datetime.now() - timedelta(days=days)
    
    print(f"Récupération des données historiques pour {symbol}...")
    df = await collector.fetch_historical_data(
        symbol=symbol,
        timeframe='1m',
        start_date=start_date
    )
    
    print("Préparation des features...")
    feature_extractor = FeatureExtractor()
    log_buffer = LogarithmicBuffer()
    
    X, y = prepare_data(df, feature_extractor, log_buffer)
    return X, y

async def main():
    # Paramètres
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    
    # Récupération des données
    X, y = await prepare_training_data(days=30)
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Création des datasets
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Création et entraînement du modèle
    model = CryptoSTST(feature_dim=X.shape[-1])
    trainer = Trainer(model, train_loader, val_loader, LEARNING_RATE)
    
    print("Démarrage de l'entraînement...")
    history = trainer.train()
    
    print("Entraînement terminé!")
    print(f"Meilleure précision: {max(history['accuracies']):.4f}")

if __name__ == "__main__":
    asyncio.run(main())