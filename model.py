import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import ccxt.async_support as ccxt
import asyncio

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2400):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class CryptoSTST(nn.Module):
    def __init__(
        self,
        feature_dim=10,  # Nombre de features techniques
        d_model=64,      # Dimension du modèle
        nhead=4,         # Nombre de têtes d'attention
        num_layers=4,    # Nombre de couches transformer
        dim_feedforward=256,
        dropout=0.1,
        context_length=128  # ~1 minute de données
    ):
        super().__init__()
        
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 3)  # 3 classes: baisse, stable, hausse
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        x = self.feature_projection(x)
        x = self.pos_encoder(x)
        
        # Attention spatiotemporelle
        mask = torch.zeros(x.size(0), x.size(1), x.size(1), dtype=bool)
        x = self.transformer_encoder(x)
        
        # LSTM pour capturer les dépendances temporelles
        x, _ = self.lstm(x)
        
        # On utilise seulement le dernier état pour la prédiction
        x = x[:, -1, :]
        
        return self.classifier(x)

class CryptoFeatureExtractor:
    def __init__(self, window_size=128):
        self.window_size = window_size
        self.price_buffer = deque(maxlen=window_size)
        self.volume_buffer = deque(maxlen=window_size)
        
    def compute_features(self, price, volume):
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        if len(self.price_buffer) < self.window_size:
            return None
            
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        
        features = []
        
        # Prix relatifs
        features.append((prices[-1] / prices[-2]) - 1)  # Rendement instantané
        
        # Moyennes mobiles
        for window in [5, 15, 30, 60]:
            ma = prices[-window:].mean()
            features.append(prices[-1] / ma - 1)
        
        # Volatilité réalisée
        for window in [10, 30, 60]:
            returns = np.diff(np.log(prices[-window:]))
            vol = np.std(returns) * np.sqrt(2400)  # Annualisée
            features.append(vol)
        
        # Volume moyen
        vol_ma = volumes[-30:].mean()
        features.append(volumes[-1] / vol_ma - 1)
        
        # RSI
        returns = np.diff(prices)
        gains = np.sum(returns[-14:] * (returns[-14:] > 0))
        losses = -np.sum(returns[-14:] * (returns[-14:] < 0))
        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)
        
        return np.array(features)

class CryptoTrader:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CryptoSTST().to(self.device)
        self.feature_extractor = CryptoFeatureExtractor()
        self.context_window = deque(maxlen=128)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def update_features(self, price, volume):
        features = self.feature_extractor.compute_features(price, volume)
        if features is not None:
            self.context_window.append(features)
            
    def predict(self):
        if len(self.context_window) < 128:
            return None
            
        with torch.no_grad():
            x = torch.tensor(list(self.context_window), dtype=torch.float32)
            x = x.unsqueeze(0).to(self.device)  # Add batch dimension
            output = self.model(x)
            probs = F.softmax(output, dim=1)
            return probs.cpu().numpy()[0]

    async def trading_loop(self):
        exchange = ccxt.bitget()
        await exchange.load_markets()
        
        try:
            while True:
                try:
                    ticker = await exchange.fetch_ticker('DOGE/USDT')
                    price = ticker['last']
                    volume = ticker['quoteVolume']
                    
                    self.update_features(price, volume)
                    prediction = self.predict()
                    
                    if prediction is not None:
                        print(f"\rPrix: {price:.6f} | Prédiction: Baisse: {prediction[0]:.2f} "
                              f"Stable: {prediction[1]:.2f} Hausse: {prediction[2]:.2f}", end='')
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"\nErreur: {e}")
                    await asyncio.sleep(1)
                    
        finally:
            await exchange.close()

# Fonction d'aide pour l'entraînement
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')



# Instancier le trader
trader = CryptoTrader()

# Démarrer la boucle de trading
asyncio.run(trader.trading_loop())