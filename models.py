import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # Ajout d'une dimension batch
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]  # Broadcasting sur le batch

class CryptoSTST(nn.Module):
    def __init__(
        self,
        feature_dim: int = 24,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        context_length: int = 128
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
            batch_first=True,
            bidirectional=True  # Ajout du bidirectionnel
        )
        
        # Adapté pour LSTM bidirectionnel
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 pour bidirectionnel
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, feature_dim]
        x = self.feature_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # LSTM bidirectionnel
        x, _ = self.lstm(x)  # Output: [batch_size, seq_len, d_model*2]
        x = x[:, -1, :]      # Prend le dernier état
        
        return self.classifier(x)

class CryptoTrader:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CryptoSTST(feature_dim=24).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Stats pour monitoring
        self.prediction_count = 0
        self.confidence_stats = []
        
    def predict(self, features: np.ndarray, confidence_threshold: float = 0.4) -> Tuple[Optional[np.ndarray], float]:
        """
        Fait une prédiction à partir des features
        Returns: (prédiction, confiance) ou (None, confiance) si sous le seuil
        """
        with torch.no_grad():
            x = torch.from_numpy(features).float()
            x = x.unsqueeze(0).to(self.device)
            output = self.model(x)
            probs = F.softmax(output, dim=1)
            
            prediction = probs.cpu().numpy()[0]
            confidence = np.max(prediction)
            
            self.confidence_stats.append(confidence)
            self.prediction_count += 1
            
            if confidence < confidence_threshold:
                return None, confidence
                
            return prediction, confidence
    
    def get_diagnostic_info(self) -> dict:
        """Retourne les statistiques de prédiction"""
        if not self.confidence_stats:
            return {"status": "Pas assez de données"}
            
        confidence_array = np.array(self.confidence_stats)
        return {
            "total_predictions": self.prediction_count,
            "avg_confidence": float(np.mean(confidence_array)),
            "max_confidence": float(np.max(confidence_array)),
            "min_confidence": float(np.min(confidence_array)),
            "std_confidence": float(np.std(confidence_array))
        }