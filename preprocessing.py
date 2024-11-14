import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional

class LogarithmicBuffer:
    def __init__(self, base_interval_seconds=30, max_points=128):
        self.base_interval = base_interval_seconds
        self.max_points = max_points
        
        # Calcul des intervalles logarithmiques
        total_duration = 365 * 24 * 3600  # 1 an en secondes
        factor = np.log(total_duration/base_interval_seconds) / max_points
        self.intervals = [
            int(base_interval_seconds * np.exp(factor * i))
            for i in range(max_points)
        ]
        self.intervals.reverse()
        
        # Buffers
        self.reset()
        
    def reset(self):
        """Réinitialise les buffers"""
        self.data = []
        self.features = []
        self.timestamps = []
        
    def update(self, timestamp: float, data: dict, features: np.ndarray) -> bool:
        """
        Met à jour le buffer avec de nouvelles données
        Returns: True si le buffer est prêt pour les prédictions
        """
        # Vérifie si on doit mettre à jour
        if not self.should_update(timestamp):
            return len(self.timestamps) >= self.max_points
            
        # Met à jour les points pour chaque intervalle
        new_timestamps = []
        new_data = []
        new_features = []
        
        for interval in self.intervals:
            target_time = timestamp - interval
            idx = self._find_closest_time(target_time)
            
            if idx is not None and abs(self.timestamps[idx] - target_time) <= interval * 0.1:
                new_timestamps.append(self.timestamps[idx])
                new_data.append(self.data[idx])
                new_features.append(self.features[idx])
            else:
                new_timestamps.append(target_time)
                new_data.append(data)
                new_features.append(features)
        
        self.timestamps = new_timestamps
        self.data = new_data
        self.features = new_features
        
        return len(self.timestamps) >= self.max_points
    
    def _find_closest_time(self, target_time: float) -> Optional[int]:
        if not self.timestamps:
            return None
        timestamps = np.array(self.timestamps)
        idx = np.abs(timestamps - target_time).argmin()
        return idx
        
    def should_update(self, current_time: float) -> bool:
        if not self.timestamps:
            return True
            
        for interval in self.intervals:
            target_time = current_time - interval
            idx = self._find_closest_time(target_time)
            if idx is None or abs(self.timestamps[idx] - target_time) > interval * 0.1:
                return True
        return False
    
    def get_features(self) -> Optional[np.ndarray]:
        """Retourne les features avec information temporelle"""
        if len(self.features) < self.max_points:
            return None
            
        features_with_time = []
        for i, timestamp in enumerate(self.timestamps):
            # Features temporelles
            dt = datetime.fromtimestamp(timestamp)
            
            # Heure (cyclique)
            hour = dt.hour + dt.minute/60
            time_features = [
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dt.weekday() / 7),
                np.cos(2 * np.pi * dt.weekday() / 7)
            ]
            
            # Combine avec les features techniques
            combined = np.concatenate([
                np.array(time_features),
                self.features[i]
            ])
            features_with_time.append(combined)
            
        return np.array(features_with_time)

class FeatureExtractor:
    def __init__(self, buffer_size=300):
        self.price_buffer = deque(maxlen=buffer_size)
        self.volume_buffer = deque(maxlen=buffer_size)
        self.min_points = 60
        
    def compute_features(self, price: float, volume: float) -> Optional[np.ndarray]:
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        if len(self.price_buffer) < self.min_points:
            return None
            
        features = []
        prices = np.array(list(self.price_buffer))
        volumes = np.array(list(self.volume_buffer))
        returns = np.diff(np.log(prices))
        
        # Prix relatifs et rendements
        features.extend([
            (prices[-1] / prices[-2]) - 1,
            np.mean(returns[-10:]),
            np.mean(returns[-30:])
        ])
        
        # Volatilité réalisée
        for window in [10, 30, 60]:
            vol = np.std(returns[-window:]) * np.sqrt(48*60)
            features.append(vol)
            
        # Moyennes mobiles relatives
        for window in [10, 30, 60]:
            ma = np.mean(prices[-window:])
            features.append(prices[-1] / ma - 1)
            
        # Volume relatif
        vol_ma = np.mean(volumes[-30:])
        features.extend([
            volumes[-1] / vol_ma - 1,
            np.std(volumes[-30:]) / vol_ma
        ])
        
        # ROC
        for window in [10, 30, 60]:
            roc = (prices[-1] / prices[-window] - 1) * 100
            features.append(roc)
            
        # RSI
        delta = np.diff(prices[-15:])
        gains = delta * (delta > 0)
        losses = -delta * (delta < 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)
        
        return np.array(features)

def prepare_data(df: pd.DataFrame, 
                feature_extractor: FeatureExtractor,
                log_buffer: LogarithmicBuffer,
                threshold: float = 0.0015) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prépare les données pour l'entraînement
    Returns: (X, y) avec X les features et y les labels
    """
    X, y = [], []
    
    for i in range(len(df)-1):
        price = df.iloc[i]['close']
        volume = df.iloc[i]['volume']
        timestamp = df.iloc[i]['timestamp'].timestamp()
        
        # Calcul des features
        features = feature_extractor.compute_features(price, volume)
        if features is None:
            continue
            
        # Mise à jour du buffer logarithmique
        data = {'close': price, 'volume': volume}
        if not log_buffer.update(timestamp, data, features):
            continue
            
        # Récupération des features avec contexte
        X_i = log_buffer.get_features()
        if X_i is None:
            continue
            
        # Calcul du label
        next_price = df.iloc[i+1]['close']
        pct_change = (next_price / price) - 1
        
        if pct_change < -threshold:
            label = 0  # Baisse
        elif pct_change > threshold:
            label = 2  # Hausse
        else:
            label = 1  # Stable
            
        X.append(X_i)
        y.append(label)
    
    return np.array(X), np.array(y)