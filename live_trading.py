import asyncio
import time
from datetime import datetime
import json
from typing import Optional
import numpy as np

from preprocessing import FeatureExtractor, LogarithmicBuffer
from models import CryptoTrader
from data_collection import LiveDataCollector

class TradingSystem:
    def __init__(self, model_path: str, symbol: str = 'DOGE/USDT'):
        self.symbol = symbol
        self.trader = CryptoTrader(model_path)
        self.feature_extractor = FeatureExtractor()
        self.log_buffer = LogarithmicBuffer()
        
        # Pour le monitoring
        self.last_price = None
        self.predictions = []
        self.start_time = None
        
    async def on_tick(self, ticker: dict):
        """Callback appelé à chaque nouveau tick"""
        timestamp = time.time()
        price = ticker['last']
        volume = ticker['quoteVolume']
        
        # Extraction des features
        features = self.feature_extractor.compute_features(price, volume)
        if features is None:
            return
            
        # Mise à jour du buffer logarithmique
        data = {'close': price, 'volume': volume}
        if not self.log_buffer.update(timestamp, data, features):
            return
            
        # Prédiction
        X = self.log_buffer.get_features()
        if X is None:
            return
            
        prediction, confidence = self.trader.predict(X)
        
        # Affichage
        self._display_status(price, prediction, confidence)
        
        # Sauvegarde des prédictions
        if prediction is not None:
            self.predictions.append({
                'timestamp': timestamp,
                'price': price,
                'prediction': prediction.tolist(),
                'confidence': float(confidence)
            })
    
    def _display_status(self, price: float, prediction: Optional[np.ndarray], confidence: float):
        """Affiche l'état actuel du système"""
        now = datetime.now()
        
        if self.start_time is None:
            self.start_time = now
            
        runtime = (now - self.start_time).total_seconds()
        
        # Calcul de la variation de prix
        if self.last_price:
            price_change = (price - self.last_price) / self.last_price * 100
        else:
            price_change = 0
        self.last_price = price
        
        # Affichage
        status = f"\rPrix: {price:.6f} USDT ({price_change:+.3f}%) | "
        
        if prediction is not None:
            status += f"Prédiction [conf={confidence:.2f}]: "
            status += f"Baisse: {prediction[0]:.2f} "
            status += f"Stable: {prediction[1]:.2f} "
            status += f"Hausse: {prediction[2]:.2f}"
        else:
            status += "Attente..."
            
        status += f" | Runtime: {runtime:.0f}s"
        print(status, end='')
        
        # Affiche les diagnostics périodiquement
        if len(self.predictions) % 100 == 0 and self.predictions:
            diag = self.trader.get_diagnostic_info()
            print(f"\nDiagnostic: {json.dumps(diag, indent=2)}")
    
    def save_predictions(self, filename: str = 'predictions.json'):
        """Sauvegarde l'historique des prédictions"""
        with open(filename, 'w') as f:
            json.dump(self.predictions, f)
        print(f"\nPrédictions sauvegardées dans {filename}")

async def main():
    # Charge le meilleur modèle
    trading_system = TradingSystem('models/best_model.pth')
    
    # Démarre la collecte en direct
    collector = LiveDataCollector(
        trading_system.on_tick,
        symbol='DOGE/USDT',
        tick_interval=0.5
    )
    
    try:
        print("Démarrage du trading en direct...")
        await collector.start()
    except KeyboardInterrupt:
        print("\nArrêt du système...")
        collector.stop()
        trading_system.save_predictions()

if __name__ == "__main__":
    asyncio.run(main())