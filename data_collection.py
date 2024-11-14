import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import time

class HistoricalDataCollector:
    def __init__(self):
        self.exchange = ccxt.bitget({
            'enableRateLimit': True,
        })
    
    async def fetch_historical_data(self, symbol='DOGE/USDT', timeframe='1m', start_date=None, end_date=None):
        """
        Récupère les données historiques depuis Bitget
        Returns: DataFrame avec colonnes timestamp, open, high, low, close, volume
        """
        await self.exchange.load_markets()
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        all_candles = []
        current_timestamp = start_timestamp
        
        try:
            while current_timestamp < end_timestamp:
                candles = await self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_timestamp,
                    limit=1000  # Maximum par requête
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                current_timestamp = candles[-1][0] + 1
                print(f"\rRécupération jusqu'à {datetime.fromtimestamp(current_timestamp/1000)}", end='')
                
                # Respect des limites de l'API
                await asyncio.sleep(self.exchange.rateLimit / 1000)
                
        finally:
            await self.exchange.close()
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    async def fetch_recent_ticks(self, symbol='DOGE/USDT', limit=100):
        """
        Récupère les ticks récents via l'API de trades
        """
        trades = await self.exchange.fetch_trades(symbol, limit=limit)
        df = pd.DataFrame(trades)
        return df[['timestamp', 'price', 'amount']].rename(
            columns={'price': 'close', 'amount': 'volume'}
        )

class LiveDataCollector:
    def __init__(self, callback, symbol='DOGE/USDT', tick_interval=0.5):
        self.exchange = ccxt.bitget({'enableRateLimit': True})
        self.callback = callback
        self.symbol = symbol
        self.tick_interval = tick_interval
        self.is_running = False
        
    async def start(self):
        await self.exchange.load_markets()
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    ticker = await self.exchange.fetch_ticker(self.symbol)
                    await self.callback(ticker)
                    await asyncio.sleep(self.tick_interval)
                except Exception as e:
                    print(f"Erreur lors de la récupération du tick: {e}")
                    await asyncio.sleep(1)
        finally:
            self.is_running = False
            await self.exchange.close()
            
    def stop(self):
        self.is_running = False

async def main():
    # Exemple d'utilisation
    collector = HistoricalDataCollector()
    
    # Récupération des données du dernier mois
    start_date = datetime.now() - timedelta(days=30)
    hist_data = await collector.fetch_historical_data(
        start_date=start_date,
        timeframe='1m'
    )
    print(f"\nDonnées récupérées: {len(hist_data)} points")
    hist_data.to_csv('historical_data.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())