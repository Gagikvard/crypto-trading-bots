import asyncio
import time
import sqlite3
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
LOOKBACK = 50           # Window for Mean & Standard Deviation
Z_ENTRY_THRESHOLD = 1.8 # Signal threshold
SYMBOL_LIMIT = 250      # Perpetual contracts to scan
TIMEFRAME = '1h'        
LEVERAGE = 5            # 5x Leverage
RISK_PER_TRADE = 0.02   # Risk 2% of free balance
STOP_LOSS_PCT = 0.02    # 2% Stop Loss
TAKE_PROFIT_PCT = 0.04  # 4% Take Profit

ENABLE_LIVE_TRADING = True
API_KEY = 'YOUR_BINANCE_API_KEY'
SECRET_KEY = 'YOUR_BINANCE_API_SECRET'
DB_NAME = 'crypto_trading_data.db'
# =================================================

class FuturesQuantitativeBot:
    def __init__(self):
        config = {'enableRateLimit': True, 'options': {'defaultType': 'future'}}
        if ENABLE_LIVE_TRADING and API_KEY and SECRET_KEY:
            config['apiKey'] = API_KEY
            config['secret'] = SECRET_KEY
        self.exchange = ccxt.binance(config)
        self.init_db()

    def init_db(self):
        """Initializes SQLite database to store scan history."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_a_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                price REAL,
                z_score REAL,
                signal_type TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_signals_to_db(self, df: pd.DataFrame):
        if df.empty:
            return
        conn = sqlite3.connect(DB_NAME)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        df_copy = df.copy()
        df_copy['timestamp'] = timestamp
        df_copy['signal_type'] = np.where(df_copy['z_score'] <= -Z_ENTRY_THRESHOLD, 'LONG', 'SHORT')
        
        db_df = df_copy[['timestamp', 'symbol', 'price', 'z_score', 'signal_type']]
        db_df.to_sql('option_a_signals', conn, if_exists='append', index=False)
        conn.close()
        print(f"[DATABASE] Logged {len(db_df)} quantitative signals to {DB_NAME}")

    async def fetch_and_evaluate(self, symbol: str) -> dict | None:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LOOKBACK + 10)
            if not ohlcv or len(ohlcv) < LOOKBACK:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            rolling_mean = df['close'].rolling(LOOKBACK).mean()
            rolling_std = df['close'].rolling(LOOKBACK).std()
            
            latest_std = rolling_std.iloc[-1]
            if latest_std == 0 or np.isnan(latest_std):
                return None

            latest_close = df['close'].iloc[-1]
            z_score = (latest_close - rolling_mean.iloc[-1]) / latest_std

            return {'symbol': symbol, 'price': latest_close, 'z_score': round(z_score, 4)}
        except Exception:
            return None

    async def scan_market(self) -> pd.DataFrame:
        markets = await self.exchange.load_markets()
        futures_pairs = [
            s for s, m in markets.items()
            if m.get('swap', False) and m.get('linear', False) and s.endswith('/USDT:USDT')
        ][:SYMBOL_LIMIT]

        tasks = [self.fetch_and_evaluate(sym) for sym in futures_pairs]
        raw_results = await asyncio.gather(*tasks)
        return pd.DataFrame([r for r in raw_results if r is not None])

    async def execute_signals(self, df: pd.DataFrame):
        if df.empty:
            return

        signals = df[(df['z_score'] <= -Z_ENTRY_THRESHOLD) | (df['z_score'] >= Z_ENTRY_THRESHOLD)]
        longs = df[df['z_score'] <= -Z_ENTRY_THRESHOLD].sort_values('z_score', ascending=True)
        shorts = df[df['z_score'] >= Z_ENTRY_THRESHOLD].sort_values('z_score', ascending=False)

        print(f"\n================ FUTURES SIGNALS ================")
        print(f"Oversold (LONG) Signals: {len(longs)}")
        if not longs.empty:
            print(longs.head(5).to_string(index=False))

        print(f"\nOverbought (SHORT) Signals: {len(shorts)}")
        if not shorts.empty:
            print(shorts.head(5).to_string(index=False))

        if not signals.empty:
            self.log_signals_to_db(signals)

        if not ENABLE_LIVE_TRADING:
            print("\n[DRY RUN MODE] Live execution disabled. No exchange keys required.")

    async def run_once(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing Option A Scan Cycle...")
        try:
            df = await self.scan_market()
            await self.execute_signals(df)
        except Exception as e:
            print(f"Execution Error: {e}")
        finally:
            await self.exchange.close()

if __name__ == '__main__':
    bot = FuturesQuantitativeBot()
    asyncio.run(bot.run_once())
