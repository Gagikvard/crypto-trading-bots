import time
import asyncio
import sqlite3
import ccxt.async_support as ccxt
import pandas as pd
from tradingview_screener import crypto, col

# ================= CONFIGURATION =================
RSI_LONG_THRESHOLD = 40.0   # Buy signal when RSI <= 40
RSI_SHORT_THRESHOLD = 65.0  # Short signal when RSI >= 65
SYMBOL_LIMIT = 250
LEVERAGE = 5                
RISK_PER_TRADE = 0.02       
STOP_LOSS_PCT = 0.02        
TAKE_PROFIT_PCT = 0.04      

ENABLE_LIVE_TRADING = False # Set to True when ready for live execution
API_KEY = ''                 # Your Binance Futures API Key
SECRET_KEY = ''              # Your Binance Futures API Secret
DB_NAME = 'crypto_trading_data.db'
# =================================================

class OptionCTradingViewFuturesBot:
    def __init__(self):
        print("\n[OPTION C FUTURES] Initializing TradingView Engine with Database Logging...")
        config = {'enableRateLimit': True, 'options': {'defaultType': 'future'}}
        if ENABLE_LIVE_TRADING and API_KEY and SECRET_KEY:
            config['apiKey'] = API_KEY
            config['secret'] = SECRET_KEY
            
        self.exchange = ccxt.binance(config)
        self.init_db()

    def init_db(self):
        """Initializes SQLite database to record RSI scan data."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_c_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                name TEXT,
                close REAL,
                change REAL,
                rsi REAL,
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
        df_copy['signal_type'] = df_copy['RSI'].apply(lambda rsi: 'LONG' if rsi <= RSI_LONG_THRESHOLD else 'SHORT')
        
        db_df = df_copy[['timestamp', 'name', 'close', 'change', 'RSI', 'signal_type']].rename(columns={'RSI': 'rsi'})
        db_df.to_sql('option_c_signals', conn, if_exists='append', index=False)
        conn.close()
        print(f"[DATABASE] Logged {len(db_df)} technical RSI signals to {DB_NAME}")

    def scan_market(self) -> tuple[int, pd.DataFrame]:
        start_time = time.time()
        q = (crypto()
             .select('name', 'close', 'change', 'volume', 'RSI')
             .where(col('exchange') == 'BINANCE')
             .order_by('volume', ascending=False)
             .limit(SYMBOL_LIMIT))
        
        count, df = q.get_scanner_data()
        elapsed = time.time() - start_time
        print(f"Scan finished in {elapsed:.2f} seconds. Processed {count} total contracts.")
        return count, df

    async def execute_signals(self, df: pd.DataFrame):
        if df.empty:
            return

        futures_df = df[df['name'].str.contains(r'USDT\.P|PERP|USDT', regex=True, na=False)].copy()
        longs = futures_df[futures_df['RSI'] <= RSI_LONG_THRESHOLD].sort_values('RSI', ascending=True)
        shorts = futures_df[futures_df['RSI'] >= RSI_SHORT_THRESHOLD].sort_values('RSI', ascending=False)
        
        all_signals = pd.concat([longs, shorts])

        print(f"\n================ FUTURES SIGNALS GENERATED ================")
        print(f"Retrieved {len(futures_df)} perpetual futures contracts.")
        print(f"Found {len(longs)} LONG setups (RSI <= {RSI_LONG_THRESHOLD})")
        print(f"Found {len(shorts)} SHORT setups (RSI >= {RSI_SHORT_THRESHOLD})")
        
        if not longs.empty:
            print("\n--- TOP LONG CANDIDATES (LOWEST RSI) ---")
            print(longs[['name', 'close', 'change', 'RSI']].head(5).to_string(index=False))

        if not shorts.empty:
            print("\n--- TOP SHORT CANDIDATES (HIGHEST RSI) ---")
            print(shorts[['name', 'close', 'change', 'RSI']].head(5).to_string(index=False))

        if not all_signals.empty:
            self.log_signals_to_db(all_signals)

        if not ENABLE_LIVE_TRADING:
            print(f"\n[DRY RUN MODE] Live execution disabled (Target Leverage: {LEVERAGE}x). No keys required.")

    async def run_once(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing Option C Scan Cycle...")
        try:
            _, df = self.scan_market()
            await self.execute_signals(df)
        except Exception as e:
            print(f"Execution Error: {e}")
        finally:
            await self.exchange.close()

if __name__ == '__main__':
    bot = OptionCTradingViewFuturesBot()
    asyncio.run(bot.run_once())
