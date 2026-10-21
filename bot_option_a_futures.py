import asyncio
import time
import sqlite3
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
LOOKBACK = 50            # Window for Mean & Standard Deviation
Z_ENTRY_THRESHOLD = 1.8  # Signal threshold
SYMBOL_LIMIT = 50        # Perpetual contracts to scan
TIMEFRAME = '1h'        
LEVERAGE = 5             # 5x Leverage
RISK_PER_TRADE = 0.02    # Risk 2% of account balance per trade
STOP_LOSS_PCT = 0.02     # 2% Stop Loss
TAKE_PROFIT_PCT = 0.04   # 4% Take Profit

INITIAL_PAPER_BALANCE = 10000.0  # Starting Virtual USD Balance
DB_NAME = 'crypto_paper_trading.db'
# =================================================

class PaperFuturesQuantitativeBot:
    def __init__(self):
        # Public market data initialization (No API Key Required)
        config = {'enableRateLimit': True, 'options': {'defaultType': 'future'}}
        self.exchange = ccxt.binance(config)
        self.init_db()

    def init_db(self):
        """Initializes SQLite database for signals, paper trades, and portfolio tracking."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # 1. Signals Log
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

        # 2. Paper Trades Table (Active & Closed Positions)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                amount REAL,
                leverage INTEGER,
                realized_pnl REAL,
                status TEXT, -- 'OPEN' or 'CLOSED'
                entry_time TEXT,
                exit_time TEXT
            )
        ''')

        # 3. Paper Account Balance Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cash_balance REAL,
                unrealized_pnl REAL,
                total_equity REAL
            )
        ''')

        # Seed initial account balance if empty
        cursor.execute("SELECT COUNT(*) FROM paper_portfolio")
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO paper_portfolio (timestamp, cash_balance, unrealized_pnl, total_equity)
                VALUES (?, ?, 0.0, ?)
            ''', (time.strftime('%Y-%m-%d %H:%M:%S'), INITIAL_PAPER_BALANCE, INITIAL_PAPER_BALANCE))

        conn.commit()
        conn.close()

    def get_portfolio_state(self):
        """Fetches current cash balance and calculate equity."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT cash_balance FROM paper_portfolio ORDER BY id DESC LIMIT 1")
        cash_balance = cursor.fetchone()[0]
        conn.close()
        return cash_balance

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

    def process_paper_orders_and_pnl(self, current_prices: dict, signals_df: pd.DataFrame):
        """Executes simulated purchases, monitors active trades for SL/TP, and updates PnL."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # 1. Fetch current open paper positions
        cursor.execute("SELECT id, symbol, side, entry_price, amount FROM paper_trades WHERE status = 'OPEN'")
        open_positions = cursor.fetchall()

        total_unrealized_pnl = 0.0
        cash_balance = self.get_portfolio_state()

        # 2. Check open positions for Stop-Loss / Take-Profit exit triggers
        for trade_id, symbol, side, entry_price, amount in open_positions:
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            
            # Compute price return percentage
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            unrealized_pnl = amount * entry_price * pnl_pct * LEVERAGE
            total_unrealized_pnl += unrealized_pnl

            # Check SL / TP conditions
            close_trade = False
            if pnl_pct <= -STOP_LOSS_PCT:
                print(f"[STOP LOSS TRIGGERED] Closing {side} on {symbol} at ${current_price:.4f} (PnL: ${unrealized_pnl:.2f})")
                close_trade = True
            elif pnl_pct >= TAKE_PROFIT_PCT:
                print(f"[TAKE PROFIT TRIGGERED] Closing {side} on {symbol} at ${current_price:.4f} (PnL: ${unrealized_pnl:.2f})")
                close_trade = True

            if close_trade:
                realized_pnl = unrealized_pnl
                cash_balance += realized_pnl
                cursor.execute('''
                    UPDATE paper_trades
                    SET exit_price = ?, realized_pnl = ?, status = 'CLOSED', exit_time = ?
                    WHERE id = ?
                ''', (current_price, round(realized_pnl, 4), timestamp, trade_id))

        # 3. Process new entry signals (Avoid duplicate active entries for same symbol)
        cursor.execute("SELECT DISTINCT symbol FROM paper_trades WHERE status = 'OPEN'")
        active_symbols = {row[0] for row in cursor.fetchall()}

        if not signals_df.empty:
            for _, row in signals_df.iterrows():
                symbol = row['symbol']
                price = row['price']
                z_score = row['z_score']
                side = 'LONG' if z_score <= -Z_ENTRY_THRESHOLD else 'SHORT'

                if symbol in active_symbols:
                    continue  # Already in an active trade for this symbol

                # Calculate trade position sizing (Risking 2% of account balance)
                position_value = cash_balance * RISK_PER_TRADE * LEVERAGE
                amount = position_value / price

                cursor.execute('''
                    INSERT INTO paper_trades (symbol, side, entry_price, amount, leverage, status, entry_time)
                    VALUES (?, ?, ?, ?, ?, 'OPEN', ?)
                ''', (symbol, side, price, round(amount, 6), LEVERAGE, timestamp))

                print(f"[PAPER {side} EXECUTED] {symbol} | Entry: ${price} | Position Size: ${position_value:.2f} | Z-Score: {z_score}")

        # 4. Save updated portfolio state
        total_equity = cash_balance + total_unrealized_pnl
        cursor.execute('''
            INSERT INTO paper_portfolio (timestamp, cash_balance, unrealized_pnl, total_equity)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, round(cash_balance, 2), round(total_unrealized_pnl, 2), round(total_equity, 2)))

        conn.commit()
        conn.close()

        print(f"\n--- PORTFOLIO SUMMARY ---")
        print(f"Cash Balance:   ${cash_balance:,.2f}")
        print(f"Unrealized PnL: ${total_unrealized_pnl:,.2f}")
        print(f"Total Equity:   ${total_equity:,.2f}\n")

    async def run_once(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing Quantitative Paper Scan...")
        try:
            df = await self.scan_market()
            if df.empty:
                print("No market data fetched.")
                return

            current_prices = dict(zip(df['symbol'], df['price']))
            signals = df[(df['z_score'] <= -Z_ENTRY_THRESHOLD) | (df['z_score'] >= Z_ENTRY_THRESHOLD)]

            # Log quantitative signals
            if not signals.empty:
                self.log_signals_to_db(signals)

            # Run paper execution engine
            self.process_paper_orders_and_pnl(current_prices, signals)

        except Exception as e:
            print(f"Execution Error: {e}")
        finally:
            await self.exchange.close()

if __name__ == '__main__':
    bot = PaperFuturesQuantitativeBot()
    asyncio.run(bot.run_once())
