import asyncio
import time
import sqlite3
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
LOOKBACK = 50            # Window for Option A Mean & Standard Deviation
Z_ENTRY_THRESHOLD = 1.8  # Option A Signal threshold
SYMBOL_LIMIT = 50        # Perpetual contracts to scan
TIMEFRAME = '1h'        
LEVERAGE = 5             # 5x Leverage
RISK_PER_TRADE = 0.02    # Risk 2% of account balance per trade
STOP_LOSS_PCT = 0.02     # 2% Stop Loss
TAKE_PROFIT_PCT = 0.04   # 4% Take Profit

# OPTION C CONFIGURATION (BREAKOUT ENGINE)
VOLUME_MA_LOOKBACK = 20  # Moving average period for volume
VOLUME_MULTIPLE = 2.5    # Volume surge threshold (2.5x normal volume)

INITIAL_PAPER_BALANCE = 10000.0  # Starting Virtual USD Balance
DB_NAME = 'crypto_paper_trading.db'
# =================================================

class DualStrategyFuturesBot:
    def __init__(self):
        config = {'enableRateLimit': True, 'options': {'defaultType': 'future'}}
        self.exchange = ccxt.binance(config)
        self.init_db()

    def init_db(self):
        """Initializes SQLite tables for Option A, Option C, and paper trading."""
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Signals Logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT, -- 'OPTION_A' or 'OPTION_C'
                symbol TEXT,
                price REAL,
                metric_val REAL,
                signal_type TEXT
            )
        ''')

        # Paper Trades Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
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

        # Paper Portfolio Table
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
            
            # --- OPTION A CALCULATIONS (Z-Score) ---
            rolling_mean = df['close'].rolling(LOOKBACK).mean()
            rolling_std = df['close'].rolling(LOOKBACK).std()
            latest_std = rolling_std.iloc[-1]
            latest_close = df['close'].iloc[-1]
            
            z_score = 0.0
            if latest_std > 0 and not np.isnan(latest_std):
                z_score = (latest_close - rolling_mean.iloc[-1]) / latest_std

            # --- OPTION C CALCULATIONS (Volume Breakout) ---
            vol_ma = df['volume'].rolling(VOLUME_MA_LOOKBACK).mean().iloc[-1]
            latest_vol = df['volume'].iloc[-1]
            vol_ratio = (latest_vol / vol_ma) if vol_ma > 0 else 0.0

            period_high = df['high'].iloc[-21:-1].max()
            period_low = df['low'].iloc[-21:-1].min()

            option_c_signal = None
            if vol_ratio >= VOLUME_MULTIPLE:
                if latest_close > period_high:
                    option_c_signal = 'LONG'
                elif latest_close < period_low:
                    option_c_signal = 'SHORT'

            return {
                'symbol': symbol,
                'price': latest_close,
                'z_score': round(z_score, 4),
                'vol_ratio': round(vol_ratio, 2),
                'option_c_signal': option_c_signal
            }
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

    def process_paper_orders(self, df: pd.DataFrame):
        if df.empty:
            return

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # 1. Fetch current open positions
        cursor.execute("SELECT id, symbol, side, entry_price, amount FROM paper_trades WHERE status = 'OPEN'")
        open_positions = cursor.fetchall()
        current_prices = dict(zip(df['symbol'], df['price']))

        cash_balance = self.get_portfolio_state()
        total_unrealized_pnl = 0.0

        # 2. Evaluate active positions for SL / TP
        for trade_id, symbol, side, entry_price, amount in open_positions:
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            pnl_pct = (current_price - entry_price) / entry_price if side == 'LONG' else (entry_price - current_price) / entry_price
            unrealized_pnl = amount * entry_price * pnl_pct * LEVERAGE
            total_unrealized_pnl += unrealized_pnl

            if pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT:
                cash_balance += unrealized_pnl
                cursor.execute('''
                    UPDATE paper_trades
                    SET exit_price = ?, realized_pnl = ?, status = 'CLOSED', exit_time = ?
                    WHERE id = ?
                ''', (current_price, round(unrealized_pnl, 4), timestamp, trade_id))

        # 3. Process Signals for Option A and Option C
        cursor.execute("SELECT DISTINCT symbol FROM paper_trades WHERE status = 'OPEN'")
        active_symbols = {row[0] for row in cursor.fetchall()}

        for _, row in df.iterrows():
            symbol = row['symbol']
            price = row['price']
            
            if symbol in active_symbols:
                continue

            # Option A Signal Check
            if row['z_score'] <= -Z_ENTRY_THRESHOLD:
                self._execute_paper_trade(cursor, 'OPTION_A', symbol, 'LONG', price, cash_balance, timestamp)
                active_symbols.add(symbol)
            elif row['z_score'] >= Z_ENTRY_THRESHOLD:
                self._execute_paper_trade(cursor, 'OPTION_A', symbol, 'SHORT', price, cash_balance, timestamp)
                active_symbols.add(symbol)

            # Option C Breakout Signal Check
            if row['option_c_signal'] and symbol not in active_symbols:
                self._execute_paper_trade(cursor, 'OPTION_C', symbol, row['option_c_signal'], price, cash_balance, timestamp)
                active_symbols.add(symbol)

        # 4. Record Portfolio State
        total_equity = cash_balance + total_unrealized_pnl
        cursor.execute('''
            INSERT INTO paper_portfolio (timestamp, cash_balance, unrealized_pnl, total_equity)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, round(cash_balance, 2), round(total_unrealized_pnl, 2), round(total_equity, 2)))

        conn.commit()
        conn.close()

        print(f"Portfolio Equity: ${total_equity:,.2f} | Cash: ${cash_balance:,.2f} | Unrealized PnL: ${total_unrealized_pnl:,.2f}")

    def _execute_paper_trade(self, cursor, strategy, symbol, side, price, balance, timestamp):
        position_value = balance * RISK_PER_TRADE * LEVERAGE
        amount = position_value / price
        cursor.execute('''
            INSERT INTO paper_trades (strategy, symbol, side, entry_price, amount, leverage, status, entry_time)
            VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?)
        ''', (strategy, symbol, side, price, round(amount, 6), LEVERAGE, timestamp))
        print(f"[{strategy} {side} PAPER TRADE] {symbol} @ ${price} | Size: ${position_value:.2f}")

    async def run_once(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running Dual-Strategy Scan (Option A + Option C)...")
        try:
            df = await self.scan_market()
            self.process_paper_orders(df)
        except Exception as e:
            print(f"Scan Error: {e}")
        finally:
            await self.exchange.close()

if __name__ == '__main__':
    bot = DualStrategyFuturesBot()
    asyncio.run(bot.run_once())
