import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ib_insync
from ib_insync import *
import asyncio
import nest_asyncio
nest_asyncio.apply()
import gymnasium as gym
import praw
import pandas_ta as ta
from tqdm import tqdm
import time
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import sys
import os
import requests
from bs4 import BeautifulSoup
import torch as th
import tweepy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import ray.tune as tune

logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
executed_lines = set()
SCRIPT_PATH = os.path.abspath(__file__)

def trace_lines(frame, event, arg):
    if event == 'line' and frame.f_code.co_filename == SCRIPT_PATH:
        executed_lines.add(frame.f_lineno)
    return trace_lines

class TradingEnv(gym.Env):
    def __init__(self, bot):
        super(TradingEnv, self).__init__()
        self.bot = bot
        self.action_space = gym.spaces.Discrete(5)  # BUY, SELL, SHORT, COVER, HOLD
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(91,))
        self.current_step = 0
        self.max_steps = 300  # 5-minute sim

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.bot.positions.clear()
        self.bot.trades.clear()
        self.bot.cash = self.bot.initial_cash
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        reward = 0
        done = self.current_step >= self.max_steps
        info = {}
        if action != 4:  # Not HOLD
            ranked_assets = self.bot.rank_assets(self.bot.dfs, self.bot.vix_df, self.current_step)
            if ranked_assets:
                key, action_type, _ = ranked_assets[0]
                idx = min(self.current_step, len(self.bot.dfs[key]) - 1)
                price = self.bot.dfs[key].iloc[idx]['close']
                qty = (self.bot.cash * 0.02) / price
                self.bot.simulate_trade(key, action_type, price, datetime.now(), qty, self.bot.dfs[key].iloc[idx]['ATR'], self.current_step)
                portfolio_value = self.bot.cash + sum(qty * self.bot.dfs[k].iloc[-1]['close'] for k, qty in self.bot.positions.items() if qty != 0)
                reward = (portfolio_value - self.bot.initial_cash) / self.bot.initial_cash - 0.05 * abs(sum(qty * price for k, qty in self.bot.positions.items()))
        return self._get_observation(), reward, done, False, info

    def _get_observation(self):
        state = []
        for key in self.bot.limited_contracts:
            if key in self.bot.dfs:
                df = self.bot.dfs[key]
                idx = min(self.current_step, len(df) - 1)
                state.extend([
                    df['RSI'].iloc[idx] if not pd.isna(df['RSI'].iloc[idx]) else 50,
                    df['MACD'].iloc[idx] if not pd.isna(df['MACD'].iloc[idx]) else 0,
                    self.bot.positions.get(key, 0)
                ])
        state.append(self.bot.cash)
        return np.array(state)

class TradingBot:
    def __init__(self, simulate=True):
        self.ib = IB()
        self.simulate = simulate
        self.market_type = 'bull'
        self.daily_limit = 60
        self.cash = 500000
        self.initial_cash = 500000
        self.cash_limit_entry = 350000
        self.cash_limit_exit = 50000
        self.positions = {}
        self.trades = {}
        self.trade_count = {}
        self.total_trades = 0
        self.max_trades_per_symbol = 5
        self.fetch_count = 0
        self.stocks = {symbol: Stock(symbol.replace('BRK.B', 'BRK B'), 'SMART', 'USD') for symbol in [
            'AAPL', 'TSLA', 'NVDA', 'META', 'GOOGL', 'JPM', 'BRK.B', 'VOO', 'XLF', 'MSFT', 'QQQ', 'XPEV',
            'EBAY', 'VST', 'QUBT', 'SEZL', 'MU', 'PLTR', 'AXON', 'APP', 'CRWD', 'ORCL', 'LRCX', 'NOW',
            'SHOP', 'SMCI', 'TSM', 'NFLX', 'CRM', 'AVGO', 'GOOG', 'SNOW', 'AMAT', 'ASML', 'AMD', 'INTC',
            'CSCO', 'IBM', 'ADBE', 'PYPL', 'UBER', 'LYFT', 'ZM', 'DOCU', 'WMT', 'TGT', 'COST',
            'HD', 'LOW', 'NKE', 'LULU', 'DIS', 'CMCSA', 'T', 'VZ', 'PFE', 'JNJ', 'MRK', 'GILD', 'BA',
            'LMT', 'RTX', 'CAT', 'DE', 'MMM', 'GE', 'HON', 'UNH', 'CVS', 'CI', 'GS', 'SQ'
        ]}
        self.leveraged_etfs = {symbol: Stock(symbol, 'SMART', 'USD') for symbol in ['SSO', 'QLD', 'UWM', 'UPRO', 'TQQQ', 'TNA']}
        self.inverted_leveraged_etfs = {symbol: Stock(symbol, 'SMART', 'USD') for symbol in ['SDS', 'QID', 'TWM', 'SPXU', 'SQQQ', 'SRTY']}
        self.forex = {symbol: Forex(symbol.replace('.', '')) for symbol in ['EUR.USD', 'USD.JPY', 'GBP.USD']}
        self.crypto = {symbol: Crypto(symbol, 'USD', 'KRAKEN') for symbol in ['BTC', 'ETH', 'BNB', 'SOL']}
        self.commodities = {symbol: Stock(symbol, 'SMART', 'USD') for symbol in ['GLD', 'SLV', 'USO', 'DBA']}
        self.bonds = {symbol: Stock(symbol, 'SMART', 'USD') for symbol in ['TLT', 'SHY']}
        self.international = {symbol: Stock(symbol, 'SMART', 'USD') for symbol in ['EEM', 'EFA']}
        self.volatility = {'VXX': Stock('VXX', 'SMART', 'USD')}
            'international': list(self.international.keys()),
            'volatility': list(self.volatility.keys()),
            'stock_options': list(self.stock_puts.keys()),
            'index_options': ['SPY_CALL', 'SPY_PUT', 'VIX_CALL', 'VIX_PUT']
        }
        self.sectors = {
            'tech': ['AAPL', 'TSLA', 'NVDA', 'META', 'GOOGL', 'MSFT', 'QQQ', 'MU', 'PLTR', 'AXON', 'APP', 'CRWD', 'ORCL', 'LRCX', 'NOW', 'SHOP', 'SMCI', 'TSM', 'NFLX', 'CRM', 'AVGO', 'GOOG', 'SNOW', 'AMAT', 'ASML', 'AMD', 'INTC', 'CSCO', 'IBM', 'ADBE', 'PYPL', 'UBER', 'LYFT', 'ZM', 'DOCU', 'SQ'],
            'finance': ['JPM', 'BRK.B', 'VOO', 'XLF', 'GS'],
            'retail': ['WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'LULU', 'EBAY'],
            'media': ['DIS', 'CMCSA', 'T', 'VZ'],
            'health': ['PFE', 'JNJ', 'MRK', 'GILD', 'UNH', 'CVS', 'CI'],
            'industrial': ['BA', 'LMT', 'RTX', 'CAT', 'DE', 'MMM', 'GE', 'HON'],
            'other': ['XPEV', 'VST', 'QUBT', 'SEZL']
        }
        self.dfs = {}
        self.large_trades = []
        self.last_large_trade_check = datetime.now()
        self.trade_steps = {}
        self.last_trade_time = None
        self.current_group_idx = 0
        self.hold_counter = 0
        self.spy_price_history = []
        self.vix_price_history = []
        self.last_spike_time = 0
        self.last_check_step = {}
        self.api_usage = {
            'twitter': {'count': 0, 'limit': 10000, 'reset_date': '2025-05-01'},
            'newsdata': {'count': 0, 'limit': 200},
            'bloomberg': {'count': 0, 'limit': 1000},
            'ft': {'count': 0, 'limit': 1000},
            'wallstreetjournal': {'count': 0, 'limit': 1000},
            'cnbc': {'count': 0, 'limit': 1000},
            'reddit': {'count': 0, 'limit': float('inf')},
            'stocktwits': {'count': 0, 'limit': 1000},
            'yahoo': {'count': 0, 'limit': 1000},
            'google_news': {'count': 0, 'limit': 1000},
            'reuters': {'count': 0, 'limit': 500},
            'seeking_alpha': {'count': 0, 'limit': 1000},
            'marketwatch': {'count': 0, 'limit': 1000},
            'benzinga': {'count': 0, 'limit': 1000},
            'finviz': {'count': 0, 'limit': 1000},
            'investing': {'count': 0, 'limit': 1000},
            'motleyfool': {'count': 0, 'limit': 1000}
        }
        self.source_priority = ['twitter', 'newsdata', 'bloomberg', 'ft', 'wallstreetjournal', 'cnbc', 'reddit', 'stocktwits', 'yahoo', 'google_news', 'reuters', 'seeking_alpha', 'marketwatch', 'benzinga', 'finviz', 'investing', 'motleyfool']
        try:
            self.twitter_client = tweepy.Client(
                bearer_token=os.getenv('TWITTER_BEARER_TOKEN', 'AAAAAAAAAAAAAAAAAAAAAFXm0AEAAAAAjRIyhuYI6uujMiUu6fPoincO7uY%3DC3SqRxZrqknqCcjPEkugN74AC3cxaGvYYQ6qcJpizrztGaNCnC'),
                consumer_key=os.getenv('TWITTER_CONSUMER_KEY', 'yY5CWWwZ6uI4YSJWSQlTCw9Am'),
                consumer_secret=os.getenv('TWITTER_CONSUMER_SECRET', 'wUmqOHZ7FikUnaWgjiW5Oci2LCfqUEue4tKf2YCfBujkncu6Wt'),
                access_token=os.getenv('TWITTER_ACCESS_TOKEN', '1897754560929660928-EkB7svYkQcTxib7LMhHMDCmmYfxAnT'),
                access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET', 'JCVldDK3zYbfXwdVUNPneJ02f94mJIJ4dEWoaEwYP2onW')
            )
        except Exception as e:
            print(f"Failed to initialize Twitter client: {e}", flush=True)
            logging.error(f"Failed to initialize Twitter client: {e}")
            self.twitter_client = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
        try:
            for contract in self.limited_contracts[:5]:
                self.sentiment_cache[contract] = {
                    'twitter': {'sentiment': self.fetch_twitter_sentiment(contract), 'timestamp': datetime.now()},
                    'newsdata': {'sentiment': self.fetch_news_sentiment(contract), 'timestamp': datetime.now()},
                    'bloomberg': {'sentiment': self.fetch_bloomberg_sentiment(contract), 'timestamp': datetime.now()},
                    'ft': {'sentiment': self.fetch_ft_sentiment(contract), 'timestamp': datetime.now()},
                    'wallstreetjournal': {'sentiment': self.fetch_wallstreetjournal_sentiment(contract), 'timestamp': datetime.now()},
                    'cnbc': {'sentiment': self.fetch_cnbc_sentiment(contract), 'timestamp': datetime.now()},
                    'reddit': {'sentiment': self.fetch_reddit_sentiment(contract), 'timestamp': datetime.now()},
                    'stocktwits': {'sentiment': self.fetch_stocktwits_sentiment(contract), 'timestamp': datetime.now()},
                    'yahoo': {'sentiment': self.fetch_yahoo_sentiment(contract), 'timestamp': datetime.now()},
                    'google_news': {'sentiment': self.fetch_google_news_sentiment(contract), 'timestamp': datetime.now()},
                    'reuters': {'sentiment': self.fetch_reuters_sentiment(contract), 'timestamp': datetime.now()},
                    'seeking_alpha': {'sentiment': self.fetch_seeking_alpha_sentiment(contract), 'timestamp': datetime.now()},
                    'marketwatch': {'sentiment': self.fetch_marketwatch_sentiment(contract), 'timestamp': datetime.now()},
                    'benzinga': {'sentiment': self.fetch_benzinga_sentiment(contract), 'timestamp': datetime.now()},
                    'finviz': {'sentiment': self.fetch_finviz_sentiment(contract), 'timestamp': datetime.now()},
                    'investing': {'sentiment': self.fetch_investing_sentiment(contract), 'timestamp': datetime.now()},
                    'motleyfool': {'sentiment': self.fetch_motleyfool_sentiment(contract), 'timestamp': datetime.now()}
                }
        except Exception as e:
            print(f"Failed to pre-cache sentiment: {e}", flush=True)
            logging.error(f"Failed to pre-cache sentiment: {e}")
        self.env = TradingEnv(self)
        self.ppo = None

    def connect(self):
        if not self.simulate:
            self.ib.connect('127.0.0.1', 7497, clientId=1)
            logging.info("Connected to Interactive Brokers Paper Account")

    def disconnect(self):
        if not self.simulate:
            self.ib.disconnect()
            logging.info("Disconnected from Interactive Brokers")

    def fetch_historical_data(self, contract, duration='7d', bar_size='5m'):
        ticker_map = {**{k: k for k in self.stocks.keys()}, **{k: k for k in self.leveraged_etfs.keys()},
                      **{k: k for k in self.inverted_leveraged_etfs.keys()}, 'EUR.USD': 'EURUSD=X', 
                      'USD.JPY': 'USDJPY=X', 'GBP.USD': 'GBPUSD=X', 'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 
                      'BNB': 'BNB-USD', 'SOL': 'SOL-USD', 'SPY': 'SPY', 'VIX': '^VIX', 
                      **{k: k for k in self.commodities.keys()}, **{k: k for k in self.bonds.keys()},
                      **{k: k for k in self.international.keys()}, 'VXX': 'VXX',
                      **{k: k.split('_')[0] for k in self.stock_puts.keys()}}
        ticker = ticker_map.get(contract if isinstance(contract, str) else contract.symbol, 'SPY')
        if self.simulate:
            for attempt in range(3):  # Retry logic
                try:
                    df = yf.download(ticker, period=duration, interval=bar_size, progress=False)
                    if df.empty:
                        logging.warning(f"No data fetched for {ticker} on attempt {attempt + 1}, generating synthetic data")
                        return self.generate_synthetic_data(contract, duration, bar_size)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0].lower() for col in df.columns]
                    df = df.reset_index().rename(columns={'Datetime': 'date'})
                    return df.tail(5000)
                except Exception as e:
                    logging.error(f"Error fetching data for {ticker} on attempt {attempt + 1}: {e}")
                    if attempt < 2:
                        time.sleep(5)  # Wait before retry
                    else:
                        logging.warning(f"Failed after 3 attempts for {ticker}, generating synthetic data")
                        return self.generate_synthetic_data(contract, duration, bar_size)
        else:
            bars = self.ib.reqHistoricalData(contract, endDateTime='', durationStr=duration, 
                                             barSizeSetting=bar_size, whatToShow='TRADES', useRTH=True, formatDate=1)
            df = util.df(bars)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df

    def generate_synthetic_data(self, contract, duration='7d', bar_size='5m'):
        np.random.seed(42)
        duration_map = {'1d': 1440, '7d': 10080, '30d': 43200}
        periods = duration_map.get(duration, 10080)
        bar_size_map = {'1m': '1min', '5m': '5min', '1h': '1h'}
        freq = bar_size_map.get(bar_size, '5min')
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        base_price = {'SPY': 550, 'VIX': 20, 'EUR.USD': 1.2, 'BTC': 58000}.get(contract.symbol if not isinstance(contract, str) else contract, 150)
        prices = [base_price]
        for _ in range(periods - 1):
            trend = random.uniform(-0.05, 0.05)
            change = trend + np.random.normal(0, 0.02)
            prices.append(max(prices[-1] * (1 + change), 0.01))
        df = pd.DataFrame({'date': dates, 'close': prices})
        noise_scale = max(0.01 * abs(df['close'].mean()), 0.01)
        df['open'] = df['close'].shift(1).fillna(prices[0]) + np.random.normal(0, noise_scale, periods)
        df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0, 0.01 * df['close'].mean(), periods)
        df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0, 0.01 * df['close'].mean(), periods)
        df['volume'] = np.random.randint(500000, 3000000, periods)
        return df

    def fetch_twitter_sentiment(self, symbol):
        print(f"Fetching Twitter sentiment for {symbol}...", flush=True)
        if datetime.now() < datetime.strptime(self.api_usage['twitter']['reset_date'], '%Y-%m-%d'):
            print(f"Twitter limit reached, skipping until reset on {self.api_usage['twitter']['reset_date']}", flush=True)
            return None  # Return None to indicate failure
        try:
            tweets = self.twitter_client.search_recent_tweets(query=symbol, max_results=10)
            if not tweets.data:
                return 0
            sentiment_score = 0
            count = 0
            for tweet in tweets.data:
                text = tweet.text
                sentiment = self.sentiment_analyzer.polarity_scores(text)['compound']
                sentiment_score += sentiment
                count += 1
            self.api_usage['twitter']['count'] += 1
            return sentiment_score / count if count > 0 else 0
        except Exception as e:
            print(f"Twitter fetch failed for {symbol}: {e}", flush=True)
            return None  # Return None on failure

    def fetch_news_sentiment(self, symbol):
        print(f"Fetching news sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and (datetime.now() - self.sentiment_cache[symbol]['newsdata']['timestamp']).total_seconds() < 3600:
            print(f"Using cached sentiment for {symbol}", flush=True)
            return self.sentiment_cache[symbol]['newsdata']['sentiment']
        time.sleep(1)
        try:
            url = "https://newsdata.io/api/1/news"
            params = {
                'q': symbol,
                'language': 'en',
                'apikey': 'pub_76679d9ff0f9015a44f80b10f0016a4b757f7'
            }
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, timeout=5)
                    response.raise_for_status()
                    break
                except requests.Timeout:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    logging.error(f"NewsData.io timeout for {symbol} after 3 attempts")
                    print(f"NewsData.io timeout for {symbol} after 3 attempts", flush=True)
                    return None
            news = response.json()
            articles = news.get('results', [])
            sentiment_score = 0
            count = 0
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text:
                    sentiment = self.sentiment_analyzer.polarity_scores(text)['compound']
                    sentiment_score += sentiment
                    count += 1
            sentiment_score = sentiment_score / count if count > 0 else 0
            self.sentiment_cache[symbol]['newsdata'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['newsdata']['count'] += 1
            print(f"Completed news sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            logging.error(f"Error fetching news sentiment for {symbol}: {e}")
            print(f"Error fetching news sentiment for {symbol}: {e}", flush=True)
            return None  # Return None on failure

    def fetch_reuters_sentiment(self, symbol):
        print(f"Fetching Reuters sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and (datetime.now() - self.sentiment_cache[symbol]['reuters']['timestamp']).total_seconds() < 3600:
            print(f"Using cached sentiment for {symbol}", flush=True)
            return self.sentiment_cache[symbol]['reuters']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Reuters sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol]['reuters'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['reuters']['count'] += 1
            print(f"Completed Reuters sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Reuters fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_reddit_sentiment(self, symbol):
        print(f"Fetching Reddit sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and (datetime.now() - self.sentiment_cache[symbol]['reddit']['timestamp']).total_seconds() < 3600:
            print(f"Using cached sentiment for {symbol}", flush=True)
            return self.sentiment_cache[symbol]['reddit']['sentiment']
        time.sleep(1)
        try:
            url = f"https://www.reddit.com/r/stocks/search/?q={symbol}&restrict_sr=1&sort=new"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            posts = [p.text for p in soup.find_all('h3', class_='_eYtD2XCVieq6emjKBH3m')[:5]]
            if not posts:
                return 0
            sentiment_score = 0
            count = 0
            for text in posts:
                sentiment = self.sentiment_analyzer.polarity_scores(text)['compound']
                sentiment_score += sentiment
                count += 1
            sentiment_score = sentiment_score / count if count > 0 else 0
            self.sentiment_cache[symbol]['reddit'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['reddit']['count'] += 1
            print(f"Completed Reddit sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Reddit fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_bloomberg_sentiment(self, symbol):
        print(f"Fetching Bloomberg sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'bloomberg' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['bloomberg']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Bloomberg sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['bloomberg']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Bloomberg sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['bloomberg'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['bloomberg']['count'] += 1
            print(f"Completed Bloomberg sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Bloomberg fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_ft_sentiment(self, symbol):
        print(f"Fetching Financial Times sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'ft' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['ft']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Financial Times sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['ft']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock FT sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['ft'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['ft']['count'] += 1
            print(f"Completed Financial Times sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Financial Times fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_yahoo_sentiment(self, symbol):
        print(f"Fetching Yahoo Finance sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'yahoo' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['yahoo']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Yahoo Finance sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['yahoo']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Yahoo sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['yahoo'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['yahoo']['count'] += 1
            print(f"Completed Yahoo Finance sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Yahoo Finance fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_stocktwits_sentiment(self, symbol):
        print(f"Fetching StockTwits sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'stocktwits' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['stocktwits']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached StockTwits sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['stocktwits']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock StockTwits sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['stocktwits'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['stocktwits']['count'] += 1
            print(f"Completed StockTwits sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"StockTwits fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_cnbc_sentiment(self, symbol):
        print(f"Fetching CNBC sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'cnbc' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['cnbc']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached CNBC sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['cnbc']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock CNBC sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['cnbc'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['cnbc']['count'] += 1
            print(f"Completed CNBC sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"CNBC fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_google_news_sentiment(self, symbol):
        print(f"Fetching Google News sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'google_news' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['google_news']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Google News sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['google_news']['sentiment']
        time.sleep(1)
        try:
            url = f"https://news.google.com/search?q={symbol}%20stock&hl=en-US"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            headlines = [h.text.strip() for h in soup.find_all('a', class_='JtKRv')[:5]]
            if not headlines:
                return 0
            sentiment_score = 0
            count = 0
            for text in headlines:
                sentiment = self.sentiment_analyzer.polarity_scores(text)['compound']
                sentiment_score += sentiment
                count += 1
            sentiment_score = sentiment_score / count if count > 0 else 0
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['google_news'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['google_news']['count'] += 1
            print(f"Completed Google News sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Google News fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_wallstreetjournal_sentiment(self, symbol):
        print(f"Fetching Wall Street Journal sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'wallstreetjournal' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['wallstreetjournal']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Wall Street Journal sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['wallstreetjournal']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Wall Street Journal sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['wallstreetjournal'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['wallstreetjournal']['count'] += 1
            print(f"Completed Wall Street Journal sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Wall Street Journal fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_seeking_alpha_sentiment(self, symbol):
        print(f"Fetching Seeking Alpha sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'seeking_alpha' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['seeking_alpha']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Seeking Alpha sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['seeking_alpha']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Seeking Alpha sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['seeking_alpha'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['seeking_alpha']['count'] += 1
            print(f"Completed Seeking Alpha sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Seeking Alpha fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_marketwatch_sentiment(self, symbol):
        print(f"Fetching MarketWatch sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'marketwatch' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['marketwatch']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached MarketWatch sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['marketwatch']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock MarketWatch sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['marketwatch'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['marketwatch']['count'] += 1
            print(f"Completed MarketWatch sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"MarketWatch fetch failed for {symbol}: {e}", flush=True)
            return None
    def fetch_benzinga_sentiment(self, symbol):
        print(f"Fetching Benzinga sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'benzinga' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['benzinga']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Benzinga sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['benzinga']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Benzinga sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['benzinga'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['benzinga']['count'] += 1
            print(f"Completed Benzinga sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Benzinga fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_finviz_sentiment(self, symbol):
        print(f"Fetching Finviz sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'finviz' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['finviz']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Finviz sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['finviz']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Finviz sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['finviz'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['finviz']['count'] += 1
            print(f"Completed Finviz sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Finviz fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_investing_sentiment(self, symbol):
        print(f"Fetching Investing.com sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'investing' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['investing']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached Investing.com sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['investing']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock Investing.com sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['investing'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['investing']['count'] += 1
            print(f"Completed Investing.com sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"Investing.com fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_motleyfool_sentiment(self, symbol):
        print(f"Fetching The Motley Fool sentiment for {symbol}...", flush=True)
        if symbol in self.sentiment_cache and 'motleyfool' in self.sentiment_cache[symbol]:
            elapsed = (datetime.now() - self.sentiment_cache[symbol]['motleyfool']['timestamp']).total_seconds() / 3600
            if elapsed < 1:
                print(f"Using cached The Motley Fool sentiment for {symbol}", flush=True)
                return self.sentiment_cache[symbol]['motleyfool']['sentiment']
        time.sleep(1)
        try:
            # Placeholder: Mock The Motley Fool sentiment until API implemented
            sentiment_score = random.uniform(-1, 1)
            self.sentiment_cache[symbol] = self.sentiment_cache.get(symbol, {})
            self.sentiment_cache[symbol]['motleyfool'] = {'sentiment': sentiment_score, 'timestamp': datetime.now()}
            self.api_usage['motleyfool']['count'] += 1
            print(f"Completed The Motley Fool sentiment for {symbol}: {sentiment_score}", flush=True)
            return sentiment_score
        except Exception as e:
            print(f"The Motley Fool fetch failed for {symbol}: {e}", flush=True)
            return None

    def fetch_combined_sentiment(self, symbol):
        print(f"Fetching combined sentiment for {symbol}...", flush=True)
        sources = {
            'twitter': {'func': self.fetch_twitter_sentiment, 'weight': 0.15, 'score': None},
            'newsdata': {'func': self.fetch_news_sentiment, 'weight': 0.15, 'score': None},
            'bloomberg': {'func': self.fetch_bloomberg_sentiment, 'weight': 0.1, 'score': None},
            'ft': {'func': self.fetch_ft_sentiment, 'weight': 0.1, 'score': None},
            'wallstreetjournal': {'func': self.fetch_wallstreetjournal_sentiment, 'weight': 0.1, 'score': None},
            'cnbc': {'func': self.fetch_cnbc_sentiment, 'weight': 0.1, 'score': None},
            'reddit': {'func': self.fetch_reddit_sentiment, 'weight': 0.05, 'score': None},
            'stocktwits': {'func': self.fetch_stocktwits_sentiment, 'weight': 0.05, 'score': None},
            'yahoo': {'func': self.fetch_yahoo_sentiment, 'weight': 0.05, 'score': None},
            'google_news': {'func': self.fetch_google_news_sentiment, 'weight': 0.05, 'score': None},
            'reuters': {'func': self.fetch_reuters_sentiment, 'weight': 0.05, 'score': None},
            'seeking_alpha': {'func': self.fetch_seeking_alpha_sentiment, 'weight': 0.05, 'score': None},
            'marketwatch': {'func': self.fetch_marketwatch_sentiment, 'weight': 0.05, 'score': None},
            'benzinga': {'func': self.fetch_benzinga_sentiment, 'weight': 0.05, 'score': None},
            'finviz': {'func': self.fetch_finviz_sentiment, 'weight': 0.05, 'score': None},
            'investing': {'func': self.fetch_investing_sentiment, 'weight': 0.05, 'score': None},
            'motleyfool': {'func': self.fetch_motleyfool_sentiment, 'weight': 0.05, 'score': None}
        }
        available_sources = []
        for source in self.source_priority:
            if self.api_usage[source]['count'] >= self.api_usage[source]['limit']:
                print(f"API limit reached for {source}, skipping...", flush=True)
                continue
            if symbol in self.sentiment_cache and source in self.sentiment_cache[symbol]:
                elapsed = (datetime.now() - self.sentiment_cache[symbol][source]['timestamp']).total_seconds() / 3600
                cached_score = self.sentiment_cache[symbol][source]['sentiment'] * (1 - 0.1 * elapsed)  # 10%/hour decay
                if elapsed < 1:
                    sources[source]['score'] = cached_score
                    available_sources.append(source)
                    continue
            try:
                score = sources[source]['func'](symbol)
                if score is not None:
                    sources[source]['score'] = score
                    self.api_usage[source]['count'] += 1
                    available_sources.append(source)
            except Exception as e:
                print(f"Failed to fetch {source} sentiment for {symbol}: {e}", flush=True)
                logging.error(f"Failed to fetch {source} sentiment for {symbol}: {e}")
                sources[source]['score'] = None

        available = {k: v for k, v in sources.items() if v['score'] is not None}
        if not available:
            print(f"No sentiment sources available for {symbol}, defaulting to 0", flush=True)
            return 0

        total_weight = sum(info['weight'] for info in available.values())
        combined_score = sum(info['score'] * (info['weight'] / total_weight) for info in available.values())
        print(f"Combined sentiment for {symbol}: {combined_score} (Sources: {', '.join(available.keys())})", flush=True)
        return combined_score

    def deep_search(self, symbol):
        print(f"Performing DeepSearch for {symbol}...", flush=True)
        try:
            news_score = self.fetch_news_sentiment(symbol)
            reuters_score = self.fetch_reuters_sentiment(symbol)
            reddit_score = self.fetch_reddit_sentiment(symbol)
            bloomberg_score = self.fetch_bloomberg_sentiment(symbol)
            ft_score = self.fetch_ft_sentiment(symbol)
            yahoo_score = self.fetch_yahoo_sentiment(symbol)
            cnbc_score = self.fetch_cnbc_sentiment(symbol)
            stocktwits_score = self.fetch_stocktwits_sentiment(symbol)
            google_news_score = self.fetch_google_news_sentiment(symbol)
            scores = [s for s in [news_score, reuters_score, reddit_score, bloomberg_score, ft_score, yahoo_score, cnbc_score, stocktwits_score, google_news_score] if s is not None]
            if not scores:
                return 0
            combined_score = sum(scores) / len(scores)
            print(f"DeepSearch result for {symbol}: {combined_score}", flush=True)
            return combined_score
        except Exception as e:
            print(f"DeepSearch failed for {symbol}: {e}", flush=True)
            return 0

    def calculate_technical_indicators(self, df):
        close_series = df['close']
        df['SMA10'] = close_series.rolling(10).mean()
        df['SMA20'] = close_series.rolling(20).mean()
        std_dev = close_series.rolling(20).std()
        df['BB_upper'] = df['SMA20'] + 2 * std_dev
        df['BB_lower'] = df['SMA20'] - 2 * std_dev
        df['SMA50'] = close_series.rolling(50).mean()
        df['SMA50_slope'] = df['SMA50'].pct_change(20, fill_method=None)
        df['SMA200'] = close_series.rolling(200).mean()
        df['lowest_14'] = close_series.rolling(14).min()
        df['highest_14'] = close_series.rolling(14).max()
        df['stoch_k'] = 100 * (close_series - df['lowest_14']) / (df['highest_14'] - df['lowest_14']).replace(0, np.nan).fillna(50)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(50)
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean().fillna(close_series.mean())
        df['RSI'] = ta.rsi(close_series, length=14)
        macd = ta.macd(close_series, fast=6, slow=10, signal=3)
        df['MACD'] = macd['MACD_6_10_3'].fillna(0) if macd is not None else 0
        df['MACD_signal'] = macd['MACDs_6_10_3'].fillna(0) if macd is not None else 0
        df['cup'] = df.apply(lambda row: self.detect_cup(df.loc[:row.name]), axis=1)
        df['cup_with_handle'] = df.apply(lambda row: self.detect_cup_with_handle(df.loc[:row.name]), axis=1)
        df['double_bottom'] = df.apply(lambda row: self.detect_double_bottom(df.loc[:row.name]), axis=1)
        df['support'] = df['low'].rolling(50).min()
        df['resistance'] = df['high'].rolling(50).max()
        denominator = (df['high'] - df['low']).replace(0, np.nan)
        put_call_ratio = (df['volume'] * (close_series - df['open']) / denominator).rolling(20).mean().fillna(1)
        df['put_call_ratio'] = pd.Series(put_call_ratio, index=df.index) if isinstance(put_call_ratio, np.ndarray) else put_call_ratio
        ad_line = np.cumsum(df['volume'] * (close_series - df['open']) / denominator).fillna(0)
        df['ad_line'] = pd.Series(ad_line, index=df.index) if isinstance(ad_line, np.ndarray) else ad_line
        df['breadth'] = df['ad_line'].pct_change(5, fill_method=None)
        df['ROC'] = ta.roc(close_series, length=10)
        df['vol_spread'] = self.vix_df['close'].iloc[-1] - df['ATR'].iloc[-1] if hasattr(self, 'vix_df') else 0
        df['fib_38_2'] = df['lowest_14'].iloc[-1] + 0.382 * (df['highest_14'].iloc[-1] - df['lowest_14'].iloc[-1])
        df['fib_50'] = df['lowest_14'].iloc[-1] + 0.5 * (df['highest_14'].iloc[-1] - df['lowest_14'].iloc[-1])
        df['fib_61_8'] = df['lowest_14'].iloc[-1] + 0.618 * (df['highest_14'].iloc[-1] - df['lowest_14'].iloc[-1])
        df['doji'] = (abs(close_series - df['open']) < (df['high'] - df['low']) * 0.1).astype(int)
        df['hammer'] = ((df['low'] < df['open']) & (close_series > df['open']) & ((df['high'] - close_series) < (close_series - df['low']) * 0.3)).astype(int)
        df['volume_sma20'] = df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] > df['volume_sma20'] * 1.5).astype(int)
        df['round_number'] = (close_series % 10 < 0.1).astype(int)
        return df.bfill().fillna(0)

    def detect_cup(self, df):
        if len(df) < 5:
            return 0
        prices = df['close'].values
        mid = len(prices) // 2
        left_peak = max(prices[:mid])
        right_peak = max(prices[mid:])
        trough = min(prices[mid-2:mid+2])
        return 1 if left_peak > trough and right_peak > trough and abs(left_peak - right_peak) < 0.1 * left_peak else 0

    def detect_cup_with_handle(self, df):
        if len(df) < 7:
            return 0
        prices = df['close'].values
        mid = len(prices) // 2
        left_peak = max(prices[:mid])
        right_peak = max(prices[mid:-2])
        trough = min(prices[mid-2:mid+2])
        handle = prices[-2:]
        if left_peak > trough and right_peak > trough and abs(left_peak - right_peak) < 0.1 * left_peak:
            if max(handle) < right_peak and max(handle) - min(handle) < 0.05 * right_peak:
                return 1
        return 0

    def detect_double_bottom(self, df):
        if len(df) < 5:
            return 0
        prices = df['low'].values
        bottoms = [i for i in range(1, len(prices)-1) if prices[i] < prices[i-1] and prices[i] < prices[i+1]]
        if len(bottoms) >= 2:  # Fixed: 'customs' changed to 'bottoms'
            b1, b2 = bottoms[-2], bottoms[-1]
            peak = max(prices[b1:b2])
            return 1 if abs(prices[b1] - prices[b2]) < 0.1 * prices[b1] and peak > prices[b1] * 1.05 else 0
        return 0

    def calculate_portfolio_risk(self):
        portfolio_value = self.cash
        for k, qty in list(self.positions.items()):
            if qty != 0:
                if k in self.dfs:
                    price = self.dfs[k].iloc[-1]['close']
                    portfolio_value += qty * price
                else:
                    print(f"Warning: {k} not found in dfs, skipping in portfolio risk calculation", flush=True)
                    logging.warning(f"Symbol {k} not found in dfs: dfs keys: {list(self.dfs.keys())[:5]}")
                    continue
        returns = [self.dfs[k]['close'].pct_change().dropna() for k, qty in self.positions.items() if qty != 0 and k in self.dfs]
        if returns:
            portfolio_returns = pd.concat(returns, axis=1).mean(axis=1)
            var = np.percentile(portfolio_returns, 5) * portfolio_value
            return var >= -0.05 * self.initial_cash
        return True

    def detect_market_regime(self, step):
        spy_trend = self.spy_df['SMA10'].iloc[-1] > self.spy_df['SMA20'].iloc[-1]
        vix_level = self.vix_df['close'].iloc[-1] > 5
        vix_trend = self.vix_df['close'].iloc[-1] > self.vix_df['SMA10'].iloc[-1]
        spy_drop = (self.spy_price_history[-1] - self.spy_price_history[-300]) / self.spy_price_history[-300] if len(self.spy_price_history) >= 300 else 0
        spy_bear = spy_drop < -0.01
        vix_spike = (self.vix_price_history[-1] - self.vix_price_history[-100]) / self.vix_price_history[-100] if len(self.vix_price_history) >= 100 else 0
        vix_bear = vix_spike > 0.05
        previous_market_type = self.market_type
        self.market_type = 'bull' if (spy_trend and not vix_level and not vix_trend and not spy_bear) else 'bear'
        if self.market_type != previous_market_type:
            print(f"Market regime shifted to {self.market_type} at step {step}", flush=True)
            self.handle_regime_change(step)
        return self.market_type

    def handle_regime_change(self, step):
        for k in list(self.trades.keys()):
            if self.trades[k]['remaining_qty'] > 0:
                qty = self.trades[k]['remaining_qty']
                action = 'SELL' if self.trades[k]['action'] == 'LONG' else 'COVER'
                price = self.dfs[k].iloc[-1]['close']
                self.simulate_trade(k, action, price, datetime.now(), qty, self.dfs[k].iloc[-1]['ATR'], step)

    def evaluate_position(self, key, df, idx, vix_df):
        if key not in self.positions or self.positions[key] == 0:
            return 'HOLD', 0
        current_price = df['close'].iloc[idx]
        position_qty = self.positions[key]
        is_long = position_qty > 0
        entry_price = self.positions.get(f"{key}_entry_price", current_price)
        unrealized_pnl = (current_price - entry_price) / entry_price * 100 if is_long else (entry_price - current_price) / entry_price * 100
        holding_duration = (datetime.now().timestamp() - self.positions.get(f"{key}_entry_time", datetime.now().timestamp())) / 3600
        atr = df['ATR'].iloc[idx] if 'ATR' in df.columns else (df['high'].iloc[idx] - df['low'].iloc[idx])
        volatility = atr / current_price * 100 if current_price != 0 else 0
        take_profit_threshold = 3.0 if self.market_type == 'bull' else 1.5
        stop_loss_threshold = -2.0 if self.market_type == 'bull' else -1.0
        if unrealized_pnl >= take_profit_threshold:
            return 'SELL' if is_long else 'COVER', unrealized_pnl
        elif unrealized_pnl <= stop_loss_threshold:
            return 'SELL' if is_long else 'COVER', unrealized_pnl
        if holding_duration > 24:
            return 'SELL' if is_long else 'COVER', 0
        if 'SMA50' in df.columns:
            price = df['close'].iloc[idx]
            sma_50 = df['SMA50'].iloc[idx]
            if self.market_type == 'bull' and price < sma_50 and is_long:
                return 'SELL', 0
            elif self.market_type == 'bear' and price > sma_50 and not is_long:
                return 'COVER', 0
        return 'HOLD', 0

    def choose_action(self, key, df, idx, vix_df, threshold=0.3):
        print(f"Choosing action for {key}...", flush=True)
        idx = min(idx, len(df) - 1)
        portfolio_value = self.cash + sum(qty * self.dfs[k]['close'].iloc[idx] for k, qty in self.positions.items() if k in self.dfs and idx < len(self.dfs[k]))
        cash_ratio = self.cash / self.initial_cash
        if cash_ratio < 0.1 and self.cash < 10000:
            print(f"Low cash ({self.cash:.2f}), forcing HOLD or SELL", flush=True)
            position_action, position_score = self.evaluate_position(key, df, idx, vix_df)
            if position_action in ['SELL', 'COVER']:
                return position_action, position_score
            return 'HOLD', 0
        position_action, position_score = self.evaluate_position(key, df, idx, vix_df)
        if position_action in ['SELL', 'COVER']:
            return position_action, position_score
        sentiment = self.fetch_combined_sentiment(key) if self.total_trades < self.daily_limit else 0
        buy_score = sell_score = short_score = cover_score = 0
        has_position = key in self.positions and self.positions[key] != 0
        is_long = has_position and self.positions[key] > 0
        is_short = has_position and self.positions[key] < 0
        factors = 24
        if self.market_type == 'bull':
            buy_score += 1 if df['RSI'].iloc[idx] < 30 else 0
            buy_score += 1 if df['MACD'].iloc[idx] > df['MACD_signal'].iloc[idx] else 0
            buy_score += 1 if df['close'].iloc[idx] < df['BB_lower'].iloc[idx] else 0
            buy_score += 1 if df['SMA10'].iloc[idx] > df['SMA20'].iloc[idx] else 0
            buy_score += 1 if df['close'].iloc[idx] < df['SMA200'].iloc[idx] * 1.05 else 0
            buy_score += 1 if df['stoch_k'].iloc[idx] < 20 and df['stoch_k'].iloc[idx] > df['stoch_d'].iloc[idx] else 0
            buy_score += 1 if df['cup'].iloc[idx] > 0 else 0
            buy_score += 1 if df['cup_with_handle'].iloc[idx] > 0 else 0
            buy_score += 1 if df['double_bottom'].iloc[idx] > 0 else 0
            buy_score += 1 if df['close'].iloc[idx] < df['support'].iloc[idx] * 1.05 else 0
            buy_score += 1 if df['close'].iloc[idx] in [df['fib_38_2'].iloc[idx], df['fib_50'].iloc[idx], df['fib_61_8'].iloc[idx]] else 0
            buy_score += 1 if df['doji'].iloc[idx] > 0 else 0
            buy_score += 1 if df['hammer'].iloc[idx] > 0 else 0
            buy_score += 1 if df['volume_spike'].iloc[idx] > 0 else 0
            buy_score += 1 if df['round_number'].iloc[idx] > 0 else 0
            buy_score += 1 if df['put_call_ratio'].iloc[idx] < 0.8 else 0
            buy_score += 1 if df['ad_line'].iloc[idx] > df['ad_line'].rolling(20).mean().iloc[idx] else 0
            buy_score += 1 if df['breadth'].iloc[idx] > 0 else 0
            buy_score += 1 if self.total_trades < self.daily_limit * 0.8 else 0
            buy_score += 1 if self.trade_count.get(key, 0) < self.max_trades_per_symbol * 0.8 else 0
            buy_score += 1 if len(self.large_trades) < 5 else 0
            buy_score += 1 if (datetime.now() - self.last_large_trade_check).total_seconds() > 1800 else 0
            buy_score += 1 if self.trade_steps.get(key, 0) < 50 else 0
            buy_score += 1 if sentiment > 0.5 else 0
            sell_score += 1 if df['RSI'].iloc[idx] > 70 else 0
            sell_score += 1 if df['MACD'].iloc[idx] < df['MACD_signal'].iloc[idx] else 0
            sell_score += 1 if df['close'].iloc[idx] > df['BB_upper'].iloc[idx] else 0
            sell_score += 1 if df['SMA10'].iloc[idx] < df['SMA20'].iloc[idx] else 0
            sell_score += 1 if df['close'].iloc[idx] > df['resistance'].iloc[idx] * 0.95 else 0
            sell_score += 1 if df['stoch_k'].iloc[idx] > 80 and df['stoch_k'].iloc[idx] < df['stoch_d'].iloc[idx] else 0
            sell_score += 1 if df['put_call_ratio'].iloc[idx] > 1.2 else 0
            sell_score += 1 if df['ad_line'].iloc[idx] < df['ad_line'].rolling(20).mean().iloc[idx] else 0
            sell_score += 1 if df['breadth'].iloc[idx] < 0 else 0
            sell_score += 1 if sentiment < -0.5 else 0
        else:
            buy_score += 1 if (key in self.inverted_leveraged_etfs or key in self.crypto or key in self.forex) and df['RSI'].iloc[idx] < 35 else 0
            buy_score += 1 if df['MACD'].iloc[idx] > df['MACD_signal'].iloc[idx] else 0
            buy_score += 1 if df['close'].iloc[idx] < df['BB_lower'].iloc[idx] else 0
            buy_score += 1 if self.vix_df['close'].pct_change(5).iloc[-1] > 0.05 else 0
            buy_score += 1 if sentiment > 0.1 else 0
            short_score += 1 if df['RSI'].iloc[idx] > 65 else 0
            short_score += 1 if df['MACD'].iloc[idx] < df['MACD_signal'].iloc[idx] else 0
            short_score += 1 if df['close'].iloc[idx] > df['BB_upper'].iloc[idx] else 0
            short_score += 1 if df['SMA10'].iloc[idx] < df['SMA20'].iloc[idx] else 0
            short_score += 1 if df['close'].iloc[idx] > df['resistance'].iloc[idx] * 0.95 else 0
            short_score += 1 if df['stoch_k'].iloc[idx] > 80 else 0
            short_score += 1 if df['put_call_ratio'].iloc[idx] > 1.2 else 0
            short_score += 1 if df['ad_line'].iloc[idx] < df['ad_line'].rolling(20).mean().iloc[idx] else 0
            short_score += 1 if df['breadth'].iloc[idx] < 0 else 0
            short_score += 1 if self.total_trades < self.daily_limit * 0.8 else 0
            short_score += 1 if self.trade_count.get(key, 0) < self.max_trades_per_symbol * 0.8 else 0
            short_score += 1 if len(self.large_trades) < 5 else 0
            short_score += 1 if (datetime.now() - self.last_large_trade_check).total_seconds() > 1800 else 0
            short_score += 1 if self.trade_steps.get(key, 0) < 50 else 0
            short_score += 1 if sentiment < -0.1 else 0
            cover_score += 1 if df['RSI'].iloc[idx] < 35 else 0
            cover_score += 1 if df['MACD'].iloc[idx] > df['MACD_signal'].iloc[idx] else 0
            cover_score += 1 if df['close'].iloc[idx] < df['BB_lower'].iloc[idx] else 0
            cover_score += 1 if df['SMA10'].iloc[idx] > df['SMA20'].iloc[idx] else 0
            cover_score += 1 if df['close'].iloc[idx] < df['support'].iloc[idx] * 1.05 else 0
            cover_score += 1 if df['stoch_k'].iloc[idx] < 20 else 0
            cover_score += 1 if df['cup'].iloc[idx] > 0 else 0
            cover_score += 1 if df['cup_with_handle'].iloc[idx] > 0 else 0
            cover_score += 1 if df['double_bottom'].iloc[idx] > 0 else 0
            cover_score += 1 if df['put_call_ratio'].iloc[idx] < 0.8 else 0
            cover_score += 1 if df['ad_line'].iloc[idx] > df['ad_line'].rolling(20).mean().iloc[idx] else 0
            cover_score += 1 if df['breadth'].iloc[idx] > 0 else 0
            cover_score += 1 if sentiment > 0.1 else 0
        buy_score /= factors
        sell_score /= factors
        short_score /= factors
        cover_score /= factors
        if buy_score >= threshold and not has_position:
            print(f"Action for {key}: BUY, Score: {buy_score}", flush=True)
            self.large_trades.append((key, 'BUY', df['close'].iloc[idx])) if df['volume'].iloc[idx] > df['volume_sma20'].iloc[idx] * 2 else None
            return 'BUY', buy_score
        elif sell_score >= threshold and is_long:
            print(f"Action for {key}: SELL, Score: {sell_score}", flush=True)
            self.large_trades.append((key, 'SELL', df['close'].iloc[idx])) if df['volume'].iloc[idx] > df['volume_sma20'].iloc[idx] * 2 else None
            return 'SELL', sell_score
        elif short_score >= threshold and not has_position:
            print(f"Action for {key}: SHORT, Score: {short_score}", flush=True)
            self.large_trades.append((key, 'SHORT', df['close'].iloc[idx])) if df['volume'].iloc[idx] > df['volume_sma20'].iloc[idx] * 2 else None
            return 'SHORT', short_score
        elif cover_score >= threshold and is_short:
            print(f"Action for {key}: COVER, Score: {cover_score}", flush=True)
            self.large_trades.append((key, 'COVER', df['close'].iloc[idx])) if df['volume'].iloc[idx] > df['volume_sma20'].iloc[idx] * 2 else None
            return 'COVER', cover_score
        print(f"Action for {key}: HOLD, Score: 0", flush=True)
        return 'HOLD', 0

    def simulate_trade(self, key, action, price, timestamp, qty, atr, step):
        stop_loss_factor = 1.5 if self.market_type == 'bear' else 1
        trailing_factor = 1.5 if self.market_type == 'bear' else 1
        if action in ['BUY', 'SHORT']:
            price *= 1.005  # Slippage
            price *= 1.001  # Fee
        else:
            price *= 0.995  # Slippage
            price *= 0.999  # Fee
        total_cost = price * qty
        if key in self.stocks or key in self.leveraged_etfs or key in self.inverted_leveraged_etfs:
            if action == 'BUY':
                if total_cost <= self.cash and total_cost <= self.cash_limit_entry:
                    self.cash -= total_cost
                    self.positions[key] = self.positions.get(key, 0) + qty
                    self.trades[key] = {
                        'action': 'LONG',
                        'entry_price': price,
                        'entry_time': timestamp.timestamp(),
                        'remaining_qty': qty,
                        'stop_loss': price * (1 - stop_loss_factor * atr / price),
                        'trailing_stop': None,
                        'high_price': price,
                        'low_price': price
                    }
                    self.trade_count[key] = self.trade_count.get(key, 0) + 1
                    self.total_trades += 1
                    self.trade_steps[key] = step
                    logging.info(f"{timestamp}: BUY {key}, Qty: {qty:.2f}, Price: {price:.2f}, Cash: {self.cash:.2f}")
                else:
                    print(f"Insufficient cash to BUY {key}: Required {total_cost:.2f}, Available {self.cash:.2f}", flush=True)
            elif action == 'SHORT':
                if total_cost <= self.cash and total_cost <= self.cash_limit_entry:
                    self.cash -= total_cost
                    self.positions[key] = self.positions.get(key, 0) - qty
                    self.trades[key] = {
                        'action': 'SHORT',
                        'entry_price': price,
                        'entry_time': timestamp.timestamp(),
                        'remaining_qty': qty,
                        'stop_loss': price * (1 + stop_loss_factor * atr / price),
                        'trailing_stop': None,
                        'high_price': price,
                        'low_price': price
                    }
                    self.trade_count[key] = self.trade_count.get(key, 0) + 1
                    self.total_trades += 1
                    self.trade_steps[key] = step
                    logging.info(f"{timestamp}: SHORT {key}, Qty: {qty:.2f}, Price: {price:.2f}, Cash: {self.cash:.2f}")
                else:
                    print(f"Insufficient cash to SHORT {key}: Required {total_cost:.2f}, Available {self.cash:.2f}", flush=True)
        elif action in ['SELL', 'COVER']:
            if key in self.positions and self.positions[key] != 0:
                current_qty = self.positions[key]
                remaining_qty = self.trades[key]['remaining_qty']
                sell_qty = min(qty, abs(current_qty), remaining_qty)
                if action == 'SELL' and current_qty > 0:
                    self.cash += price * sell_qty
                    self.positions[key] -= sell_qty
                    self.trades[key]['remaining_qty'] -= sell_qty
                    self.trade_count[key] = self.trade_count.get(key, 0) + 1
                    self.total_trades += 1
                    self.trade_steps[key] = step
                    logging.info(f"{timestamp}: SELL {key}, Qty: {sell_qty:.2f}, Price: {price:.2f}, Cash: {self.cash:.2f}")
                    if self.trades[key]['remaining_qty'] <= 0:
                        del self.trades[key]
                elif action == 'COVER' and current_qty < 0:
                    self.cash += (2 * self.trades[key]['entry_price'] - price) * sell_qty
                    self.positions[key] += sell_qty
                    self.trades[key]['remaining_qty'] -= sell_qty
                    self.trade_count[key] = self.trade_count.get(key, 0) + 1
                    self.total_trades += 1
                    self.trade_steps[key] = step
                    logging.info(f"{timestamp}: COVER {key}, Qty: {sell_qty:.2f}, Price: {price:.2f}, Cash: {self.cash:.2f}")
                    if self.trades[key]['remaining_qty'] <= 0:
                        del self.trades[key]
                if self.positions[key] == 0:
                    del self.positions[key]
        if self.total_trades >= self.daily_limit:
            print(f"Daily trade limit of {self.daily_limit} reached", flush=True)

    def rank_assets(self, dfs, vix_df, step):
        ranked_assets = []
        for key in self.limited_contracts[:self.daily_limit]:
            if key not in dfs or len(dfs[key]) <= 1:
                continue
            idx = min(step, len(dfs[key]) - 1)
            action, score = self.choose_action(key, dfs[key], idx, vix_df)
            ranked_assets.append((key, action, score))
        ranked_assets.sort(key=lambda x: x[2], reverse=True)
        top_assets = []
        for group in self.asset_groups.values():
            group_assets = [asset for asset in ranked_assets if asset[0] in group]
            top_assets.extend(group_assets[:2])
        for sector in self.sectors.values():
            sector_assets = [asset for asset in ranked_assets if asset[0] in sector]
            top_assets.extend(sector_assets[:2])
        unique_assets = list(dict.fromkeys(top_assets))
        return unique_assets[:10]

    def run_realtime(self, runtime_minutes=5):
        sys.settrace(trace_lines)
        self.positions.clear()
        self.trades.clear()
        print("Fetching initial data for limited contracts...", flush=True)
        self.dfs = {contract: self.calculate_technical_indicators(self.fetch_historical_data(self.contract_to_key_map[contract])) 
                    for contract in tqdm(self.limited_contracts[:self.daily_limit], desc="Fetching initial data")}
        self.spy_df = self.calculate_technical_indicators(self.fetch_historical_data(self.spy))
        self.vix_df = self.calculate_technical_indicators(self.fetch_historical_data(self.vix))
        self.spy_price_history = list(self.spy_df['close'].tail(5000))
        self.vix_price_history = list(self.vix_df['close'].tail(5000))
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=runtime_minutes)
        step = 0
        check_interval = 5
        trade_interval = 5
        print(f"Starting real-time trading simulation for {runtime_minutes} minutes...", flush=True)
        while datetime.now() < end_time and step < 600:
            step += 1
            self.spy_price_history.append(self.spy_df['close'].iloc[min(step, len(self.spy_df) - 1)])
            self.vix_price_history.append(self.vix_df['close'].iloc[min(step, len(self.vix_df) - 1)])
            self.detect_market_regime(step)
            for k in list(self.trades.keys()):
                if k not in self.dfs:
                    print(f"Warning: {k} not in dfs, skipping trade evaluation", flush=True)
                    logging.warning(f"Trade key {k} not found in dfs: dfs keys: {list(self.dfs.keys())[:5]}")
                    continue
                idx = min(step, len(self.dfs[k]) - 1)
                current_price = self.dfs[k]['close'].iloc[idx]
                current_time = datetime.now()
                trade = self.trades[k]
                if trade['action'] == 'LONG':
                    gain = (current_price - trade['entry_price']) / trade['entry_price']
                    trade['high_price'] = max(trade['high_price'], current_price)
                    peak_gain = (trade['high_price'] - trade['entry_price']) / trade['entry_price']
                    if trade['trailing_stop'] is None and gain >= 0.015:
                        trade['trailing_stop'] = current_price * 0.98
                    elif trade['trailing_stop'] is not None:
                        trade['trailing_stop'] = max(trade['trailing_stop'], current_price * (1 - trailing_factor * atr / trade['entry_price']))
                        if current_price <= trade['trailing_stop'] or current_price <= trade['stop_loss']:
                            self.simulate_trade(k, 'SELL', current_price, current_time, trade['remaining_qty'], atr, step)
                    elif current_price <= trade['stop_loss']:
                        self.simulate_trade(k, 'SELL', current_price, current_time, trade['remaining_qty'], atr, step)
                elif trade['action'] == 'SHORT':
                    gain = (trade['entry_price'] - current_price) / trade['entry_price']
                    trade['low_price'] = min(trade['low_price'], current_price)
                    peak_gain = (trade['entry_price'] - trade['low_price']) / trade['entry_price']
                    if trade['trailing_stop'] is None and gain >= 0.015:
                        trade['trailing_stop'] = current_price * 1.02
                    elif trade['trailing_stop'] is not None:
                        trade['trailing_stop'] = min(trade['trailing_stop'], current_price * (1 + trailing_factor * atr / trade['entry_price']))
                        if current_price >= trade['trailing_stop'] or current_price >= trade['stop_loss']:
                            self.simulate_trade(k, 'COVER', current_price, current_time, trade['remaining_qty'], atr, step)
                    elif current_price >= trade['stop_loss']:
                        self.simulate_trade(k, 'COVER', current_price, current_time, trade['remaining_qty'], atr, step)
                self.last_check_step[k] = step
            self.hold_counter = self.hold_counter + 1 if not ranked_assets else 0
            if step % check_interval == 0:
                ranked_assets = self.rank_assets(self.dfs, self.vix_df, step)
                print(f"Step {step}: Top ranked assets: {', '.join([f'{asset[0]} ({asset[1]})' for asset in ranked_assets[:5]])}", flush=True)
                portfolio_value = self.cash + sum(self.positions.get(k, 0) * self.dfs[k]['close'].iloc[min(step, len(self.dfs[k]) - 1)] for k in self.positions if k in self.dfs)
                print(f"Portfolio Value: ${portfolio_value:,.2f}, Cash: ${self.cash:,.2f}, Total Trades: {self.total_trades}", flush=True)
                if self.hold_counter > 50:
                    print(f"Too many consecutive holds ({self.hold_counter}), adjusting strategy...", flush=True)
                    self.hold_counter = 0
            if step % trade_interval == 0 and ranked_assets:
                for key, action, score in ranked_assets[:5]:
                    if action not in ['HOLD', 'SELL', 'COVER']:
                        idx = min(step, len(self.dfs[key]) - 1)
                        price = self.dfs[key]['close'].iloc[idx]
                        atr = self.dfs[key]['ATR'].iloc[idx]
                        qty = (self.cash * 0.02) / price if price != 0 else 0
                        if qty > 0:
                            self.simulate_trade(key, action, price, datetime.now(), qty, atr, step)
            if step % 300 == 0:
                self.spy_df = self.calculate_technical_indicators(self.fetch_historical_data(self.spy))
                self.vix_df = self.calculate_technical_indicators(self.fetch_historical_data(self.vix))
                self.spy_price_history = list(self.spy_df['close'].tail(5000))
                self.vix_price_history = list(self.vix_df['close'].tail(5000))
            time.sleep(1)
        sys.settrace(None)
        self.generate_execution_report()

    def generate_execution_report(self):
        print("Generating Execution Report...", flush=True)
        executed_lines_list = sorted(list(executed_lines))
        print(f"Executed lines: {executed_lines_list}", flush=True)
        total_lines = len(open(__file__).readlines())
        missed_lines = sorted(set(range(1, total_lines + 1)) - executed_lines)
        print(f"Missed lines: {missed_lines}", flush=True)
        portfolio_value = self.cash + sum(self.positions.get(k, 0) * self.dfs[k]['close'].iloc[-1] for k in self.positions if k in self.dfs)
        returns = (portfolio_value - self.initial_cash) / self.initial_cash * 100
        print(f"Final Equity: ${portfolio_value:,.2f}, Return: {returns:.2f}%", flush=True)
        print(f"Executed: {len(executed_lines)}, Missed: {len(missed_lines)}", flush=True)
        logging.info(f"Execution Report - Final Equity: ${portfolio_value:,.2f}, Return: {returns:.2f}%, Executed: {len(executed_lines)}, Missed: {len(missed_lines)}")

def pretrain_rl(bot):
    try:
        model_path = os.path.expanduser("~/Documents/ppo_trading_bot_500k.zip")
        if os.path.exists(model_path):
            bot.ppo = PPO.load(model_path, env=bot.env)
            print(f"PPO model loaded from {model_path}", flush=True)
        else:
            bot.ppo = PPO('MlpPolicy', bot.env, verbose=1)
            bot.ppo.learn(total_timesteps=500000)
            bot.ppo.save(model_path)
            print(f"PPO model trained and saved to {model_path}", flush=True)
    except Exception as e:
        print(f"Failed to pretrain RL model: {e}", flush=True)
        logging.error(f"Failed to pretrain RL model: {e}")

def main():
    bot = TradingBot(simulate=True)
    bot.connect()
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):  # Suppress Chromedriver noise
                bot.run_realtime(runtime_minutes=5)
    except Exception as e:
        print(f"Error in main execution: {e}", flush=True)
        logging.error(f"Error in main execution: {e}")
    finally:
        bot.disconnect()

if __name__ == "__main__":
    main()
