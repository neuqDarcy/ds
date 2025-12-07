#!/usr/bin/env python3
"""
真实历史数据缠论回测系统
从交易所获取真实K线数据进行回测
"""

import os
import sys
import json
import time
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ====================== 真实数据获取器 ======================
class RealDataFetcher:
    """真实交易所数据获取器"""
    
    def __init__(self):
        # 初始化交易所连接（只读，不需要API密钥）
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
            'rateLimit': 1200  # Binance API限制
        })
        
        # 数据缓存 - 放在外层目录
        self.data_cache = {}
        # 获取当前文件所在目录的父目录（即代码目录的外层目录）
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "historical_data")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_historical_ohlcv(self, 
                           symbol: str, 
                           timeframe: str, 
                           start_date: str, 
                           end_date: str,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        获取真实历史K线数据
        
        Args:
            symbol: 交易对，如"BTC/USDT"
            timeframe: 时间框架，如"1m", "5m", "15m", "30m", "1h", "4h", "1d"
            start_date: 开始日期，格式："2024-01-01 00:00:00"
            end_date: 结束日期，格式："2024-06-01 00:00:00"
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame with OHLCV数据
        """
        # 生成缓存文件名
        cache_key = f"{symbol.replace('/', '_')}_{timeframe}_{start_date[:10]}_{end_date[:10]}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
        
        # 检查缓存
        if use_cache and os.path.exists(cache_file):
            print(f"📂 从缓存加载数据: {cache_file}")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        print(f"📡 从Binance获取数据: {symbol} {timeframe}")
        print(f"   时间范围: {start_date} 到 {end_date}")
        
        # 转换日期为时间戳
        start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        
        # Binance需要毫秒时间戳
        since = int(start_dt.timestamp() * 1000)
        all_data = []
        
        # 分批次获取（避免API限制）
        current_since = since
        max_limit = 1000  # Binance单次最多1000根K线
        batch_count = 0
        
        while current_since < int(end_dt.timestamp() * 1000):
            try:
                # 计算本次获取的数量
                time.sleep(0.1)  # 遵守API频率限制
                
                # 获取数据
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_since,
                    limit=max_limit
                )
                
                if not ohlcv:
                    break
                
                # 转换为DataFrame
                df_batch = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # 时间戳转换
                df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')
                
                all_data.append(df_batch)
                batch_count += 1
                
                # 更新起始时间（使用最后一条数据的时间戳+1个间隔）
                last_timestamp = df_batch['timestamp'].iloc[-1]
                
                # 计算下一个起始时间
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    current_since = int((last_timestamp + pd.Timedelta(minutes=minutes)).timestamp() * 1000)
                elif timeframe.endswith('h'):
                    hours = int(timeframe[:-1])
                    current_since = int((last_timestamp + pd.Timedelta(hours=hours)).timestamp() * 1000)
                elif timeframe.endswith('d'):
                    days = int(timeframe[:-1])
                    current_since = int((last_timestamp + pd.Timedelta(days=days)).timestamp() * 1000)
                else:
                    # 默认1小时
                    current_since = int((last_timestamp + pd.Timedelta(hours=1)).timestamp() * 1000)
                
                # 显示进度
                if batch_count % 5 == 0:
                    print(f"  已获取 {len(df_batch) * batch_count} 根K线，最新时间: {last_timestamp}")
                
                # 检查是否已经超过结束时间
                if last_timestamp >= end_dt:
                    break
                    
            except Exception as e:
                print(f"❌ 获取数据失败: {e}")
                print(f"   当前since: {current_since}")
                break
        
        if not all_data:
            print("❌ 未获取到任何数据")
            return pd.DataFrame()
        
        # 合并所有数据
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        
        # 过滤时间范围
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        
        # 添加技术指标
        df = self.add_technical_indicators(df)
        
        print(f"✅ 数据获取完成，共 {len(df)} 根K线")
        print(f"   时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")
        
        # 保存到缓存
        if use_cache:
            df.to_csv(cache_file, index=False)
            print(f"💾 数据已缓存: {cache_file}")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加常用技术指标"""
        if df.empty:
            return df
        
        # 简单移动平均线
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 成交量指标
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 价格变化
        df['price_change'] = df['close'].pct_change() * 100
        df['high_low_spread'] = (df['high'] - df['low']) / df['low'] * 100
        
        return df
    
    def get_available_timeframes(self) -> List[str]:
        """获取可用的时间框架"""
        return ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    
    def get_available_symbols(self) -> List[str]:
        """获取可用的交易对"""
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except:
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']

# ====================== 可配置的回测系统 ======================
class RealBacktestConfig:
    """真实回测配置（可交互）"""
    
    def __init__(self):
        # 默认配置
        self.symbol = "BTC/USDT"
        self.timeframe = "30m"
        self.start_date = "2024-01-01 00:00:00"
        self.end_date = "2024-06-01 00:00:00"
        
        self.initial_capital = 10000.0
        self.trade_amount = 0.001
        self.leverage = 1
        
        self.stop_loss_pct = 1.5
        self.take_profit_pct = 3.0
        self.fee_rate = 0.001
        self.slippage = 0.0005
        
        self.fractal_period = 5
        self.confidence_threshold = 0.6
        
        self.enable_entry_scoping = True
        self.verbose = False
        self.plot_results = True
        self.save_results = True
    
    def interactive_setup(self):
        """交互式配置向导"""
        print("\n" + "="*60)
        print("🤖 真实数据回测系统 - 配置向导")
        print("="*60)
        
        data_fetcher = RealDataFetcher()
        
        # 1. 选择交易对
        print("\n1. 选择交易对:")
        available_symbols = data_fetcher.get_available_symbols()[:10]  # 显示前10个
        for i, sym in enumerate(available_symbols, 1):
            print(f"   {i}. {sym}")
        print(f"   {len(available_symbols)+1}. 手动输入")
        
        symbol_choice = input(f"\n   请选择交易对 (1-{len(available_symbols)+1}，默认1): ").strip()
        if symbol_choice and symbol_choice.isdigit():
            choice = int(symbol_choice)
            if 1 <= choice <= len(available_symbols):
                self.symbol = available_symbols[choice-1]
            elif choice == len(available_symbols) + 1:
                self.symbol = input("   请输入交易对 (如 BTC/USDT): ").strip()
        else:
            self.symbol = available_symbols[0]
        
        # 2. 选择时间框架
        print("\n2. 选择时间框架:")
        timeframes = data_fetcher.get_available_timeframes()
        for i, tf in enumerate(timeframes[:10], 1):
            print(f"   {i}. {tf}")
        print(f"   {len(timeframes[:10])+1}. 手动输入")
        
        tf_choice = input(f"\n   请选择时间框架 (1-{len(timeframes[:10])+1}，默认5): ").strip()
        if tf_choice and tf_choice.isdigit():
            choice = int(tf_choice)
            if 1 <= choice <= len(timeframes[:10]):
                self.timeframe = timeframes[choice-1]
            elif choice == len(timeframes[:10]) + 1:
                self.timeframe = input("   请输入时间框架: ").strip()
        else:
            self.timeframe = timeframes[4]  # 默认30m
        
        # 3. 设置时间范围
        print("\n3. 设置回测时间范围:")
        print("   格式: YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DD")
        
        start_input = input(f"   开始日期 (默认: {self.start_date}): ").strip()
        if start_input:
            if len(start_input) == 10:  # YYYY-MM-DD
                self.start_date = f"{start_input} 00:00:00"
            else:
                self.start_date = start_input
        
        end_input = input(f"   结束日期 (默认: {self.end_date}): ").strip()
        if end_input:
            if len(end_input) == 10:
                self.end_date = f"{end_input} 23:59:59"
            else:
                self.end_date = end_input
        
        # 4. 设置资金参数
        print("\n4. 设置资金参数:")
        capital_input = input(f"   初始资金 (默认: {self.initial_capital}): ").strip()
        if capital_input:
            self.initial_capital = float(capital_input)
        
        amount_input = input(f"   每次交易数量 (默认: {self.trade_amount}): ").strip()
        if amount_input:
            self.trade_amount = float(amount_input)
        
        # 5. 设置风险参数
        print("\n5. 设置风险参数:")
        sl_input = input(f"   止损百分比 (默认: {self.stop_loss_pct}%): ").strip()
        if sl_input:
            self.stop_loss_pct = float(sl_input)
        
        tp_input = input(f"   止盈百分比 (默认: {self.take_profit_pct}%): ").strip()
        if tp_input:
            self.take_profit_pct = float(tp_input)
        
        # 6. 设置缠论参数
        print("\n6. 设置缠论参数:")
        fractal_input = input(f"   分型周期 (默认: {self.fractal_period}): ").strip()
        if fractal_input:
            self.fractal_period = int(fractal_input)
        
        confidence_input = input(f"   信号信心阈值 (默认: {self.confidence_threshold}): ").strip()
        if confidence_input:
            self.confidence_threshold = float(confidence_input)
        
        # 7. 设置其他选项
        print("\n7. 设置其他选项:")
        entry_scoping = input("   启用5分钟精确入场? (y/n, 默认: y): ").strip().lower()
        self.enable_entry_scoping = entry_scoping != 'n'
        
        verbose_input = input("   显示详细日志? (y/n, 默认: n): ").strip().lower()
        self.verbose = verbose_input == 'y'
        
        plot_input = input("   生成图表? (y/n, 默认: y): ").strip().lower()
        self.plot_results = plot_input != 'n'
        
        print("\n✅ 配置完成!")
        return self
    
    def display(self):
        """显示配置"""
        print("\n" + "="*60)
        print("📋 回测配置详情")
        print("="*60)
        
        config_dict = {
            "交易对": self.symbol,
            "时间框架": self.timeframe,
            "开始时间": self.start_date,
            "结束时间": self.end_date,
            "初始资金": f"${self.initial_capital:,.2f}",
            "交易数量": f"{self.trade_amount:.4f}",
            "止损": f"{self.stop_loss_pct}%",
            "止盈": f"{self.take_profit_pct}%",
            "手续费": f"{self.fee_rate*100:.2f}%",
            "滑点": f"{self.slippage*100:.2f}%",
            "分型周期": self.fractal_period,
            "信心阈值": self.confidence_threshold,
            "精确入场": "启用" if self.enable_entry_scoping else "禁用",
            "详细日志": "是" if self.verbose else "否"
        }
        
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        
        print("="*60)

# ====================== 真实缠论回测引擎 ======================
class RealChanBacktestEngine:
    """真实数据的缠论回测引擎"""
    
    def __init__(self, config: RealBacktestConfig):
        self.config = config
        self.data_fetcher = RealDataFetcher()
        
        # 回测状态
        self.capital = config.initial_capital
        self.btc_amount = 0.0
        self.trades = []
        self.equity_history = []
        self.signals_history = []
        
        # 持仓状态
        self.position = None
        self.position_entry_price = 0.0
        
        print(f"🚀 初始化真实数据回测引擎")
        print(f"   交易对: {config.symbol}")
        print(f"   时间框架: {config.timeframe}")
        print(f"   时间范围: {config.start_date} 到 {config.end_date}")
    
    def load_real_data(self) -> pd.DataFrame:
        """加载真实历史数据"""
        print("\n📊 加载历史数据...")
        
        df = self.data_fetcher.get_historical_ohlcv(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            use_cache=True
        )
        
        if df.empty:
            print("❌ 数据加载失败")
            return df
        
        # 显示数据统计
        print(f"\n📈 数据统计:")
        print(f"   数据条数: {len(df)}")
        print(f"   时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")
        print(f"   价格范围: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"   平均成交量: {df['volume'].mean():.2f}")
        
        return df
    
    def calculate_position_size(self, price: float) -> float:
        """计算仓位大小"""
        max_position_value = self.capital * 0.1  # 最多使用10%资金
        position_amount = min(self.config.trade_amount, max_position_value / price)
        return position_amount
    
    def execute_buy(self, price: float, timestamp: datetime, signal: Dict[str, Any]):
        """执行买入"""
        # 计算考虑滑点的实际价格
        actual_price = price * (1 + self.config.slippage)
        
        # 计算交易量
        position_amount = self.calculate_position_size(actual_price)
        
        # 计算交易成本
        trade_value = position_amount * actual_price
        fee = trade_value * self.config.fee_rate
        
        # 检查资金是否足够
        if trade_value + fee > self.capital:
            if self.config.verbose:
                print(f"❌ 资金不足，无法买入。需要: {trade_value+fee:.2f}，可用: {self.capital:.2f}")
            return False
        
        # 更新持仓
        self.btc_amount += position_amount
        self.capital -= (trade_value + fee)
        
        # 记录持仓
        self.position = {
            'side': 'long',
            'entry_price': actual_price,
            'entry_time': timestamp,
            'amount': position_amount,
            'stop_loss': actual_price * (1 - self.config.stop_loss_pct / 100),
            'take_profit': actual_price * (1 + self.config.take_profit_pct / 100)
        }
        
        # 记录交易
        trade_record = {
            'timestamp': timestamp,
            'type': 'BUY',
            'price': actual_price,
            'amount': position_amount,
            'value': trade_value,
            'fee': fee,
            'signal': signal,
            'capital_before': self.capital + trade_value + fee,
            'capital_after': self.capital,
            'btc_amount': self.btc_amount
        }
        self.trades.append(trade_record)
        
        if self.config.verbose:
            print(f"💰 [{timestamp.strftime('%Y-%m-%d %H:%M')}] 买入: {position_amount:.4f} BTC @ ${actual_price:.2f}")
            print(f"    交易金额: ${trade_value:.2f}，手续费: ${fee:.2f}")
            print(f"    现金余额: ${self.capital:.2f}，BTC持仓: {self.btc_amount:.4f}")
        
        return True
    
    def execute_sell(self, price: float, timestamp: datetime, signal: Dict[str, Any]):
        """执行卖出"""
        if self.btc_amount <= 0:
            if self.config.verbose:
                print("❌ 无BTC持仓，无法卖出")
            return False
        
        # 计算考虑滑点的实际价格
        actual_price = price * (1 - self.config.slippage)
        
        # 卖出全部持仓
        position_amount = self.btc_amount
        trade_value = position_amount * actual_price
        fee = trade_value * self.config.fee_rate
        
        # 更新持仓和资金
        self.btc_amount = 0.0
        self.capital += (trade_value - fee)
        
        # 计算盈亏
        if self.position:
            entry_price = self.position['entry_price']
            pnl = (actual_price - entry_price) * position_amount
            pnl_pct = (actual_price - entry_price) / entry_price * 100
            
            # 记录交易
            trade_record = {
                'timestamp': timestamp,
                'type': 'SELL',
                'price': actual_price,
                'amount': position_amount,
                'value': trade_value,
                'fee': fee,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'signal': signal,
                'capital_before': self.capital - trade_value + fee,
                'capital_after': self.capital,
                'btc_amount': self.btc_amount
            }
            self.trades.append(trade_record)
            
            if self.config.verbose:
                print(f"💰 [{timestamp.strftime('%Y-%m-%d %H:%M')}] 卖出: {position_amount:.4f} BTC @ ${actual_price:.2f}")
                print(f"    交易金额: ${trade_value:.2f}，手续费: ${fee:.2f}")
                print(f"    盈亏: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                print(f"    现金余额: ${self.capital:.2f}，BTC持仓: {self.btc_amount:.4f}")
            
            # 清空持仓
            self.position = None
            
            return True
        
        return False
    
    def check_stop_loss_take_profit(self, current_price: float, timestamp: datetime) -> bool:
        """检查止损止盈"""
        if not self.position:
            return False
        
        entry_price = self.position['entry_price']
        stop_loss = self.position['stop_loss']
        take_profit = self.position['take_profit']
        
        if current_price <= stop_loss:
            signal = {
                'signal': 'SELL',
                'reason': f'触发止损: {current_price:.2f} <= {stop_loss:.2f}',
                'confidence': 1.0
            }
            self.execute_sell(current_price, timestamp, signal)
            return True
        
        if current_price >= take_profit:
            signal = {
                'signal': 'SELL',
                'reason': f'触发止盈: {current_price:.2f} >= {take_profit:.2f}',
                'confidence': 1.0
            }
            self.execute_sell(current_price, timestamp, signal)
            return True
        
        return False
    
    def calculate_equity(self, current_price: float) -> float:
        """计算当前权益"""
        btc_value = self.btc_amount * current_price
        return self.capital + btc_value
    
    def find_fractals(self, df: pd.DataFrame, current_idx: int) -> Dict[str, List]:
        """缠论分型识别"""
        if current_idx < 4 or current_idx >= len(df) - 4:
            return {'top': [], 'bottom': []}
        
        window_start = max(0, current_idx - 20)
        window_end = min(len(df), current_idx + 1)
        window_df = df.iloc[window_start:window_end]
        
        if len(window_df) < 5:
            return {'top': [], 'bottom': []}
        
        highs = window_df['high'].values
        lows = window_df['low'].values
        
        top_fractals, bottom_fractals = [], []
        
        for i in range(2, len(window_df) - 2):
            global_i = window_start + i
            
            # 顶分型
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                top_fractals.append({
                    'index': global_i,
                    'price': highs[i],
                    'timestamp': window_df.iloc[i]['timestamp']
                })
            
            # 底分型
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                bottom_fractals.append({
                    'index': global_i,
                    'price': lows[i],
                    'timestamp': window_df.iloc[i]['timestamp']
                })
        
        return {'top': top_fractals, 'bottom': bottom_fractals}
    
    def identify_bi_segments(self, fractals: Dict[str, List]) -> List[Dict]:
        """识别笔"""
        all_fractals = fractals['top'] + fractals['bottom']
        all_fractals.sort(key=lambda x: x['index'])
        
        bi_segments = []
        for i in range(len(all_fractals) - 1):
            current = all_fractals[i]
            next_f = all_fractals[i + 1]
            
            bi_segments.append({
                'start': current,
                'end': next_f,
                'direction': 'up' if current['price'] < next_f['price'] else 'down',
                'price_change': next_f['price'] - current['price'],
                'pct_change': (next_f['price'] - current['price']) / current['price'] * 100
            })
        
        return bi_segments
    
    def analyze_trend(self, bi_segments: List[Dict]) -> Dict[str, Any]:
        """分析趋势"""
        if len(bi_segments) < 2:
            return {'trend': 'unknown', 'strength': 0}
        
        recent_bis = bi_segments[-3:] if len(bi_segments) >= 3 else bi_segments
        up_count = sum(1 for bi in recent_bis if bi['direction'] == 'up')
        down_count = len(recent_bis) - up_count
        
        strength = 0
        if recent_bis:
            price_changes = [bi['pct_change'] for bi in recent_bis]
            strength = abs(np.mean(price_changes))
        
        if up_count > down_count and strength > 0.5:
            trend = 'up'
        elif down_count > up_count and strength > 0.5:
            trend = 'down'
        else:
            trend = 'consolidation'
        
        return {'trend': trend, 'strength': strength}
    
    def find_buy_signal(self, df: pd.DataFrame, current_idx: int, trend_analysis: Dict) -> Dict[str, Any]:
        """寻找买点信号"""
        current_price = df.iloc[current_idx]['close']
        current_time = df.iloc[current_idx]['timestamp']
        
        # 获取近期数据
        lookback = min(10, current_idx)
        recent_data = df.iloc[current_idx - lookback:current_idx + 1]
        
        recent_low = recent_data['low'].min()
        recent_high = recent_data['high'].max()
        
        signals = []
        
        # 基于价格的信号
        if current_price <= recent_low * 1.01:  # 接近近期低点
            signals.append({
                'type': 'near_low',
                'confidence': 0.6,
                'description': '价格接近近期低点'
            })
        
        # 基于RSI的信号
        if 'rsi' in df.columns:
            current_rsi = df.iloc[current_idx]['rsi']
            if current_rsi < 30:  # 超卖
                signals.append({
                    'type': 'rsi_oversold',
                    'confidence': 0.7,
                    'description': f'RSI超卖: {current_rsi:.1f}'
                })
        
        # 基于MACD的信号
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df.iloc[current_idx]['macd']
            macd_signal = df.iloc[current_idx]['macd_signal']
            if macd > macd_signal and df.iloc[current_idx-1]['macd'] <= df.iloc[current_idx-1]['macd_signal']:
                signals.append({
                    'type': 'macd_cross',
                    'confidence': 0.65,
                    'description': 'MACD金叉'
                })
        
        # 基于布林带的信号
        if 'bb_lower' in df.columns:
            bb_lower = df.iloc[current_idx]['bb_lower']
            if current_price <= bb_lower * 1.01:
                signals.append({
                    'type': 'bb_lower',
                    'confidence': 0.6,
                    'description': '价格接近布林带下轨'
                })
        
        if signals:
            best_signal = max(signals, key=lambda x: x['confidence'])
            return {
                'signal': 'BUY',
                'reason': best_signal['description'],
                'confidence': best_signal['confidence'],
                'price': current_price,
                'timestamp': current_time,
                'type': best_signal['type']
            }
        
        return {
            'signal': 'HOLD',
            'reason': '无明确买点信号',
            'confidence': 0,
            'price': current_price,
            'timestamp': current_time
        }
    
    def find_sell_signal(self, df: pd.DataFrame, current_idx: int, trend_analysis: Dict) -> Dict[str, Any]:
        """寻找卖点信号"""
        current_price = df.iloc[current_idx]['close']
        current_time = df.iloc[current_idx]['timestamp']
        
        # 获取近期数据
        lookback = min(10, current_idx)
        recent_data = df.iloc[current_idx - lookback:current_idx + 1]
        
        recent_high = recent_data['high'].max()
        
        signals = []
        
        # 基于价格的信号
        if current_price >= recent_high * 0.99:  # 接近近期高点
            signals.append({
                'type': 'near_high',
                'confidence': 0.6,
                'description': '价格接近近期高点'
            })
        
        # 基于RSI的信号
        if 'rsi' in df.columns:
            current_rsi = df.iloc[current_idx]['rsi']
            if current_rsi > 70:  # 超买
                signals.append({
                    'type': 'rsi_overbought',
                    'confidence': 0.7,
                    'description': f'RSI超买: {current_rsi:.1f}'
                })
        
        # 基于MACD的信号
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df.iloc[current_idx]['macd']
            macd_signal = df.iloc[current_idx]['macd_signal']
            if macd < macd_signal and df.iloc[current_idx-1]['macd'] >= df.iloc[current_idx-1]['macd_signal']:
                signals.append({
                    'type': 'macd_death_cross',
                    'confidence': 0.65,
                    'description': 'MACD死叉'
                })
        
        # 基于布林带的信号
        if 'bb_upper' in df.columns:
            bb_upper = df.iloc[current_idx]['bb_upper']
            if current_price >= bb_upper * 0.99:
                signals.append({
                    'type': 'bb_upper',
                    'confidence': 0.6,
                    'description': '价格接近布林带上轨'
                })
        
        if signals:
            best_signal = max(signals, key=lambda x: x['confidence'])
            return {
                'signal': 'SELL',
                'reason': best_signal['description'],
                'confidence': best_signal['confidence'],
                'price': current_price,
                'timestamp': current_time,
                'type': best_signal['type']
            }
        
        return {
            'signal': 'HOLD',
            'reason': '无明确卖点信号',
            'confidence': 0,
            'price': current_price,
            'timestamp': current_time
        }
    
    def run_backtest(self):
        """运行回测"""
        print("\n" + "="*60)
        print("🚀 开始真实数据缠论回测")
        print("="*60)
        
        # 加载数据
        df = self.load_real_data()
        if df.empty:
            return None
        
        print(f"\n🔍 执行回测分析...")
        
        # 需要足够的数据才能开始分析
        start_idx = max(20, self.config.fractal_period + 5)
        
        for idx in range(start_idx, len(df)):
            current_row = df.iloc[idx]
            current_price = current_row['close']
            current_time = current_row['timestamp']
            
            # 显示进度
            if idx % 100 == 0 and self.config.verbose:
                progress = (idx - start_idx) / (len(df) - start_idx) * 100
                print(f"  进度: {progress:.1f}% ({idx}/{len(df)})，价格: ${current_price:.2f}")
            
            # 检查止损止盈
            if self.position and self.check_stop_loss_take_profit(current_price, current_time):
                continue
            
            # 缠论分析
            fractals = self.find_fractals(df, idx)
            bi_segments = self.identify_bi_segments(fractals)
            trend_analysis = self.analyze_trend(bi_segments)
            
            # 根据持仓状态决定分析类型
            if not self.position:  # 无持仓，寻找买点
                signal = self.find_buy_signal(df, idx, trend_analysis)
                
                if signal['signal'] == 'BUY' and signal['confidence'] >= self.config.confidence_threshold:
                    self.signals_history.append(signal)
                    self.execute_buy(current_price, current_time, signal)
                    
            else:  # 有持仓，寻找卖点
                signal = self.find_sell_signal(df, idx, trend_analysis)
                
                if signal['signal'] == 'SELL' and signal['confidence'] >= self.config.confidence_threshold:
                    self.signals_history.append(signal)
                    self.execute_sell(current_price, current_time, signal)
            
            # 记录权益历史
            equity = self.calculate_equity(current_price)
            self.equity_history.append({
                'timestamp': current_time,
                'equity': equity,
                'price': current_price,
                'capital': self.capital,
                'btc_amount': self.btc_amount,
                'btc_value': self.btc_amount * current_price
            })
        
        # 回测结束，平掉所有持仓
        if self.btc_amount > 0:
            last_price = df.iloc[-1]['close']
            signal = {
                'signal': 'SELL',
                'reason': '回测结束，强制平仓',
                'confidence': 1.0
            }
            self.execute_sell(last_price, df.iloc[-1]['timestamp'], signal)
        
        print("\n✅ 回测完成!")
        
        return self.generate_report(df)
    
    def generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成回测报告"""
        if not self.equity_history:
            return {}
        
        initial_equity = self.config.initial_capital
        final_equity = self.equity_history[-1]['equity']
        total_return_pct = (final_equity - initial_equity) / initial_equity * 100
        
        # 计算年化收益率
        start_time = df.iloc[0]['timestamp']
        end_time = df.iloc[-1]['timestamp']
        days_diff = (end_time - start_time).days
        years = days_diff / 365.0
        annual_return_pct = ((final_equity / initial_equity) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # 计算最大回撤
        equity_series = pd.Series([h['equity'] for h in self.equity_history])
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown_pct = drawdown.min()
        
        # 计算胜率
        winning_trades = 0
        total_sell_trades = len([t for t in self.trades if t['type'] == 'SELL' and 'pnl' in t])
        
        for trade in self.trades:
            if trade['type'] == 'SELL' and 'pnl' in trade:
                if trade['pnl'] > 0:
                    winning_trades += 1
        
        win_rate = (winning_trades / total_sell_trades * 100) if total_sell_trades > 0 else 0
        
        # 计算总盈亏
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        total_fees = sum(t.get('fee', 0) for t in self.trades)
        
        # 计算平均持仓时间
        hold_times = []
        buy_time = None
        
        for trade in self.trades:
            if trade['type'] == 'BUY':
                buy_time = trade['timestamp']
            elif trade['type'] == 'SELL' and buy_time:
                hold_time = (trade['timestamp'] - buy_time).total_seconds() / 3600  # 小时
                hold_times.append(hold_time)
                buy_time = None
        
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        report = {
            'summary': {
                'initial_capital': initial_equity,
                'final_equity': final_equity,
                'total_return_pct': total_return_pct,
                'annual_return_pct': annual_return_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'total_trades': len(self.trades),
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_fees': total_fees,
                'avg_hold_time_hours': avg_hold_time,
                'data_points': len(df),
                'time_period_days': days_diff
            },
            'config': {
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct,
                'fractal_period': self.config.fractal_period,
                'confidence_threshold': self.config.confidence_threshold
            },
            'trades': self.trades,
            'signals': self.signals_history,
            'equity_history': self.equity_history
        }
        
        return report
    
    def plot_results(self, report: Dict[str, Any]):
        """绘制回测结果图表"""
        if not report or not self.equity_history:
            print("❌ 没有回测数据可绘制")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. 权益曲线 (左上)
        times = [h['timestamp'] for h in self.equity_history]
        equities = [h['equity'] for h in self.equity_history]
        prices = [h['price'] for h in self.equity_history]
        
        ax1 = axes[0, 0]
        ax1.plot(times, equities, 'b-', linewidth=2, label='总权益')
        ax1.axhline(y=self.config.initial_capital, color='r', linestyle='--', alpha=0.5, label='初始资金')
        ax1.set_title(f'权益曲线 - 总收益: {report["summary"]["total_return_pct"]:.2f}%', fontsize=12)
        ax1.set_ylabel('权益 (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 价格曲线和交易点 (右上)
        ax2 = axes[0, 1]
        ax2.plot(times, prices, 'g-', alpha=0.7, linewidth=1, label='价格')
        ax2.set_title(f'{self.config.symbol} 价格走势', fontsize=12)
        ax2.set_ylabel('价格 (USDT)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 标记买入卖出点
        buy_times = [t['timestamp'] for t in self.trades if t['type'] == 'BUY']
        buy_prices = [t['price'] for t in self.trades if t['type'] == 'BUY']
        sell_times = [t['timestamp'] for t in self.trades if t['type'] == 'SELL']
        sell_prices = [t['price'] for t in self.trades if t['type'] == 'SELL']
        
        if buy_times:
            ax2.scatter(buy_times, buy_prices, color='green', s=80, marker='^', label='买入', zorder=5)
        if sell_times:
            ax2.scatter(sell_times, sell_prices, color='red', s=80, marker='v', label='卖出', zorder=5)
        
        # 3. 回撤曲线 (中左)
        ax3 = axes[1, 0]
        equity_series = pd.Series(equities)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        ax3.fill_between(times, drawdown, 0, color='red', alpha=0.3)
        ax3.plot(times, drawdown, 'r-', linewidth=1)
        ax3.set_title(f'资金回撤 - 最大回撤: {report["summary"]["max_drawdown_pct"]:.2f}%', fontsize=12)
        ax3.set_ylabel('回撤 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 持仓市值 (中右)
        ax4 = axes[1, 1]
        btc_values = [h['btc_value'] for h in self.equity_history]
        capitals = [h['capital'] for h in self.equity_history]
        
        ax4.stackplot(times, btc_values, capitals, 
                     labels=['BTC持仓', '现金'], 
                     colors=['orange', 'lightblue'], alpha=0.7)
        ax4.set_title('资产构成', fontsize=12)
        ax4.set_ylabel('价值 (USDT)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 技术指标: RSI (左下)
        ax5 = axes[2, 0]
        if hasattr(self, 'df_with_indicators'):
            df = self.df_with_indicators
            # 对齐时间
            common_times = [t for t in times if t in df['timestamp'].values]
            if len(common_times) > 0:
                df_filtered = df[df['timestamp'].isin(common_times)].sort_values('timestamp')
                if 'rsi' in df_filtered.columns:
                    ax5.plot(df_filtered['timestamp'], df_filtered['rsi'], 'purple', linewidth=1)
                    ax5.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                    ax5.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                    ax5.fill_between(df_filtered['timestamp'], 70, df_filtered['rsi'], 
                                    where=(df_filtered['rsi'] >= 70), color='red', alpha=0.3)
                    ax5.fill_between(df_filtered['timestamp'], 30, df_filtered['rsi'], 
                                    where=(df_filtered['rsi'] <= 30), color='green', alpha=0.3)
                    ax5.set_title('RSI指标', fontsize=12)
                    ax5.set_ylabel('RSI')
                    ax5.set_ylim(0, 100)
                    ax5.grid(True, alpha=0.3)
        
        # 6. 交易统计 (右下)
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        stats_text = f"""
        回测统计:
        初始资金: ${report['summary']['initial_capital']:,.2f}
        最终权益: ${report['summary']['final_equity']:,.2f}
        总收益率: {report['summary']['total_return_pct']:+.2f}%
        年化收益率: {report['summary']['annual_return_pct']:+.2f}%
        
        最大回撤: {report['summary']['max_drawdown_pct']:.2f}%
        总交易次数: {report['summary']['total_trades']}
        胜率: {report['summary']['win_rate']:.1f}%
        总盈亏: ${report['summary']['total_pnl']:+.2f}
        总手续费: ${report['summary']['total_fees']:.2f}
        
        数据点数: {report['summary']['data_points']}
        回测天数: {report['summary']['time_period_days']}
        平均持仓: {report['summary']['avg_hold_time_hours']:.1f}小时
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=10, 
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 设置总标题
        fig.suptitle(f'{self.config.symbol} 缠论策略回测报告\n'
                    f'{self.config.start_date} 至 {self.config.end_date} | {self.config.timeframe}框架',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        if self.config.save_results:
            # 使用外层目录保存图表
            outer_reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
            os.makedirs(outer_reports_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol_clean = self.config.symbol.replace('/', '_')
            filename = f"{outer_reports_dir}/{symbol_clean}_backtest_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 图表已保存: {filename}")
        
        plt.show()
    
    def print_report(self, report: Dict[str, Any]):
        """打印回测报告"""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("📊 缠论策略回测报告")
        print("="*60)
        
        print(f"📈 策略表现:")
        print(f"   初始资金: ${summary['initial_capital']:,.2f}")
        print(f"   最终权益: ${summary['final_equity']:,.2f}")
        print(f"   总收益率: {summary['total_return_pct']:+.2f}%")
        print(f"   年化收益率: {summary['annual_return_pct']:+.2f}%")
        
        print(f"\n⚠️  风险指标:")
        print(f"   最大回撤: {summary['max_drawdown_pct']:.2f}%")
        print(f"   总手续费: ${summary['total_fees']:.2f}")
        print(f"   平均持仓: {summary['avg_hold_time_hours']:.1f}小时")
        
        print(f"\n📊 交易统计:")
        print(f"   总交易次数: {summary['total_trades']}")
        print(f"   盈利交易: {summary['winning_trades']}")
        print(f"   胜率: {summary['win_rate']:.1f}%")
        print(f"   总盈亏: ${summary['total_pnl']:+.2f}")
        
        print(f"\n🔧 配置参数:")
        print(f"   交易对: {self.config.symbol}")
        print(f"   时间框架: {self.config.timeframe}")
        print(f"   止损: {self.config.stop_loss_pct}%")
        print(f"   止盈: {self.config.take_profit_pct}%")
        print(f"   分型周期: {self.config.fractal_period}")
        print(f"   信心阈值: {self.config.confidence_threshold}")
        
        # 显示最近几笔交易
        if self.trades:
            print(f"\n📋 最近交易记录:")
            recent_trades = self.trades[-10:] if len(self.trades) > 10 else self.trades
            
            for trade in recent_trades:
                time_str = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
                if trade['type'] == 'BUY':
                    print(f"   {time_str} | 买入 {trade['amount']:.4f} BTC @ ${trade['price']:.2f}")
                else:
                    pnl_str = f"(${trade.get('pnl', 0):+.2f}, {trade.get('pnl_pct', 0):+.2f}%)" if 'pnl' in trade else ""
                    reason = trade.get('signal', {}).get('reason', '')
                    print(f"   {time_str} | 卖出 {trade['amount']:.4f} BTC @ ${trade['price']:.2f} {pnl_str}")
                    if reason:
                        print(f"      理由: {reason}")
        
        print("="*60)
        
        # 保存报告到文件
        if self.config.save_results:
            # 使用外层目录保存报告
            outer_reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
            os.makedirs(outer_reports_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol_clean = self.config.symbol.replace('/', '_')
            report_file = f"{outer_reports_dir}/{symbol_clean}_report_{timestamp}.json"
            
            # 简化报告数据
            simple_report = {
                'summary': summary,
                'config': report['config'],
                'trade_count': len(self.trades),
                'signal_count': len(self.signals_history)
            }
            
            with open(report_file, 'w') as f:
                json.dump(simple_report, f, indent=2, default=str)
            
            print(f"\n💾 报告已保存: {report_file}")

# ====================== 多策略比较器 ======================
class StrategyComparator:
    """多策略比较器"""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
    
    def add_strategy(self, name: str, config: RealBacktestConfig):
        """添加策略"""
        self.strategies[name] = config
    
    def compare_strategies(self):
        """比较多个策略"""
        print(f"\n📊 开始比较 {len(self.strategies)} 个策略...")
        
        for name, config in self.strategies.items():
            print(f"\n🔧 测试策略: {name}")
            config.display()
            
            engine = RealChanBacktestEngine(config)
            report = engine.run_backtest()
            
            if report:
                self.results[name] = report['summary']
                print(f"   收益率: {report['summary']['total_return_pct']:.2f}%")
                print(f"   最大回撤: {report['summary']['max_drawdown_pct']:.2f}%")
                print(f"   胜率: {report['summary']['win_rate']:.1f}%")
        
        return self.analyze_comparison()
    
    def analyze_comparison(self):
        """分析比较结果"""
        if not self.results:
            return None
        
        print("\n" + "="*60)
        print("🏆 策略比较结果")
        print("="*60)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # 找出最佳策略
        best_return = results_df.loc[results_df['total_return_pct'].idxmax()]
        best_risk_adj = results_df.loc[(results_df['total_return_pct'] / abs(results_df['max_drawdown_pct'])).idxmax()]
        best_win_rate = results_df.loc[results_df['win_rate'].idxmax()]
        
        print(f"\n📈 最佳收益率:")
        print(f"   策略: {best_return.name}")
        print(f"   收益率: {best_return['total_return_pct']:.2f}%")
        print(f"   最大回撤: {best_return['max_drawdown_pct']:.2f}%")
        
        print(f"\n⚖️  最佳风险调整收益:")
        print(f"   策略: {best_risk_adj.name}")
        risk_adj_return = best_risk_adj['total_return_pct'] / abs(best_risk_adj['max_drawdown_pct'])
        print(f"   风险调整收益: {risk_adj_return:.2f}")
        
        print(f"\n🎯 最高胜率:")
        print(f"   策略: {best_win_rate.name}")
        print(f"   胜率: {best_win_rate['win_rate']:.1f}%")
        
        # 显示所有策略比较
        print(f"\n📋 所有策略表现:")
        print(results_df[['total_return_pct', 'max_drawdown_pct', 'win_rate', 'total_trades']])
        
        # 保存比较结果
        # 使用外层目录保存比较结果
        outer_reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
        os.makedirs(outer_reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{outer_reports_dir}/strategy_comparison_{timestamp}.csv"
        results_df.to_csv(results_file)
        print(f"\n💾 比较结果已保存: {results_file}")
        
        return results_df

# ====================== 主函数 ======================
def main():
    """主函数"""
    print("="*60)
    print("📊 真实数据缠论回测系统")
    print("="*60)
    
    print("\n请选择模式:")
    print("1. 单策略回测")
    print("2. 多策略比较")
    print("3. 参数优化")
    print("4. 查看可用交易对")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == '1':
        # 单策略回测
        config = RealBacktestConfig()
        config.interactive_setup()
        config.display()
        
        # 确认开始回测
        confirm = input("\n🚀 是否开始回测? (y/n): ").lower() == 'y'
        if not confirm:
            print("回测已取消")
            return
        
        # 运行回测
        engine = RealChanBacktestEngine(config)
        report = engine.run_backtest()
        
        if report:
            engine.print_report(report)
            if config.plot_results:
                engine.plot_results(report)
    
    elif choice == '2':
        # 多策略比较
        print("\n📊 多策略比较模式")
        print("将测试不同参数配置的策略表现")
        
        comparator = StrategyComparator()
        
        # 定义不同策略配置
        strategies = {
            '保守策略': RealBacktestConfig(),
            '激进策略': RealBacktestConfig(),
            '日内交易': RealBacktestConfig()
        }
        
        # 修改策略配置
        strategies['保守策略'].stop_loss_pct = 1.0
        strategies['保守策略'].take_profit_pct = 2.0
        strategies['保守策略'].confidence_threshold = 0.7
        
        strategies['激进策略'].stop_loss_pct = 2.0
        strategies['激进策略'].take_profit_pct = 5.0
        strategies['激进策略'].confidence_threshold = 0.5
        
        strategies['日内交易'].timeframe = '15m'
        strategies['日内交易'].stop_loss_pct = 0.8
        strategies['日内交易'].take_profit_pct = 1.5
        strategies['日内交易'].trade_amount = 0.002
        
        # 添加到比较器
        for name, config in strategies.items():
            comparator.add_strategy(name, config)
        
        # 运行比较
        comparator.compare_strategies()
    
    elif choice == '3':
        # 参数优化
        print("\n🔧 参数优化模式")
        print("将测试不同参数组合，找出最优配置")
        
        # 基础配置
        base_config = RealBacktestConfig()
        
        # 定义参数网格
        param_grid = {
            'stop_loss_pct': [1.0, 1.5, 2.0],
            'take_profit_pct': [2.0, 3.0, 4.0],
            'confidence_threshold': [0.5, 0.6, 0.7]
        }
        
        results = []
        
        # 网格搜索
        for sl in param_grid['stop_loss_pct']:
            for tp in param_grid['take_profit_pct']:
                for ct in param_grid['confidence_threshold']:
                    print(f"\n测试参数: SL={sl}%, TP={tp}%, CT={ct}")
                    
                    # 创建配置副本
                    config = RealBacktestConfig()
                    config.stop_loss_pct = sl
                    config.take_profit_pct = tp
                    config.confidence_threshold = ct
                    config.verbose = False
                    
                    # 运行回测
                    engine = RealChanBacktestEngine(config)
                    report = engine.run_backtest()
                    
                    if report:
                        results.append({
                            'stop_loss_pct': sl,
                            'take_profit_pct': tp,
                            'confidence_threshold': ct,
                            'total_return_pct': report['summary']['total_return_pct'],
                            'max_drawdown_pct': report['summary']['max_drawdown_pct'],
                            'win_rate': report['summary']['win_rate']
                        })
        
        # 分析结果
        if results:
            results_df = pd.DataFrame(results)
            
            # 按收益率排序
            best_return = results_df.loc[results_df['total_return_pct'].idxmax()]
            
            # 按风险调整收益排序
            results_df['risk_adjusted'] = results_df['total_return_pct'] / abs(results_df['max_drawdown_pct'])
            best_risk_adj = results_df.loc[results_df['risk_adjusted'].idxmax()]
            
            print("\n" + "="*60)
            print("🏆 最优参数组合")
            print("="*60)
            
            print(f"\n💰 最高收益率:")
            print(f"   止损: {best_return['stop_loss_pct']}%")
            print(f"   止盈: {best_return['take_profit_pct']}%")
            print(f"   信心阈值: {best_return['confidence_threshold']}")
            print(f"   收益率: {best_return['total_return_pct']:.2f}%")
            print(f"   最大回撤: {best_return['max_drawdown_pct']:.2f}%")
            
            print(f"\n⚖️  最佳风险调整收益:")
            print(f"   止损: {best_risk_adj['stop_loss_pct']}%")
            print(f"   止盈: {best_risk_adj['take_profit_pct']}%")
            print(f"   信心阈值: {best_risk_adj['confidence_threshold']}")
            print(f"   风险调整收益: {best_risk_adj['risk_adjusted']:.2f}")
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            results_file = os.path.join(reports_dir, f'parameter_optimization_{timestamp}.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\n💾 优化结果已保存: {results_file}")
    
    elif choice == '4':
        # 查看可用交易对
        print("\n🔍 正在获取可用交易对...")
        data_fetcher = RealDataFetcher()
        symbols = data_fetcher.get_available_symbols()
        
        print(f"\n📋 前20个可用交易对:")
        for i, symbol in enumerate(symbols[:20], 1):
            print(f"  {i:2d}. {symbol}")
        
        print(f"\n📋 可用的时间框架:")
        timeframes = data_fetcher.get_available_timeframes()
        for tf in timeframes:
            print(f"  {tf}", end=' ')
        print()
    
    else:
        print("❌ 无效选项")

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('historical_data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()