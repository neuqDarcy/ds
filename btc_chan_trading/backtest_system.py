#!/usr/bin/env python3
"""
BTC缠论交易策略回测系统
作者: AI交易助手
版本: 1.0.0
描述: 基于历史数据的缠论策略回测分析
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 第三方库
import ccxt
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ====================== 回测配置 ======================
class BacktestConfig:
    """回测配置"""
    
    # 回测参数
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "30m"  # 主时间框架
    ENTRY_TIMEFRAME = "5m"  # 入场时间框架
    START_DATE = "2025-01-01 00:00:00"  # 回测开始时间
    END_DATE = "2025-12-01 00:00:00"    # 回测结束时间
    
    # 交易参数
    INITIAL_CAPITAL = 1000.0  # 初始资金 (USDT)
    TRADE_AMOUNT = 0.001       # 每次交易BTC数量
    LEVERAGE = 1               # 杠杆（回测建议用1）
    
    # 手续费和滑点
    FEE_RATE = 0           # 手续费率 (0.1%)
    # FEE_RATE = 0.001           # 手续费率 (0.1%)
    SLIPPAGE = 0          # 滑点 (0.05%)
    # SLIPPAGE = 0.0005          # 滑点 (0.05%)

    
    # 风险控制
    STOP_LOSS_PCT = 1.5        # 止损百分比
    TAKE_PROFIT_PCT = 3.0      # 止盈百分比
    
    # 缠论参数
    FRACTAL_PERIOD = 5         # 分型周期
    CENTRAL_PIVOT_LOOKBACK = 20  # 中枢观察周期
    
    # 回测模式
    ENABLE_ENTRY_SCOPING = True  # 是否启用5分钟精确入场
    ENABLE_AI_ANALYSIS = False   # 回测中是否使用AI分析（通常关闭以加速）
    VERBOSE = True               # 是否输出详细日志

# ====================== 数据获取器 ======================
class HistoricalDataFetcher:
    """历史数据获取器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000
        })
        
    def fetch_historical_data(self, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Args:
            timeframe: 时间框架
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            DataFrame with OHLCV数据
        """
        print(f"📊 正在获取历史数据: {self.config.SYMBOL} {timeframe}")
        print(f"    时间范围: {start_date} 到 {end_date}")
        
        # 转换日期格式
        start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        
        all_data = []
        current_start = start_dt
        
        # 分批次获取数据（避免API限制）
        while current_start < end_dt:
            try:
                # 计算批次结束时间
                batch_end = min(current_start + timedelta(days=30), end_dt)
                
                # 获取数据
                since = int(current_start.timestamp() * 1000)
                limit = 1000  # 每次最多获取1000根K线
                
                ohlcv = self.exchange.fetch_ohlcv(
                    self.config.SYMBOL,
                    timeframe,
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                # 转换为DataFrame
                df_batch = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')
                
                all_data.append(df_batch)
                
                # 更新开始时间（使用最后一条数据的时间）
                last_timestamp = df_batch['timestamp'].iloc[-1]
                current_start = last_timestamp + pd.Timedelta(seconds=1)
                
                print(f"    ✓ 已获取: {df_batch['timestamp'].iloc[0]} 到 {df_batch['timestamp'].iloc[-1]}")
                
            except Exception as e:
                print(f"    ✗ 获取数据失败: {e}")
                break
        
        if not all_data:
            print("❌ 未获取到任何数据")
            return pd.DataFrame()
        
        # 合并所有数据
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        
        # 过滤时间范围
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        
        print(f"✅ 数据获取完成，共 {len(df)} 根K线")
        return df
    
    def save_data_to_csv(self, df: pd.DataFrame, filename: str):
        """保存数据到CSV文件"""
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"💾 数据已保存到: {filename}")
    
    def load_data_from_csv(self, filename: str) -> pd.DataFrame:
        """从CSV文件加载数据"""
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"📂 从文件加载数据: {filename}，共 {len(df)} 根K线")
            return df
        else:
            print(f"❌ 文件不存在: {filename}")
            return pd.DataFrame()

# ====================== 缠论回测分析器 ======================
class ChanBacktestAnalyzer:
    """缠论回测分析器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.fractals_history = []
        self.bi_segments_history = []
        self.central_pivots_history = []
        self.signals = []
        
    def find_fractals(self, df: pd.DataFrame, current_idx: int) -> Dict[str, List]:
        """在指定位置识别分型"""
        if current_idx < 4 or current_idx >= len(df) - 4:
            return {'top': [], 'bottom': []}
        
        # 获取当前窗口的数据
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
                    'timestamp': window_df.iloc[i]['timestamp'],
                    'type': 'top'
                })
            
            # 底分型
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                bottom_fractals.append({
                    'index': global_i,
                    'price': lows[i],
                    'timestamp': window_df.iloc[i]['timestamp'],
                    'type': 'bottom'
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
            
            if current['type'] != next_f['type']:
                bi_segments.append({
                    'start': current,
                    'end': next_f,
                    'direction': 'up' if current['type'] == 'bottom' else 'down',
                    'price_change': next_f['price'] - current['price'],
                    'pct_change': (next_f['price'] - current['price']) / current['price'] * 100
                })
        
        return bi_segments
    
    def find_central_pivot(self, bi_segments: List[Dict]) -> List[Dict]:
        """识别中枢"""
        if len(bi_segments) < 3:
            return []
        
        central_pivots = []
        
        for i in range(len(bi_segments) - 2):
            seg1 = bi_segments[i]
            seg2 = bi_segments[i + 1]
            seg3 = bi_segments[i + 2]
            
            highs = [
                max(seg1['start']['price'], seg1['end']['price']),
                max(seg2['start']['price'], seg2['end']['price']),
                max(seg3['start']['price'], seg3['end']['price'])
            ]
            lows = [
                min(seg1['start']['price'], seg1['end']['price']),
                min(seg2['start']['price'], seg2['end']['price']),
                min(seg3['start']['price'], seg3['end']['price'])
            ]
            
            overlap_high = min(highs)
            overlap_low = max(lows)
            
            if overlap_low < overlap_high:
                central_pivots.append({
                    'start_idx': seg1['start']['index'],
                    'end_idx': seg3['end']['index'],
                    'high': overlap_high,
                    'low': overlap_low,
                    'center': (overlap_high + overlap_low) / 2,
                    'width': overlap_high - overlap_low
                })
        
        return central_pivots
    
    def analyze_trend(self, bi_segments: List[Dict], central_pivots: List[Dict]) -> Dict[str, Any]:
        """分析趋势"""
        if len(bi_segments) < 2:
            return {'trend': 'unknown', 'strength': 0, 'has_central_pivot': False}
        
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
        
        return {
            'trend': trend,
            'strength': strength,
            'has_central_pivot': len(central_pivots) > 0,
            'central_pivot_count': len(central_pivots)
        }
    
    def find_buy_signal(self, df: pd.DataFrame, current_idx: int, trend_analysis: Dict) -> Dict[str, Any]:
        """寻找买点信号"""
        current_price = df.iloc[current_idx]['close']
        
        # 获取近期数据
        lookback = min(10, current_idx)
        recent_data = df.iloc[current_idx - lookback:current_idx + 1]
        
        recent_low = recent_data['low'].min()
        recent_high = recent_data['high'].max()
        
        signals = []
        
        # 信号1：接近近期低点
        if current_price <= recent_low * 1.01:
            signals.append({
                'type': 'type1_bottom_divergence',
                'price': current_price,
                'description': '第一类买点：底背驰',
                'confidence': 0.7 if trend_analysis['trend'] == 'down' else 0.4
            })
        
        # 信号2：突破近期高点
        if current_price >= recent_high * 0.99:
            signals.append({
                'type': 'type2_breakout',
                'price': current_price,
                'description': '第二类买点：突破回踩',
                'confidence': 0.8 if trend_analysis['trend'] == 'up' else 0.5
            })
        
        # 信号3：中枢盘整
        if trend_analysis['has_central_pivot'] and trend_analysis['trend'] == 'consolidation':
            signals.append({
                'type': 'type3_consolidation',
                'price': current_price,
                'description': '第三类买点：中枢盘整',
                'confidence': 0.6
            })
        
        if signals:
            best_signal = max(signals, key=lambda x: x['confidence'])
            return {
                'signal': 'BUY',
                'reason': best_signal['description'],
                'confidence': best_signal['confidence'],
                'type': best_signal['type'],
                'price': current_price,
                'timestamp': df.iloc[current_idx]['timestamp']
            }
        
        return {
            'signal': 'HOLD',
            'reason': '无买点信号',
            'confidence': 0,
            'price': current_price,
            'timestamp': df.iloc[current_idx]['timestamp']
        }
    
    def find_sell_signal(self, df: pd.DataFrame, current_idx: int, trend_analysis: Dict) -> Dict[str, Any]:
        """寻找卖点信号"""
        current_price = df.iloc[current_idx]['close']
        
        # 获取近期数据
        lookback = min(10, current_idx)
        recent_data = df.iloc[current_idx - lookback:current_idx + 1]
        
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        
        signals = []
        
        # 信号1：接近近期高点
        if current_price >= recent_high * 0.99:
            signals.append({
                'type': 'type1_top_divergence',
                'price': current_price,
                'description': '第一类卖点：顶背驰',
                'confidence': 0.7 if trend_analysis['trend'] == 'up' else 0.4
            })
        
        # 信号2：跌破近期低点
        if current_price <= recent_low * 1.01:
            signals.append({
                'type': 'type2_breakdown',
                'price': current_price,
                'description': '第二类卖点：跌破反弹',
                'confidence': 0.8 if trend_analysis['trend'] == 'down' else 0.5
            })
        
        if signals:
            best_signal = max(signals, key=lambda x: x['confidence'])
            return {
                'signal': 'SELL',
                'reason': best_signal['description'],
                'confidence': best_signal['confidence'],
                'type': best_signal['type'],
                'price': current_price,
                'timestamp': df.iloc[current_idx]['timestamp']
            }
        
        return {
            'signal': 'HOLD',
            'reason': '无卖点信号',
            'confidence': 0,
            'price': current_price,
            'timestamp': df.iloc[current_idx]['timestamp']
        }

# ====================== 回测引擎 ======================
class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_fetcher = HistoricalDataFetcher(config)
        self.analyzer = ChanBacktestAnalyzer(config)
        
        # 回测状态
        self.capital = config.INITIAL_CAPITAL  # 现金
        self.btc_amount = 0.0  # BTC持仓
        self.trades = []  # 交易记录
        self.equity_history = []  # 权益历史
        self.signals_history = []  # 信号历史
        
        # 持仓状态
        self.position = None  # 当前持仓
        self.position_entry_price = 0.0
        self.position_entry_time = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载历史数据"""
        # 主时间框架数据
        main_filename = f"data/{self.config.SYMBOL.replace('/', '_')}_{self.config.TIMEFRAME}.csv"
        
        if os.path.exists(main_filename):
            df_main = self.data_fetcher.load_data_from_csv(main_filename)
        else:
            df_main = self.data_fetcher.fetch_historical_data(
                self.config.TIMEFRAME,
                self.config.START_DATE,
                self.config.END_DATE
            )
            if not df_main.empty:
                os.makedirs('data', exist_ok=True)
                self.data_fetcher.save_data_to_csv(df_main, main_filename)
        
        # 入场时间框架数据（如果启用）
        df_entry = pd.DataFrame()
        if self.config.ENABLE_ENTRY_SCOPING and self.config.TIMEFRAME != self.config.ENTRY_TIMEFRAME:
            entry_filename = f"data/{self.config.SYMBOL.replace('/', '_')}_{self.config.ENTRY_TIMEFRAME}.csv"
            
            if os.path.exists(entry_filename):
                df_entry = self.data_fetcher.load_data_from_csv(entry_filename)
            else:
                df_entry = self.data_fetcher.fetch_historical_data(
                    self.config.ENTRY_TIMEFRAME,
                    self.config.START_DATE,
                    self.config.END_DATE
                )
                if not df_entry.empty:
                    self.data_fetcher.save_data_to_csv(df_entry, entry_filename)
        
        return df_main, df_entry
    
    def calculate_position_size(self, price: float) -> float:
        """计算仓位大小"""
        max_position_value = self.capital * 0.1  # 最多使用10%资金
        position_amount = min(self.config.TRADE_AMOUNT, max_position_value / price)
        return position_amount
    
    def execute_buy(self, price: float, timestamp: datetime, signal: Dict[str, Any]):
        """执行买入"""
        # 计算考虑滑点的实际价格
        actual_price = price * (1 + self.config.SLIPPAGE)
        
        # 计算交易量
        position_amount = self.calculate_position_size(actual_price)
        
        # 计算交易成本
        trade_value = position_amount * actual_price
        fee = trade_value * self.config.FEE_RATE
        
        # 检查资金是否足够
        if trade_value + fee > self.capital:
            if self.config.VERBOSE:
                print(f"❌ 资金不足，无法买入。需要: {trade_value + fee:.2f}，可用: {self.capital:.2f}")
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
            'stop_loss': actual_price * (1 - self.config.STOP_LOSS_PCT / 100),
            'take_profit': actual_price * (1 + self.config.TAKE_PROFIT_PCT / 100)
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
        
        if self.config.VERBOSE:
            print(f"💰 买入: {position_amount:.4f} BTC @ ${actual_price:.2f}")
            print(f"    交易金额: ${trade_value:.2f}，手续费: ${fee:.2f}")
            print(f"    现金余额: ${self.capital:.2f}，BTC持仓: {self.btc_amount:.4f}")
        
        return True
    
    def execute_sell(self, price: float, timestamp: datetime, signal: Dict[str, Any]):
        """执行卖出"""
        if self.btc_amount <= 0:
            if self.config.VERBOSE:
                print("❌ 无BTC持仓，无法卖出")
            return False
        
        # 计算考虑滑点的实际价格
        actual_price = price * (1 - self.config.SLIPPAGE)
        
        # 卖出全部持仓
        position_amount = self.btc_amount
        trade_value = position_amount * actual_price
        fee = trade_value * self.config.FEE_RATE
        
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
            
            if self.config.VERBOSE:
                print(f"💰 卖出: {position_amount:.4f} BTC @ ${actual_price:.2f}")
                print(f"    交易金额: ${trade_value:.2f}，手续费: ${fee:.2f}")
                print(f"    盈亏: ${pnl:.2f} ({pnl_pct:+.2f}%)")
                print(f"    现金余额: ${self.capital:.2f}，BTC持仓: {self.btc_amount:.4f}")
            
            # 清空持仓
            self.position = None
            
            return True
        else:
            if self.config.VERBOSE:
                print("❌ 持仓信息缺失")
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
        """计算当前权益（现金 + 持仓市值）"""
        btc_value = self.btc_amount * current_price
        return self.capital + btc_value
    
    def run_backtest(self):
        """运行回测"""
        print("\n" + "="*60)
        print("🚀 开始缠论策略回测")
        print("="*60)
        
        # 加载数据
        df_main, df_entry = self.load_data()
        
        if df_main.empty:
            print("❌ 主时间框架数据为空，回测终止")
            return
        
        print(f"📊 主数据: {len(df_main)} 根K线")
        if not df_entry.empty:
            print(f"📊 入场数据: {len(df_entry)} 根K线")
        
        # 回测主循环
        print("\n🔍 执行回测分析...")
        
        # 需要至少20根K线才能开始缠论分析
        start_idx = 20
        
        for idx in range(start_idx, len(df_main)):
            current_row = df_main.iloc[idx]
            current_price = current_row['close']
            current_time = current_row['timestamp']
            
            # 检查止损止盈
            if self.position and self.check_stop_loss_take_profit(current_price, current_time):
                continue
            
            # 执行缠论分析
            fractals = self.analyzer.find_fractals(df_main, idx)
            bi_segments = self.analyzer.identify_bi_segments(fractals)
            central_pivots = self.analyzer.find_central_pivot(bi_segments)
            trend_analysis = self.analyzer.analyze_trend(bi_segments, central_pivots)
            
            # 根据持仓状态决定分析类型
            if not self.position:  # 无持仓，寻找买点
                signal = self.analyzer.find_buy_signal(df_main, idx, trend_analysis)
                
                if signal['signal'] == 'BUY' and signal['confidence'] > 0.6:
                    # 如果启用精确入场，检查5分钟级别
                    if self.config.ENABLE_ENTRY_SCOPING and not df_entry.empty:
                        entry_signal = self._check_entry_signal(df_entry, current_time, current_price)
                        if entry_signal:
                            signal = entry_signal
                    
                    if signal['signal'] == 'BUY':
                        self.execute_buy(current_price, current_time, signal)
                        self.signals_history.append(signal)
                        
            else:  # 有持仓，寻找卖点
                signal = self.analyzer.find_sell_signal(df_main, idx, trend_analysis)
                
                if signal['signal'] == 'SELL' and signal['confidence'] > 0.6:
                    self.execute_sell(current_price, current_time, signal)
                    self.signals_history.append(signal)
            
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
            
            # 进度显示
            if idx % 100 == 0:
                progress = (idx - start_idx) / (len(df_main) - start_idx) * 100
                print(f"  进度: {progress:.1f}% ({idx}/{len(df_main)})")
        
        # 回测结束，平掉所有持仓
        if self.btc_amount > 0:
            last_price = df_main.iloc[-1]['close']
            signal = {
                'signal': 'SELL',
                'reason': '回测结束，强制平仓',
                'confidence': 1.0
            }
            self.execute_sell(last_price, df_main.iloc[-1]['timestamp'], signal)
        
        print("\n✅ 回测完成!")
        print(f"   总交易次数: {len(self.trades)}")
        print(f"   最终权益: ${self.calculate_equity(df_main.iloc[-1]['close']):.2f}")
        
        return self.generate_report(df_main)
    
    def _check_entry_signal(self, df_entry: pd.DataFrame, current_time: datetime, current_price: float) -> Optional[Dict[str, Any]]:
        """检查5分钟级别入场信号"""
        # 找到当前时间对应的5分钟K线
        time_threshold = pd.Timedelta(minutes=5)
        
        # 找到最接近当前时间的5分钟K线
        time_diffs = abs(df_entry['timestamp'] - current_time)
        if time_diffs.min() <= time_threshold:
            closest_idx = time_diffs.idxmin()
            entry_row = df_entry.iloc[closest_idx]
            
            # 简单的入场逻辑：价格接近K线低点
            if current_price <= entry_row['low'] * 1.01:
                return {
                    'signal': 'BUY',
                    'reason': '5分钟级别接近低点',
                    'confidence': 0.7,
                    'price': current_price,
                    'timestamp': current_time
                }
        
        return None
    
    def generate_report(self, df_main: pd.DataFrame) -> Dict[str, Any]:
        """生成回测报告"""
        if not self.equity_history:
            return {}
        
        # 计算统计指标
        initial_equity = self.config.INITIAL_CAPITAL
        final_equity = self.equity_history[-1]['equity']
        total_return_pct = (final_equity - initial_equity) / initial_equity * 100
        
        # 计算年化收益率
        start_time = df_main.iloc[0]['timestamp']
        end_time = df_main.iloc[-1]['timestamp']
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
        total_trades = len([t for t in self.trades if t['type'] == 'SELL' and 'pnl' in t])
        
        for trade in self.trades:
            if trade['type'] == 'SELL' and 'pnl' in trade:
                if trade['pnl'] > 0:
                    winning_trades += 1
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 计算夏普比率（简化版）
        returns = []
        for i in range(1, len(self.equity_history)):
            ret = (self.equity_history[i]['equity'] - self.equity_history[i-1]['equity']) / self.equity_history[i-1]['equity']
            returns.append(ret)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(365) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 生成报告
        report = {
            'initial_capital': initial_equity,
            'final_equity': final_equity,
            'total_return_pct': total_return_pct,
            'annual_return_pct': annual_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'trades': self.trades,
            'equity_history': self.equity_history,
            'signals': self.signals_history
        }
        
        return report
    
    def plot_results(self, report: Dict[str, Any]):
        """绘制回测结果图表"""
        if not report or not self.equity_history:
            print("❌ 没有回测数据可绘制")
            return
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. 权益曲线
        times = [h['timestamp'] for h in self.equity_history]
        equities = [h['equity'] for h in self.equity_history]
        prices = [h['price'] for h in self.equity_history]
        
        axes[0].plot(times, equities, 'b-', linewidth=2, label='总权益')
        axes[0].axhline(y=self.config.INITIAL_CAPITAL, color='r', linestyle='--', alpha=0.5, label='初始资金')
        axes[0].set_title(f'缠论策略回测 - 权益曲线 (总收益: {report["total_return_pct"]:.2f}%)', fontsize=14)
        axes[0].set_ylabel('权益 (USDT)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 价格曲线和交易点
        axes[1].plot(times, prices, 'g-', alpha=0.7, label='BTC价格')
        axes[1].set_ylabel('价格 (USDT)')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        # 标记买入卖出点
        buy_times = []
        buy_prices = []
        sell_times = []
        sell_prices = []
        
        for trade in self.trades:
            if trade['type'] == 'BUY':
                buy_times.append(trade['timestamp'])
                buy_prices.append(trade['price'])
            elif trade['type'] == 'SELL':
                sell_times.append(trade['timestamp'])
                sell_prices.append(trade['price'])
        
        if buy_times:
            axes[1].scatter(buy_times, buy_prices, color='green', s=100, marker='^', label='买入', zorder=5)
        if sell_times:
            axes[1].scatter(sell_times, sell_prices, color='red', s=100, marker='v', label='卖出', zorder=5)
        
        axes[1].legend()
        
        # 3. 回撤曲线
        equity_series = pd.Series(equities)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        
        axes[2].fill_between(times, drawdown, 0, color='red', alpha=0.3)
        axes[2].plot(times, drawdown, 'r-', linewidth=1)
        axes[2].set_title(f'最大回撤: {report["max_drawdown_pct"]:.2f}%', fontsize=12)
        axes[2].set_xlabel('时间')
        axes[2].set_ylabel('回撤 (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'reports/backtest_result_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        print(f"📈 图表已保存: {filename}")
        plt.show()
    
    def print_report(self, report: Dict[str, Any]):
        """打印回测报告"""
        print("\n" + "="*60)
        print("📊 缠论策略回测报告")
        print("="*60)
        
        print(f"💰 资金统计:")
        print(f"   初始资金: ${report['initial_capital']:,.2f}")
        print(f"   最终权益: ${report['final_equity']:,.2f}")
        print(f"   总收益率: {report['total_return_pct']:+.2f}%")
        print(f"   年化收益率: {report['annual_return_pct']:+.2f}%")
        
        print(f"\n📈 风险指标:")
        print(f"   最大回撤: {report['max_drawdown_pct']:.2f}%")
        print(f"   夏普比率: {report['sharpe_ratio']:.2f}")
        
        print(f"\n📊 交易统计:")
        print(f"   总交易次数: {report['total_trades']}")
        print(f"   盈利交易: {report['winning_trades']}")
        print(f"   胜率: {report['win_rate']:.1f}%")
        
        print(f"\n🔧 策略配置:")
        print(f"   时间框架: {self.config.TIMEFRAME}")
        print(f"   入场框架: {self.config.ENTRY_TIMEFRAME if self.config.ENABLE_ENTRY_SCOPING else 'N/A'}")
        print(f"   止损: {self.config.STOP_LOSS_PCT}%")
        print(f"   止盈: {self.config.TAKE_PROFIT_PCT}%")
        print(f"   手续费率: {self.config.FEE_RATE * 100}%")
        
        # 显示最近几笔交易
        if self.trades:
            print(f"\n📋 最近交易记录:")
            recent_trades = self.trades[-10:] if len(self.trades) > 10 else self.trades
            
            for trade in recent_trades:
                time_str = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
                if trade['type'] == 'BUY':
                    print(f"   {time_str} | 买入 {trade['amount']:.4f} BTC @ ${trade['price']:.2f}")
                else:
                    pnl_str = f"(${trade['pnl']:+.2f}, {trade['pnl_pct']:+.2f}%)" if 'pnl' in trade else ""
                    print(f"   {time_str} | 卖出 {trade['amount']:.4f} BTC @ ${trade['price']:.2f} {pnl_str}")
        
        print("="*60)

# ====================== 参数优化器 ======================
class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def optimize_parameters(self):
        """优化策略参数"""
        print("\n" + "="*60)
        print("🔧 开始参数优化")
        print("="*60)
        
        # 定义参数范围
        param_grid = {
            'stop_loss_pct': [1.0, 1.5, 2.0, 2.5],
            'take_profit_pct': [2.0, 3.0, 4.0, 5.0],
            'fractal_period': [3, 5, 7],
            'confidence_threshold': [0.5, 0.6, 0.7]
        }
        
        results = []
        
        # 网格搜索
        for sl in param_grid['stop_loss_pct']:
            for tp in param_grid['take_profit_pct']:
                for fp in param_grid['fractal_period']:
                    for ct in param_grid['confidence_threshold']:
                        print(f"\n测试参数: SL={sl}%, TP={tp}%, FP={fp}, CT={ct}")
                        
                        # 创建配置副本
                        test_config = BacktestConfig()
                        test_config.STOP_LOSS_PCT = sl
                        test_config.TAKE_PROFIT_PCT = tp
                        test_config.FRACTAL_PERIOD = fp
                        test_config.VERBOSE = False
                        
                        # 运行回测
                        engine = BacktestEngine(test_config)
                        report = engine.run_backtest()
                        
                        if report:
                            results.append({
                                'stop_loss_pct': sl,
                                'take_profit_pct': tp,
                                'fractal_period': fp,
                                'confidence_threshold': ct,
                                'total_return_pct': report['total_return_pct'],
                                'max_drawdown_pct': report['max_drawdown_pct'],
                                'win_rate': report['win_rate'],
                                'sharpe_ratio': report['sharpe_ratio']
                            })
        
        # 分析结果
        if results:
            results_df = pd.DataFrame(results)
            
            # 按夏普比率排序
            best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
            
            # 按收益率排序
            best_return = results_df.loc[results_df['total_return_pct'].idxmax()]
            
            # 按风险调整收益排序（收益率/回撤）
            results_df['risk_adjusted'] = results_df['total_return_pct'] / abs(results_df['max_drawdown_pct'])
            best_risk_adj = results_df.loc[results_df['risk_adjusted'].idxmax()]
            
            print("\n" + "="*60)
            print("🏆 最佳参数组合")
            print("="*60)
            
            print(f"\n🎯 最佳夏普比率:")
            print(f"   止损: {best_sharpe['stop_loss_pct']}%")
            print(f"   止盈: {best_sharpe['take_profit_pct']}%")
            print(f"   分型周期: {best_sharpe['fractal_period']}")
            print(f"   信心阈值: {best_sharpe['confidence_threshold']}")
            print(f"   夏普比率: {best_sharpe['sharpe_ratio']:.2f}")
            print(f"   总收益: {best_sharpe['total_return_pct']:.2f}%")
            print(f"   最大回撤: {best_sharpe['max_drawdown_pct']:.2f}%")
            
            print(f"\n💰 最高收益率:")
            print(f"   止损: {best_return['stop_loss_pct']}%")
            print(f"   止盈: {best_return['take_profit_pct']}%")
            print(f"   分型周期: {best_return['fractal_period']}")
            print(f"   信心阈值: {best_return['confidence_threshold']}")
            print(f"   总收益: {best_return['total_return_pct']:.2f}%")
            print(f"   最大回撤: {best_return['max_drawdown_pct']:.2f}%")
            
            print(f"\n⚖️  最佳风险调整收益:")
            print(f"   止损: {best_risk_adj['stop_loss_pct']}%")
            print(f"   止盈: {best_risk_adj['take_profit_pct']}%")
            print(f"   分型周期: {best_risk_adj['fractal_period']}")
            print(f"   信心阈值: {best_risk_adj['confidence_threshold']}")
            print(f"   风险调整收益: {best_risk_adj['risk_adjusted']:.2f}")
            print(f"   总收益: {best_risk_adj['total_return_pct']:.2f}%")
            print(f"   最大回撤: {best_risk_adj['max_drawdown_pct']:.2f}%")
            
            # 保存结果
            os.makedirs('reports', exist_ok=True)
            results_df.to_csv('reports/parameter_optimization.csv', index=False)
            print(f"\n💾 优化结果已保存: reports/parameter_optimization.csv")
            
            return best_sharpe, best_return, best_risk_adj
        
        return None, None, None

# ====================== 主函数 ======================
def main():
    """主函数"""
    print("="*60)
    print("🤖 BTC缠论策略回测系统")
    print("版本: 1.0.0 | 作者: AI交易助手")
    print("="*60)
    
    # 创建配置
    config = BacktestConfig()
    
    # 用户选择模式
    print("\n请选择回测模式:")
    print("1. 单次回测")
    print("2. 参数优化")
    print("3. 批量回测不同参数")
    
    choice = input("\n请输入选项 (1-3): ").strip()
    
    if choice == '1':
        # 单次回测
        print("\n📊 运行单次回测...")
        
        # 可修改配置参数
        config.VERBOSE = input("显示详细日志? (y/n): ").lower() == 'y'
        
        # 运行回测
        engine = BacktestEngine(config)
        report = engine.run_backtest()
        
        if report:
            engine.print_report(report)
            engine.plot_results(report)
            
            # 询问是否保存报告
            save = input("\n是否保存回测报告? (y/n): ").lower()
            if save == 'y':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_file = f'reports/backtest_report_{timestamp}.json'
                
                # 简化报告数据以保存
                simple_report = {
                    'summary': {
                        'initial_capital': report['initial_capital'],
                        'final_equity': report['final_equity'],
                        'total_return_pct': report['total_return_pct'],
                        'annual_return_pct': report['annual_return_pct'],
                        'max_drawdown_pct': report['max_drawdown_pct'],
                        'win_rate': report['win_rate'],
                        'sharpe_ratio': report['sharpe_ratio'],
                        'total_trades': report['total_trades']
                    },
                    'config': {
                        'symbol': config.SYMBOL,
                        'timeframe': config.TIMEFRAME,
                        'stop_loss_pct': config.STOP_LOSS_PCT,
                        'take_profit_pct': config.TAKE_PROFIT_PCT,
                        'fee_rate': config.FEE_RATE
                    }
                }
                
                with open(report_file, 'w') as f:
                    json.dump(simple_report, f, indent=2, default=str)
                
                print(f"✅ 报告已保存: {report_file}")
    
    elif choice == '2':
        # 参数优化
        print("\n🔧 运行参数优化...")
        
        optimizer = ParameterOptimizer(config)
        optimizer.optimize_parameters()
    
    elif choice == '3':
        # 批量回测
        print("\n📈 运行批量回测...")
        
        # 测试不同时间框架
        timeframes = ['15m', '30m', '1h', '4h']
        
        for tf in timeframes:
            print(f"\n测试时间框架: {tf}")
            config.TIMEFRAME = tf
            
            engine = BacktestEngine(config)
            report = engine.run_backtest()
            
            if report:
                print(f"  收益率: {report['total_return_pct']:.2f}%")
                print(f"  最大回撤: {report['max_drawdown_pct']:.2f}%")
                print(f"  夏普比率: {report['sharpe_ratio']:.2f}")
    
    else:
        print("❌ 无效选项")

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 运行主函数
    main()