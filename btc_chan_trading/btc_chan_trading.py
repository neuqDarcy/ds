#!/usr/bin/env python3
"""
BTC/USDT缠论+AI量化交易机器人
作者: AI交易助手
版本: 1.0.0
描述: 基于缠论技术分析和DeepSeek AI的多时间框架自动交易系统
"""

import os
import time
import json
import schedule
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass

# 第三方库
from openai import OpenAI
import ccxt
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 忽略警告
warnings.filterwarnings('ignore')

# ====================== 配置类 ======================
@dataclass
class Config:
    """交易配置"""
    # 交易对和参数
    SYMBOL: str = "BTC/USDT"
    TRADE_AMOUNT: float = 0.001  # 每次交易BTC数量
    LEVERAGE: int = 10  # 杠杆倍数
    
    # 时间框架配置
    SIGNAL_TIMEFRAME: str = "30m"  # 信号时间框架（买点判断）
    ENTRY_TIMEFRAME: str = "5m"    # 入场时间框架（精确入场）
    EXIT_TIMEFRAME: str = "15m"    # 离场时间框架
    
    # 交易状态
    TEST_MODE: bool = True  # 测试模式，不真实下单
    ENABLE_AI_ANALYSIS: bool = True  # 启用AI分析
    
    # 风险控制
    STOP_LOSS_PCT: float = 15  # 止损百分比
    TAKE_PROFIT_PCT: float = 30  # 止盈百分比
    MAX_POSITION_SIZE: float = 0.01  # 最大持仓
    
    # 缠论参数
    FRACTAL_PERIOD: int = 5  # 分型周期
    CENTRAL_PIVOT_LOOKBACK: int = 20  # 中枢观察周期

# ====================== 枚举定义 ======================
class TradingMode(Enum):
    """交易状态枚举"""
    MONITORING = "monitoring"       # 监控状态，等待30分钟买点
    ENTRY_SCOPING = "entry_scoping" # 入场侦察，寻找5分钟入场点
    IN_POSITION = "in_position"     # 持仓状态
    COOLDOWN = "cooldown"           # 冷却状态

# ====================== 交易所管理器 ======================
class ExchangeManager:
    """交易所管理器 - 处理所有交易所相关的操作"""
    
    def __init__(self, config: Config):
        self.config = config
        self.exchange = None
        self.initialized = False
        
        if not config.TEST_MODE:
            self._init_real_exchange()
        else:
            print("🔧 运行在模拟模式，不会真实下单")
            self.initialized = True
    
    def _init_real_exchange(self):
        """初始化真实交易所连接"""
        try:
            api_key = os.getenv('BINANCE_API_KEY')
            secret = os.getenv('BINANCE_SECRET')
            
            if not api_key or not secret:
                raise ValueError("请在.env文件中设置BINANCE_API_KEY和BINANCE_SECRET")
            
            self.exchange = ccxt.binance({
                'options': {'defaultType': 'future'},
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True,
                'timeout': 30000
            })
            print("✅ 交易所连接初始化成功")
            self.initialized = True
        except Exception as e:
            print(f"❌ 交易所连接失败: {e}")
            self.initialized = False
    
    def get_ohlcv(self, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """获取K线数据"""
        try:
            return self._get_real_data(timeframe, limit)
        except Exception as e:
            print(f"❌ 获取K线数据失败: {e}")
            return pd.DataFrame()
    
    def _get_real_data(self, timeframe: str, limit: int) -> pd.DataFrame:
        """获取真实交易所数据"""
        if not self.initialized or not self.exchange:
            raise Exception("交易所未初始化")
        
        ohlcv = self.exchange.fetch_ohlcv(
            self.config.SYMBOL, 
            timeframe, 
            limit=limit
        )
        
        df = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def _get_mock_data(self, timeframe: str, limit: int) -> pd.DataFrame:
        """生成模拟数据用于测试"""
        # 时间间隔映射
        timeframe_to_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes = timeframe_to_minutes.get(timeframe, 30)
        
        # 生成时间序列
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(minutes=minutes * limit)
        dates = pd.date_range(start=start_time, end=end_time, periods=limit)
        
        # 生成价格序列
        np.random.seed(int(time.time()))  # 使用时间作为随机种子
        base_price = 50000 + np.random.uniform(-5000, 5000)
        
        # 创建价格序列（带趋势和波动）
        trend = np.random.uniform(-0.001, 0.001, limit).cumsum()
        noise = np.random.normal(0, 0.005, limit)
        price_changes = trend + noise
        
        prices = [base_price]
        for i in range(1, limit):
            new_price = prices[-1] * (1 + price_changes[i])
            prices.append(new_price)
        
        # 生成OHLCV数据
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        for i in range(limit):
            price = prices[i]
            open_price = price * np.random.uniform(0.998, 1.002) if i > 0 else price * 0.999
            
            high_price = max(open_price, price) * np.random.uniform(1.001, 1.01)
            low_price = min(open_price, price) * np.random.uniform(0.99, 0.999)
            close_price = price
            
            price_range = high_price - low_price
            avg_price = (open_price + close_price) / 2
            volume = price_range / avg_price * np.random.uniform(100, 1000)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
    
    def get_current_price(self) -> float:
        """获取当前价格"""
        try:
            if self.config.TEST_MODE:
                df = self.get_ohlcv("1m", limit=2)
                return float(df['close'].iloc[-1]) if not df.empty else 50000
            else:
                ticker = self.exchange.fetch_ticker(self.config.SYMBOL)
                return float(ticker['last'])
        except Exception as e:
            print(f"获取当前价格失败: {e}")
            return 0.0
    
    def get_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        if self.config.TEST_MODE:
            return {
                'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
                'BTC': {'free': 0.1, 'used': 0.0, 'total': 0.1}
            }
        
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            print(f"获取余额失败: {e}")
            return {}
    
    def place_test_order(self, side: str, amount: float) -> Dict[str, Any]:
        """模拟下单"""
        current_price = self.get_current_price()
        print(f"📝 模拟下单: {side} {amount} {self.config.SYMBOL} @ ${current_price:.2f}")
        
        return {
            'id': f"test_order_{int(time.time())}",
            'symbol': self.config.SYMBOL,
            'side': side,
            'amount': amount,
            'price': current_price,
            'status': 'closed',
            'timestamp': datetime.now().isoformat(),
            'cost': current_price * amount
        }
    
    def place_real_order(self, side: str, amount: float) -> Dict[str, Any]:
        """真实下单"""
        try:
            order = self.exchange.create_order(
                symbol=self.config.SYMBOL,
                type='market',
                side=side,
                amount=amount
            )
            print(f"✅ 订单执行成功: {order['id']}")
            return order
        except Exception as e:
            print(f"❌ 下单失败: {e}")
            raise

# ====================== 缠论分析器 ======================
class ChanTheoryAnalyzer:
    """缠论分析器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fractals_history = []
        self.bi_segments_history = []
        self.central_pivots_history = []
    
    def find_fractals(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        识别顶分型和底分型
        缠论定义：至少5根K线，中间K线最高/最低，两边递减/递增
        """
        if len(df) < 5:
            return {'top': [], 'bottom': []}
        
        highs = df['high'].values
        lows = df['low'].values
        
        top_fractals, bottom_fractals = [], []
        
        for i in range(2, len(df) - 2):
            # 顶分型：中间K线最高
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                top_fractals.append({
                    'index': i,
                    'price': highs[i],
                    'timestamp': df.iloc[i]['timestamp'],
                    'type': 'top'
                })
            
            # 底分型：中间K线最低
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                bottom_fractals.append({
                    'index': i,
                    'price': lows[i],
                    'timestamp': df.iloc[i]['timestamp'],
                    'type': 'bottom'
                })
        
        return {'top': top_fractals, 'bottom': bottom_fractals}
    
    def identify_bi_segments(self, fractals: Dict[str, List]) -> List[Dict]:
        """
        识别笔：相邻的顶分型和底分型
        """
        all_fractals = fractals['top'] + fractals['bottom']
        all_fractals.sort(key=lambda x: x['index'])
        
        bi_segments = []
        for i in range(len(all_fractals) - 1):
            current = all_fractals[i]
            next_f = all_fractals[i + 1]
            
            # 确保笔的方向正确（顶-底 或 底-顶）
            if current['type'] != next_f['type']:
                bi_segments.append({
                    'start': current,
                    'end': next_f,
                    'direction': 'up' if current['type'] == 'bottom' else 'down',
                    'price_change': next_f['price'] - current['price'],
                    'pct_change': (next_f['price'] - current['price']) / current['price'] * 100
                })
        
        return bi_segments
    
    def find_central_pivot(self, bi_segments: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """
        识别中枢：至少三个连续重叠的笔
        """
        if len(bi_segments) < 3:
            return []
        
        central_pivots = []
        
        for i in range(len(bi_segments) - 2):
            seg1 = bi_segments[i]
            seg2 = bi_segments[i + 1]
            seg3 = bi_segments[i + 2]
            
            # 计算重叠区域
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
            
            # 如果有重叠区域，形成一个中枢
            if overlap_low < overlap_high:
                central_pivots.append({
                    'start_idx': seg1['start']['index'],
                    'end_idx': seg3['end']['index'],
                    'high': overlap_high,
                    'low': overlap_low,
                    'center': (overlap_high + overlap_low) / 2,
                    'width': overlap_high - overlap_low,
                    'strength': abs(seg1['pct_change'] + seg2['pct_change'] + seg3['pct_change'])
                })
        
        return central_pivots
    
    def analyze_trend(self, bi_segments: List[Dict], central_pivots: List[Dict]) -> Dict[str, Any]:
        """
        分析趋势：上涨趋势、下跌趋势、盘整
        """
        if len(bi_segments) < 2:
            return {'trend': 'unknown', 'strength': 0, 'has_central_pivot': False}
        
        # 计算最近笔的方向
        recent_bis = bi_segments[-3:] if len(bi_segments) >= 3 else bi_segments
        up_count = sum(1 for bi in recent_bis if bi['direction'] == 'up')
        down_count = len(recent_bis) - up_count
        
        # 计算趋势强度
        strength = 0
        if recent_bis:
            price_changes = [bi['pct_change'] for bi in recent_bis]
            strength = abs(np.mean(price_changes))
        
        # 判断趋势
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
    
    def find_buy_signal(self, df: pd.DataFrame, trend_analysis: Dict) -> Dict[str, Any]:
        """
        寻找买点信号
        缠论三类买点：
        1. 中枢下方的背驰买点
        2. 中枢上方的突破买点
        3. 中枢内部的盘整买点
        """
        current_price = df['close'].iloc[-1]
        recent_low = df['low'].tail(10).min()
        recent_high = df['high'].tail(10).max()
        
        signals = []
        
        # 信号1：价格接近近期低点且趋势可能反转
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
        
        # 信号3：中枢内部盘整
        if trend_analysis['has_central_pivot'] and trend_analysis['trend'] == 'consolidation':
            signals.append({
                'type': 'type3_consolidation',
                'price': current_price,
                'description': '第三类买点：中枢盘整',
                'confidence': 0.6
            })
        
        return {
            'signals': signals,
            'current_price': current_price,
            'recommendation': 'BUY' if signals else 'HOLD'
        }
    
    def find_sell_signal(self, df: pd.DataFrame, trend_analysis: Dict) -> Dict[str, Any]:
        """
        寻找卖点信号
        """
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        
        signals = []
        
        # 信号1：价格接近近期高点且趋势可能反转
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
        
        return {
            'signals': signals,
            'current_price': current_price,
            'recommendation': 'SELL' if signals else 'HOLD'
        }

# ====================== AI分析器 ======================
class AIAnalyzer:
    """DeepSeek AI分析器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.history = []
        
        if config.ENABLE_AI_ANALYSIS:
            self._init_ai_client()
    
    def _init_ai_client(self):
        """初始化AI客户端"""
        try:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                print("⚠️  DEEPSEEK_API_KEY未设置，AI分析功能将禁用")
                self.config.ENABLE_AI_ANALYSIS = False
                return
            
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            print("✅ AI客户端初始化成功")
        except Exception as e:
            print(f"❌ AI客户端初始化失败: {e}")
            self.config.ENABLE_AI_ANALYSIS = False
    
    def analyze_with_chan_theory(self, df: pd.DataFrame, chan_data: Dict) -> Dict[str, Any]:
        """
        结合缠论数据和AI进行深度分析
        """
        if not self.config.ENABLE_AI_ANALYSIS or not self.client:
            return {
                'signal': 'HOLD',
                'reason': 'AI分析功能禁用',
                'confidence': 0.5,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'chan_pattern': 'unknown'
            }
        
        # 准备数据摘要
        recent_data = df.tail(20)
        price_summary = {
            'current': df['close'].iloc[-1],
            'high_24h': df['high'].tail(24).max(),
            'low_24h': df['low'].tail(24).min(),
            'volume_24h': df['volume'].tail(24).sum(),
            'price_change_24h': (df['close'].iloc[-1] - df['close'].iloc[-25]) / df['close'].iloc[-25] * 100 if len(df) > 25 else 0
        }
        
        # 构建缠论分析提示词
        prompt = self._build_chan_prompt(df, chan_data, price_summary)
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个精通缠论的加密货币交易专家。请基于缠论原理进行分析。
                        缠论核心概念：
                        1. 分型：顶分型和底分型
                        2. 笔：相邻分型之间的连线
                        3. 中枢：至少三个连续重叠的笔
                        4. 买卖点：三类买点和三类卖点
                        
                        请严格基于缠论逻辑进行分析，给出明确的交易建议。"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            ai_signal = self._parse_ai_response(result)
            
            # 保存到历史
            self.history.append({
                'timestamp': datetime.now(),
                'analysis': result[:200] + '...' if len(result) > 200 else result,
                'signal': ai_signal
            })
            
            if len(self.history) > 50:
                self.history.pop(0)
            
            return ai_signal
            
        except Exception as e:
            print(f"❌ AI分析失败: {e}")
            return {
                'signal': 'HOLD',
                'reason': 'AI分析失败',
                'confidence': 0.5,
                'entry_price': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'chan_pattern': 'unknown'
            }
    
    def _build_chan_prompt(self, df: pd.DataFrame, chan_data: Dict, price_summary: Dict) -> str:
        """构建缠论分析提示词"""
        prompt = f"""
        请基于以下比特币({self.config.SIGNAL_TIMEFRAME}级别)的缠论数据进行交易分析：
        
        【价格数据】
        - 当前价格: ${price_summary['current']:,.2f}
        - 24小时高点: ${price_summary['high_24h']:,.2f}
        - 24小时低点: ${price_summary['low_24h']:,.2f}
        - 24小时成交量: {price_summary['volume_24h']:,.0f} BTC
        - 24小时价格变化: {price_summary['price_change_24h']:+.2f}%
        
        【缠论结构分析】
        - 当前趋势: {chan_data.get('trend', 'unknown')}
        - 趋势强度: {chan_data.get('strength', 0):.2f}
        - 分型数量: 顶分型{len(chan_data.get('fractals', {}).get('top', []))}, 底分型{len(chan_data.get('fractals', {}).get('bottom', []))}
        - 笔数量: {len(chan_data.get('bi_segments', []))}
        - 中枢数量: {len(chan_data.get('central_pivots', []))}
        
        【最近笔数据】
        {self._format_recent_bis(chan_data.get('bi_segments', []))}
        
        【最近中枢信息】
        {self._format_central_pivots(chan_data.get('central_pivots', []))}
        
        【交易建议要求】
        1. 基于缠论给出明确的交易信号：BUY, SELL 或 HOLD
        2. 解释缠论分析逻辑（分型、笔、中枢、买卖点）
        3. 给出建议的入场价位
        4. 建议止损价位
        5. 建议止盈价位
        6. 评估信号信心程度（0-100%）
        
        请用JSON格式回复：
        {{
            "signal": "BUY|SELL|HOLD",
            "reason": "基于缠论的分析理由",
            "entry_price": 建议入场价,
            "stop_loss": 止损价,
            "take_profit": 止盈价,
            "confidence": 0.8,
            "chan_pattern": "描述缠论形态"
        }}
        """
        return prompt
    
    def _format_recent_bis(self, bi_segments: List[Dict]) -> str:
        """格式化最近笔数据"""
        if not bi_segments:
            return "无笔数据"
        
        recent = bi_segments[-5:] if len(bi_segments) > 5 else bi_segments
        lines = []
        for i, bi in enumerate(recent, 1):
            lines.append(f"  笔{i}: {bi['direction']} | 价格变化: {bi['pct_change']:+.2f}% | "
                        f"从{bi['start']['price']:.2f}到{bi['end']['price']:.2f}")
        return "\n".join(lines)
    
    def _format_central_pivots(self, pivots: List[Dict]) -> str:
        """格式化中枢数据"""
        if not pivots:
            return "无中枢"
        
        recent = pivots[-3:] if len(pivots) > 3 else pivots
        lines = []
        for i, pivot in enumerate(recent, 1):
            lines.append(f"  中枢{i}: 范围${pivot['low']:.2f}-${pivot['high']:.2f} | "
                        f"中枢中心: ${pivot['center']:.2f} | 宽度: ${pivot['width']:.2f}")
        return "\n".join(lines)
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """解析AI响应"""
        try:
            # 提取JSON部分
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ['signal', 'reason', 'confidence']
                if all(field in data for field in required_fields):
                    return {
                        'signal': data['signal'],
                        'reason': data['reason'],
                        'entry_price': data.get('entry_price', 0),
                        'stop_loss': data.get('stop_loss', 0),
                        'take_profit': data.get('take_profit', 0),
                        'confidence': float(data['confidence']),
                        'chan_pattern': data.get('chan_pattern', 'unknown')
                    }
        except Exception as e:
            print(f"❌ 解析AI响应失败: {e}")
        
        # 如果解析失败，尝试提取信号
        if 'BUY' in response.upper():
            signal = 'BUY'
        elif 'SELL' in response.upper():
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'signal': signal,
            'reason': 'AI响应解析失败，基于关键词判断',
            'confidence': 0.5,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'chan_pattern': 'unknown'
        }

# ====================== 状态管理器 ======================
class TradingStateManager:
    """交易状态管理器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.current_mode = TradingMode.MONITORING
        self.mode_start_time = datetime.now()
        self.signal_30m = None
        self.entry_signal_5m = None
        self.position = None
        self.signal_expiry_time = None
        self.entry_attempts = 0
        self.trade_history = []
        
    def change_mode(self, new_mode: TradingMode):
        """切换交易模式"""
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.mode_start_time = datetime.now()
        
        print(f"🔄 交易模式切换: {old_mode.value} → {new_mode.value}")
        
        # 重置相关状态
        if new_mode == TradingMode.MONITORING:
            self.signal_30m = None
            self.entry_signal_5m = None
            self.entry_attempts = 0
            self.signal_expiry_time = None
            
        elif new_mode == TradingMode.ENTRY_SCOPING:
            # 设置信号过期时间（30分钟信号有效2小时）
            self.signal_expiry_time = datetime.now() + timedelta(hours=2)
            
        elif new_mode == TradingMode.COOLDOWN:
            self.position = None
            
    def check_signal_expiry(self) -> bool:
        """检查信号是否过期"""
        if (self.current_mode == TradingMode.ENTRY_SCOPING and 
            self.signal_expiry_time and 
            datetime.now() > self.signal_expiry_time):
            print("⏰ 30分钟信号已过期，返回监控模式")
            self.change_mode(TradingMode.MONITORING)
            return True
        return False
    
    def check_entry_timeout(self) -> bool:
        """检查入场是否超时（最多等待2小时）"""
        if self.current_mode == TradingMode.ENTRY_SCOPING:
            time_in_mode = (datetime.now() - self.mode_start_time).total_seconds() / 60
            if time_in_mode > 120:  # 2小时超时
                print(f"⏰ 入场等待超时（{time_in_mode:.0f}分钟），返回监控模式")
                self.change_mode(TradingMode.MONITORING)
                return True
        return False
    
    def increment_entry_attempts(self):
        """增加入场尝试次数"""
        self.entry_attempts += 1
        if self.entry_attempts >= 10:  # 最多尝试10次
            print("🔄 入场尝试次数过多，返回监控模式")
            self.change_mode(TradingMode.MONITORING)
    
    def record_trade(self, signal: Dict, amount: float, price: float, side: str):
        """记录交易历史"""
        trade_record = {
            'timestamp': datetime.now(),
            'signal': signal.get('signal'),
            'side': side,
            'amount': amount,
            'price': price,
            'entry_price': signal.get('entry_price', 0),
            'stop_loss': signal.get('stop_loss', 0),
            'take_profit': signal.get('take_profit', 0),
            'confidence': signal.get('confidence', 0),
            'chan_pattern': signal.get('chan_pattern', ''),
            'mode': self.current_mode.value
        }
        
        self.trade_history.append(trade_record)
        
        # 保持历史记录大小
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
    
    def get_mode_duration(self) -> float:
        """获取当前模式的持续时间（分钟）"""
        return (datetime.now() - self.mode_start_time).total_seconds() / 60

# ====================== 多时间框架分析器 ======================
class MultiTimeframeAnalyzer:
    """多时间框架分析器"""
    
    def __init__(self, config: Config, exchange: ExchangeManager):
        self.config = config
        self.exchange = exchange
        self.chan_analyzers = {
            '30m': ChanTheoryAnalyzer(config),
            '5m': ChanTheoryAnalyzer(config),
            '15m': ChanTheoryAnalyzer(config)
        }
        
    def analyze_30m_signal(self) -> Dict[str, Any]:
        """分析30分钟级别信号"""
        print(f"\n📈 开始分析{self.config.SIGNAL_TIMEFRAME}级别信号...")
        
        df_30m = self.exchange.get_ohlcv(self.config.SIGNAL_TIMEFRAME, limit=50)
        if df_30m.empty:
            return {'signal': 'NO_DATA', 'confidence': 0, 'reason': '数据获取失败'}
        
        # 执行缠论分析
        analyzer = self.chan_analyzers['30m']
        
        # 1. 识别分型
        fractals = analyzer.find_fractals(df_30m)
        
        # 2. 识别笔
        bi_segments = analyzer.identify_bi_segments(fractals)
        
        # 3. 识别中枢
        central_pivots = analyzer.find_central_pivot(bi_segments, df_30m)
        
        # 4. 分析趋势
        trend_analysis = analyzer.analyze_trend(bi_segments, central_pivots)
        
        # 5. 寻找买点信号
        buy_signals = analyzer.find_buy_signal(df_30m, trend_analysis)
        
        # 6. 寻找卖点信号
        sell_signals = analyzer.find_sell_signal(df_30m, trend_analysis)
        
        current_price = df_30m['close'].iloc[-1]
        
        # 判断信号强度
        if buy_signals['signals']:
            best_buy = max(buy_signals['signals'], key=lambda x: x['confidence'])
            
            # 计算买点区域
            buy_zone = self._calculate_buy_zone(df_30m, best_buy)
            
            return {
                'signal': 'BUY',
                'reason': best_buy['description'],
                'confidence': best_buy['confidence'],
                'buy_zone': buy_zone,
                'current_price': current_price,
                'trend': trend_analysis['trend'],
                'has_central_pivot': trend_analysis['has_central_pivot'],
                'fractal_count': len(fractals['top']) + len(fractals['bottom']),
                'bi_count': len(bi_segments),
                'pivot_count': len(central_pivots),
                'timestamp': datetime.now()
            }
        
        return {
            'signal': 'HOLD',
            'reason': '30分钟无买点信号',
            'confidence': 0,
            'current_price': current_price,
            'trend': trend_analysis['trend']
        }
    
    def analyze_5m_entry(self, buy_zone: Dict[str, Any]) -> Dict[str, Any]:
        """分析5分钟级别精确入场点"""
        print(f"\n🎯 开始分析{self.config.ENTRY_TIMEFRAME}级别入场点...")
        
        df_5m = self.exchange.get_ohlcv(self.config.ENTRY_TIMEFRAME, limit=30)
        if df_5m.empty:
            return {'entry_signal': 'NO_DATA', 'confidence': 0}
        
        current_price = df_5m['close'].iloc[-1]
        
        # 检查是否在买点区域内
        if not self._in_buy_zone(current_price, buy_zone):
            return {
                'entry_signal': 'OUTSIDE_ZONE',
                'reason': f'价格{current_price:.2f}不在买点区域{buy_zone["low"]:.2f}-{buy_zone["high"]:.2f}',
                'confidence': 0,
                'current_price': current_price
            }
        
        # 执行5分钟缠论分析
        analyzer = self.chan_analyzers['5m']
        
        # 识别分型
        fractals = analyzer.find_fractals(df_5m)
        
        # 寻找最近的底分型
        bottom_fractals = fractals['bottom']
        if bottom_fractals:
            recent_bottom = bottom_fractals[-1]
            
            # 检查是否为新鲜底分型（最近3根K线内）
            if len(df_5m) - recent_bottom['index'] <= 3:
                # 计算止损和止盈
                stop_loss = recent_bottom['price'] * (1 - self.config.STOP_LOSS_PCT / 100)
                take_profit = recent_bottom['price'] * (1 + self.config.TAKE_PROFIT_PCT / 100)
                
                return {
                    'entry_signal': 'BUY',
                    'entry_price': recent_bottom['price'],
                    'confidence': 0.8,
                    'reason': '5分钟底分型确认',
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'fractal_price': recent_bottom['price']
                }
        
        # 如果没有底分型，检查价格是否在支撑位附近
        recent_low = df_5m['low'].tail(5).min()
        if current_price <= recent_low * 1.01:  # 在近期低点1%范围内
            stop_loss = current_price * (1 - self.config.STOP_LOSS_PCT / 100)
            take_profit = current_price * (1 + self.config.TAKE_PROFIT_PCT / 100)
            
            return {
                'entry_signal': 'PREPARE_BUY',
                'entry_price': current_price,
                'confidence': 0.6,
                'reason': '价格在近期低点附近',
                'current_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
        
        return {
            'entry_signal': 'HOLD',
            'reason': '等待更好的入场时机',
            'confidence': 0.3,
            'current_price': current_price
        }
    
    def analyze_exit_signal(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """分析离场信号"""
        if not position:
            return {'exit_signal': 'HOLD', 'reason': '无持仓'}
        
        # 获取当前价格
        current_price = self.exchange.get_current_price()
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        side = position.get('side', 'long')
        
        # 检查止损止盈
        if side == 'long':
            if current_price <= stop_loss:
                return {
                    'exit_signal': 'SELL',
                    'reason': f'触发止损: {current_price:.2f} <= {stop_loss:.2f}',
                    'confidence': 1.0
                }
            if current_price >= take_profit:
                return {
                    'exit_signal': 'SELL',
                    'reason': f'触发止盈: {current_price:.2f} >= {take_profit:.2f}',
                    'confidence': 1.0
                }
        else:  # short position
            if current_price >= stop_loss:
                return {
                    'exit_signal': 'BUY',
                    'reason': f'触发止损: {current_price:.2f} >= {stop_loss:.2f}',
                    'confidence': 1.0
                }
            if current_price <= take_profit:
                return {
                    'exit_signal': 'BUY',
                    'reason': f'触发止盈: {current_price:.2f} <= {take_profit:.2f}',
                    'confidence': 1.0
                }
        
        # 分析15分钟级别是否有卖点
        df_15m = self.exchange.get_ohlcv(self.config.EXIT_TIMEFRAME, limit=20)
        if not df_15m.empty:
            analyzer = self.chan_analyzers['15m']
            fractals = analyzer.find_fractals(df_15m)
            
            if side == 'long' and fractals['top']:
                recent_top = fractals['top'][-1]
                if len(df_15m) - recent_top['index'] <= 2:
                    return {
                        'exit_signal': 'SELL',
                        'reason': '15分钟顶分型出现',
                        'confidence': 0.7
                    }
        
        return {'exit_signal': 'HOLD', 'reason': '继续持有', 'confidence': 0.5}
    
    def _calculate_buy_zone(self, df: pd.DataFrame, buy_signal: Dict[str, Any]) -> Dict[str, Any]:
        """计算买点区域"""
        current_price = df['close'].iloc[-1]
        
        if buy_signal['type'] == 'type1_bottom_divergence':
            recent_low = df['low'].tail(10).min()
            zone_high = recent_low * 1.015  # 1.5%范围
            zone_low = recent_low * 0.985   # -1.5%范围
        elif buy_signal['type'] == 'type2_breakout':
            recent_high = df['high'].tail(10).max()
            zone_high = recent_high * 1.01   # 1%范围
            zone_low = recent_high * 0.99    # -1%范围
        else:
            zone_high = current_price * 1.01
            zone_low = current_price * 0.99
        
        return {
            'low': zone_low,
            'high': zone_high,
            'center': (zone_high + zone_low) / 2,
            'width_pct': (zone_high - zone_low) / zone_low * 100,
            'type': buy_signal['type']
        }
    
    def _in_buy_zone(self, price: float, buy_zone: Dict[str, Any]) -> bool:
        """检查价格是否在买点区域内"""
        if not buy_zone:
            return False
        return buy_zone['low'] <= price <= buy_zone['high']

# ====================== 智能调度器 ======================
class IntelligentScheduler:
    """智能调度器 - 根据交易状态动态调整执行频率"""
    
    def __init__(self, config: Config, state_manager: TradingStateManager, 
                 analyzer: MultiTimeframeAnalyzer, exchange: ExchangeManager,
                 ai_analyzer: AIAnalyzer):
        self.config = config
        self.state_manager = state_manager
        self.analyzer = analyzer
        self.exchange = exchange
        self.ai_analyzer = ai_analyzer
        self.last_execution_time = {}
        
    def setup_schedule(self):
        """根据当前状态设置调度"""
        schedule.clear()
        
        mode = self.state_manager.current_mode
        
        if mode == TradingMode.MONITORING:
            # 监控模式：每30分钟检查一次30分钟信号
            schedule.every(30).minutes.do(self.execute_monitoring_task)
            print(f"⏰ 设置调度：每30分钟检查一次{self.config.SIGNAL_TIMEFRAME}信号")
            
        elif mode == TradingMode.ENTRY_SCOPING:
            # 入场侦察模式：每5分钟检查一次5分钟入场点
            schedule.every(5).minutes.do(self.execute_entry_scoping_task)
            print(f"⏰ 设置调度：每5分钟检查一次{self.config.ENTRY_TIMEFRAME}入场点")
            
            # 同时每30分钟检查一次信号是否依然有效
            schedule.every(30).minutes.do(self.check_signal_validity)
            
        elif mode == TradingMode.IN_POSITION:
            # 持仓模式：每5分钟检查一次离场信号
            schedule.every(5).minutes.do(self.execute_exit_check_task)
            print(f"⏰ 设置调度：每5分钟检查一次离场信号")
            
        elif mode == TradingMode.COOLDOWN:
            # 冷却模式：不执行交易任务
            print(f"⏰ 冷却模式，暂停交易任务")
            # 30分钟后返回监控模式
            schedule.every(30).minutes.do(self.return_to_monitoring)
    
    def execute_monitoring_task(self):
        """执行监控任务"""
        print(f"\n🔍 执行监控任务 - {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. 分析30分钟信号
        signal_result = self.analyzer.analyze_30m_signal()
        
        # 2. 如果有买点信号，切换到入场侦察模式
        if signal_result['signal'] == 'BUY' and signal_result['confidence'] > 0.6:
            print(f"✅ 发现{self.config.SIGNAL_TIMEFRAME}买点信号!")
            print(f"   理由: {signal_result['reason']}")
            print(f"   信心: {signal_result['confidence']:.1%}")
            print(f"   买点区域: {signal_result['buy_zone']['low']:.2f} - {signal_result['buy_zone']['high']:.2f}")
            
            # 更新状态管理器
            self.state_manager.signal_30m = signal_result
            self.state_manager.change_mode(TradingMode.ENTRY_SCOPING)
            
            # 重新设置调度
            self.setup_schedule()
        else:
            print(f"📊 {self.config.SIGNAL_TIMEFRAME}信号: {signal_result['signal']}")
            print(f"   理由: {signal_result.get('reason', '无明确信号')}")
            print(f"   当前价格: ${signal_result.get('current_price', 0):.2f}")
            print(f"   当前趋势: {signal_result.get('trend', 'unknown')}")
    
    def execute_entry_scoping_task(self):
        """执行入场侦察任务"""
        print(f"\n🎯 执行入场侦察任务 - {datetime.now().strftime('%H:%M:%S')}")
        
        # 检查信号有效期和超时
        if (self.state_manager.check_signal_expiry() or 
            self.state_manager.check_entry_timeout()):
            self.setup_schedule()
            return
        
        # 分析5分钟入场点
        entry_result = self.analyzer.analyze_5m_entry(
            self.state_manager.signal_30m['buy_zone']
        )
        
        current_price = self.exchange.get_current_price()
        
        if entry_result['entry_signal'] == 'BUY':
            print(f"🎯 发现{self.config.ENTRY_TIMEFRAME}精确入场点!")
            print(f"   入场价格: {entry_result['entry_price']:.2f}")
            print(f"   信号信心: {entry_result['confidence']:.1%}")
            print(f"   止损: {entry_result['stop_loss']:.2f}")
            print(f"   止盈: {entry_result['take_profit']:.2f}")
            
            # 使用AI进行最终确认
            if self.config.ENABLE_AI_ANALYSIS:
                # 获取5分钟K线数据供AI分析
                df_5m = self.exchange.get_ohlcv(self.config.ENTRY_TIMEFRAME, limit=30)
                if not df_5m.empty:
                    # 构建简单的缠论数据供AI分析
                    chan_data = {
                        'trend': 'potential_buy',
                        'current_price': current_price,
                        'fractals': {'top': [], 'bottom': [{'price': entry_result['entry_price']}]}
                    }
                    
                    ai_signal = self.ai_analyzer.analyze_with_chan_theory(df_5m, chan_data)
                    
                    if ai_signal['signal'] == 'BUY' and ai_signal['confidence'] > 0.6:
                        print(f"🤖 AI确认: {ai_signal['reason']}")
                        print(f"   AI信心: {ai_signal['confidence']:.1%}")
                        
                        # 合并AI分析结果
                        entry_result.update({
                            'ai_confidence': ai_signal['confidence'],
                            'ai_reason': ai_signal['reason'],
                            'ai_chan_pattern': ai_signal.get('chan_pattern', 'unknown')
                        })
                    else:
                        print(f"🤖 AI建议谨慎: {ai_signal['reason']}")
                        # 如果AI信心不足，可以降低入场信心
                        entry_result['confidence'] *= 0.7
            
            # 执行入场交易
            if entry_result['confidence'] > 0.6:  # 信心阈值
                self.execute_entry_trade(entry_result)
            else:
                print("⚠️ 信心不足，放弃本次入场机会")
                self.state_manager.increment_entry_attempts()
                
        elif entry_result['entry_signal'] == 'PREPARE_BUY':
            print(f"⚠️ 准备入场: {entry_result['reason']}")
            print(f"   当前价格: ${current_price:.2f}")
            print(f"   买点区域: {self.state_manager.signal_30m['buy_zone']['low']:.2f} - "
                  f"{self.state_manager.signal_30m['buy_zone']['high']:.2f}")
            self.state_manager.increment_entry_attempts()
            
        else:
            print(f"⌛ {entry_result['reason']}")
            print(f"   当前价格: ${current_price:.2f}")
            print(f"   买点区域: {self.state_manager.signal_30m['buy_zone']['low']:.2f} - "
                  f"{self.state_manager.signal_30m['buy_zone']['high']:.2f}")
    
    def execute_entry_trade(self, entry_result: Dict[str, Any]):
        """执行入场交易"""
        print(f"\n💰 执行入场交易...")
        
        # 计算交易量
        trade_amount = self.config.TRADE_AMOUNT
        
        if self.config.TEST_MODE:
            # 模拟下单
            order = self.exchange.place_test_order('buy', trade_amount)
        else:
            # 真实下单
            order = self.exchange.place_real_order('buy', trade_amount)
        
        # 更新持仓信息
        self.state_manager.position = {
            'entry_price': entry_result['entry_price'],
            'stop_loss': entry_result['stop_loss'],
            'take_profit': entry_result['take_profit'],
            'side': 'long',
            'entry_time': datetime.now(),
            'amount': trade_amount,
            'order_id': order.get('id', 'N/A')
        }
        
        # 记录交易
        self.state_manager.record_trade(
            entry_result, trade_amount, 
            entry_result['entry_price'], 'buy'
        )
        
        # 切换到持仓模式
        self.state_manager.change_mode(TradingMode.IN_POSITION)
        self.setup_schedule()
        
        print(f"✅ 入场交易完成!")
        print(f"   入场价格: ${entry_result['entry_price']:.2f}")
        print(f"   交易数量: {trade_amount:.4f} BTC")
        print(f"   订单ID: {order.get('id', 'N/A')}")
    
    def execute_exit_check_task(self):
        """执行离场检查任务"""
        print(f"\n🚪 执行离场检查 - {datetime.now().strftime('%H:%M:%S')}")
        
        if not self.state_manager.position:
            print("❌ 无持仓信息，返回监控模式")
            self.state_manager.change_mode(TradingMode.MONITORING)
            self.setup_schedule()
            return
        
        # 分析离场信号
        exit_result = self.analyzer.analyze_exit_signal(self.state_manager.position)
        
        current_price = self.exchange.get_current_price()
        position = self.state_manager.position
        unrealized_pnl = 0
        
        if position['side'] == 'long':
            unrealized_pnl = (current_price - position['entry_price']) * position['amount']
        else:
            unrealized_pnl = (position['entry_price'] - current_price) * position['amount']
        
        print(f"📊 持仓状态:")
        print(f"   方向: {position['side']}")
        print(f"   入场价: ${position['entry_price']:.2f}")
        print(f"   当前价: ${current_price:.2f}")
        print(f"   浮动盈亏: ${unrealized_pnl:.2f}")
        
        if exit_result['exit_signal'] in ['BUY', 'SELL']:
            print(f"🚪 发现离场信号: {exit_result['reason']}")
            print(f"   信号信心: {exit_result['confidence']:.1%}")
            
            # 执行离场
            self.execute_exit_trade(exit_result)
        else:
            print(f"📊 {exit_result['reason']}")
    
    def execute_exit_trade(self, exit_result: Dict[str, Any]):
        """执行离场交易"""
        print(f"\n💰 执行离场交易...")
        
        position = self.state_manager.position
        trade_amount = position['amount']
        exit_side = 'sell' if position['side'] == 'long' else 'buy'
        current_price = self.exchange.get_current_price()
        
        if self.config.TEST_MODE:
            # 模拟平仓
            order = self.exchange.place_test_order(exit_side, trade_amount)
        else:
            # 真实平仓
            order = self.exchange.place_real_order(exit_side, trade_amount)
        
        # 计算盈亏
        if position['side'] == 'long':
            pnl = (current_price - position['entry_price']) * trade_amount
            pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
        else:
            pnl = (position['entry_price'] - current_price) * trade_amount
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
        
        # 记录交易
        self.state_manager.record_trade(
            exit_result, trade_amount, current_price, exit_side
        )
        
        print(f"✅ 离场交易完成!")
        print(f"   离场价格: ${current_price:.2f}")
        print(f"   盈亏: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        print(f"   理由: {exit_result['reason']}")
        
        # 切换到冷却模式
        self.state_manager.change_mode(TradingMode.COOLDOWN)
        self.state_manager.position = None
        self.setup_schedule()
    
    def check_signal_validity(self):
        """检查30分钟信号是否依然有效"""
        if self.state_manager.current_mode == TradingMode.ENTRY_SCOPING:
            current_signal = self.analyzer.analyze_30m_signal()
            
            if current_signal['signal'] != 'BUY' or current_signal['confidence'] < 0.4:
                print("❌ 30分钟买点信号已失效，返回监控模式")
                self.state_manager.change_mode(TradingMode.MONITORING)
                self.setup_schedule()
    
    def return_to_monitoring(self):
        """从冷却模式返回监控模式"""
        if self.state_manager.current_mode == TradingMode.COOLDOWN:
            print("🔄 冷却结束，返回监控模式")
            self.state_manager.change_mode(TradingMode.MONITORING)
            self.setup_schedule()

# ====================== 主交易机器人 ======================
class ChanTradingBot:
    """缠论交易机器人主类"""
    
    def __init__(self, config: Config):
        self.config = config
        
        print("🚀 初始化缠论交易机器人...")
        
        # 初始化各模块
        self.exchange = ExchangeManager(config)
        self.state_manager = TradingStateManager(config)
        self.analyzer = MultiTimeframeAnalyzer(config, self.exchange)
        self.ai_analyzer = AIAnalyzer(config)
        self.scheduler = IntelligentScheduler(
            config, self.state_manager, self.analyzer, 
            self.exchange, self.ai_analyzer
        )
        
        print("✅ 交易机器人初始化完成!")
    
    def display_status(self):
        """显示机器人状态"""
        print("\n" + "="*60)
        print(f"🤖 BTC缠论交易机器人状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # 显示模式
        mode = self.state_manager.current_mode
        mode_icons = {
            TradingMode.MONITORING: "🔍",
            TradingMode.ENTRY_SCOPING: "🎯", 
            TradingMode.IN_POSITION: "💰",
            TradingMode.COOLDOWN: "🛌"
        }
        
        print(f"交易模式: {mode_icons[mode]} {mode.value}")
        print(f"持续时间: {self.state_manager.get_mode_duration():.1f}分钟")
        
        # 显示信号信息
        if mode == TradingMode.ENTRY_SCOPING and self.state_manager.signal_30m:
            signal = self.state_manager.signal_30m
            print(f"\n📈 30分钟信号:")
            print(f"  类型: {signal['signal']}")
            print(f"  理由: {signal['reason']}")
            print(f"  信心: {signal['confidence']:.1%}")
            print(f"  买点区域: {signal['buy_zone']['low']:.2f} - {signal['buy_zone']['high']:.2f}")
            print(f"  入场尝试: {self.state_manager.entry_attempts}次")
        
        # 显示持仓信息
        if mode == TradingMode.IN_POSITION and self.state_manager.position:
            position = self.state_manager.position
            current_price = self.exchange.get_current_price()
            
            print(f"\n💰 持仓信息:")
            print(f"  方向: {position['side']}")
            print(f"  入场价: ${position['entry_price']:.2f}")
            print(f"  当前价: ${current_price:.2f}")
            print(f"  数量: {position['amount']:.4f} BTC")
            
            # 计算盈亏
            if position['side'] == 'long':
                pnl = (current_price - position['entry_price']) * position['amount']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
            else:
                pnl = (position['entry_price'] - current_price) * position['amount']
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price'] * 100
            
            print(f"  浮动盈亏: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            print(f"  止损: ${position['stop_loss']:.2f}")
            print(f"  止盈: ${position['take_profit']:.2f}")
        
        # 显示账户信息
        try:
            balance = self.exchange.get_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            current_price = self.exchange.get_current_price()
            
            print(f"\n💼 账户信息:")
            print(f"  USDT余额: ${usdt_balance:.2f}")
            print(f"  当前价格: ${current_price:,.2f}")
        except:
            pass
        
        # 显示交易历史
        if self.state_manager.trade_history:
            recent_trades = self.state_manager.trade_history[-3:]  # 最近3笔
            print(f"\n📋 最近交易:")
            for trade in recent_trades:
                time_str = trade['timestamp'].strftime('%H:%M')
                print(f"  {time_str} {trade['side']} {trade['amount']:.4f} @ ${trade['price']:.2f}")
        
        print("="*60)
    
    def start(self):
        """启动交易机器人"""
        print("\n" + "="*60)
        print("🚀 启动缠论交易机器人")
        print("="*60)
        print(f"交易对: {self.config.SYMBOL}")
        print(f"模式: {'模拟' if self.config.TEST_MODE else '实盘'}")
        print(f"信号框架: {self.config.SIGNAL_TIMEFRAME}")
        print(f"入场框架: {self.config.ENTRY_TIMEFRAME}")
        print(f"AI分析: {'启用' if self.config.ENABLE_AI_ANALYSIS else '禁用'}")
        print("="*60)
        
        # 设置初始调度
        self.scheduler.setup_schedule()
        
        # 立即执行一次监控任务
        print("\n🔍 执行初始监控...")
        self.scheduler.execute_monitoring_task()
        
        # 显示初始状态
        self.display_status()
        
        print("\n🔄 机器人运行中...")
        print("  按 Ctrl+C 停止程序")
        print("  按 's' 查看状态")
        print("-"*60)
        
        # 主循环
        last_status_time = time.time()
        
        try:
            while True:
                # 执行定时任务
                schedule.run_pending()
                
                # 每分钟显示一次状态
                current_time = time.time()
                if current_time - last_status_time > 60:
                    self.display_status()
                    last_status_time = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n👋 正在停止交易机器人...")
            schedule.clear()
            print("✅ 交易机器人已安全停止")

# ====================== 主函数 ======================
def main():
    """主函数"""
    # 欢迎信息
    print("="*60)
    print("🤖 BTC/USDT 缠论+AI 自动交易机器人")
    print("版本: 1.0.0 | 作者: AI交易助手")
    print("="*60)
    
    # 创建配置
    config = Config()
    
    # 检查环境变量
    if not config.TEST_MODE:
        if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_SECRET'):
            print("❌ 实盘模式需要设置BINANCE_API_KEY和BINANCE_SECRET环境变量")
            print("请创建.env文件或在环境中设置这些变量")
            return
    
    if config.ENABLE_AI_ANALYSIS and not os.getenv('DEEPSEEK_API_KEY'):
        print("⚠️  DEEPSEEK_API_KEY未设置，AI分析功能将禁用")
        config.ENABLE_AI_ANALYSIS = False
    
    # 确认启动
    print(f"配置确认:")
    print(f"  交易模式: {'模拟' if config.TEST_MODE else '实盘'}")
    print(f"  交易对: {config.SYMBOL}")
    print(f"  杠杆: {config.LEVERAGE}x")
    print(f"  时间框架: {config.SIGNAL_TIMEFRAME} -> {config.ENTRY_TIMEFRAME}")
    print(f"  AI分析: {'启用' if config.ENABLE_AI_ANALYSIS else '禁用'}")
    
    if not config.TEST_MODE:
        confirm = input("\n🔴 警告: 实盘模式会真实下单! 确认继续? (y/N): ")
        if confirm.lower() != 'y':
            print("取消启动")
            return
    
    # 创建并启动机器人
    bot = ChanTradingBot(config)
    
    try:
        bot.start()
    except Exception as e:
        print(f"\n❌ 机器人运行出错: {e}")
        import traceback
        traceback.print_exc()
        schedule.clear()

if __name__ == "__main__":
    main()