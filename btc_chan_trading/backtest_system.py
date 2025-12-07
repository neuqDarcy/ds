#!/usr/bin/env python3
"""
可配置的缠论交易回测系统
支持：命令行参数、配置文件、交互式输入
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ====================== 配置管理器 ======================
class ConfigManager:
    """配置管理器 - 支持多种配置方式"""
    
    DEFAULT_CONFIG = {
        # 回测参数
        'symbol': 'BTC/USDT',
        'timeframe': '30m',
        'entry_timeframe': '5m',
        'start_date': '2024-01-01 00:00:00',
        'end_date': '2024-06-01 00:00:00',
        
        # 交易参数
        'initial_capital': 10000.0,
        'trade_amount': 0.001,
        'leverage': 1,
        
        # 交易成本
        'fee_rate': 0.001,
        'slippage': 0.0005,
        
        # 风险控制
        'stop_loss_pct': 1.5,
        'take_profit_pct': 3.0,
        'max_position_size': 0.01,
        
        # 缠论参数
        'fractal_period': 5,
        'central_pivot_lookback': 20,
        'confidence_threshold': 0.6,
        
        # 回测模式
        'enable_entry_scoping': True,
        'enable_ai_analysis': False,
        'verbose': False,
        'save_results': True,
        
        # 输出选项
        'plot_results': True,
        'save_plots': True,
        'report_format': 'console'  # console, json, html
    }
    
    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        
    def load_from_file(self, config_file: str):
        """从JSON文件加载配置"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                print(f"✅ 从文件加载配置: {config_file}")
                return True
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
        return False
    
    def load_from_env(self):
        """从环境变量加载配置"""
        env_mapping = {
            'BT_SYMBOL': 'symbol',
            'BT_TIMEFRAME': 'timeframe',
            'BT_START_DATE': 'start_date',
            'BT_END_DATE': 'end_date',
            'BT_INITIAL_CAPITAL': 'initial_capital',
            'BT_STOP_LOSS': 'stop_loss_pct',
            'BT_TAKE_PROFIT': 'take_profit_pct'
        }
        
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                try:
                    # 尝试转换为适当类型
                    if '.' in value:
                        self.config[config_key] = float(value)
                    else:
                        self.config[config_key] = value
                    print(f"   从环境变量加载: {env_var} -> {config_key}")
                except:
                    self.config[config_key] = value
        
        return True
    
    def parse_args(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description='缠论交易策略回测系统',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  %(prog)s --start 2024-01-01 --end 2024-06-01 --capital 50000
  %(prog)s --config backtest_config.json
  %(prog)s --symbol ETH/USDT --timeframe 1h --verbose
  
环境变量示例:
  export BT_SYMBOL="BTC/USDT"
  export BT_INITIAL_CAPITAL="10000"
  export BT_STOP_LOSS="2.0"
  """
        )
        
        # 回测参数组
        backtest_group = parser.add_argument_group('回测参数')
        backtest_group.add_argument('--config', type=str, help='配置文件路径')
        backtest_group.add_argument('--symbol', type=str, help='交易对 (默认: BTC/USDT)')
        backtest_group.add_argument('--timeframe', type=str, help='时间框架 (默认: 30m)')
        backtest_group.add_argument('--start', '--start-date', type=str, 
                                   help='开始日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)')
        backtest_group.add_argument('--end', '--end-date', type=str,
                                   help='结束日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)')
        backtest_group.add_argument('--days', type=int, 
                                   help='回测天数 (从结束日期往前推)')
        
        # 资金参数组
        capital_group = parser.add_argument_group('资金参数')
        capital_group.add_argument('--capital', '--initial-capital', type=float,
                                  help='初始资金 (默认: 10000)')
        capital_group.add_argument('--amount', '--trade-amount', type=float,
                                  help='每次交易数量 (默认: 0.001)')
        capital_group.add_argument('--leverage', type=float, help='杠杆 (默认: 1)')
        
        # 风险参数组
        risk_group = parser.add_argument_group('风险参数')
        risk_group.add_argument('--stop-loss', '--sl', type=float,
                               help='止损百分比 (默认: 1.5)')
        risk_group.add_argument('--take-profit', '--tp', type=float,
                               help='止盈百分比 (默认: 3.0)')
        risk_group.add_argument('--fee', '--fee-rate', type=float,
                               help='手续费率 (默认: 0.001)')
        risk_group.add_argument('--slippage', type=float, help='滑点 (默认: 0.0005)')
        
        # 策略参数组
        strategy_group = parser.add_argument_group('策略参数')
        strategy_group.add_argument('--fractal-period', type=int,
                                   help='分型周期 (默认: 5)')
        strategy_group.add_argument('--confidence', type=float,
                                   help='信号信心阈值 (默认: 0.6)')
        strategy_group.add_argument('--no-entry-scoping', action='store_true',
                                   help='禁用精确入场')
        strategy_group.add_argument('--enable-ai', action='store_true',
                                   help='启用AI分析')
        
        # 输出参数组
        output_group = parser.add_argument_group('输出参数')
        output_group.add_argument('--verbose', '-v', action='store_true',
                                 help='详细输出模式')
        output_group.add_argument('--no-plot', action='store_true',
                                 help='不显示图表')
        output_group.add_argument('--no-save', action='store_true',
                                 help='不保存结果')
        output_group.add_argument('--output', '-o', type=str,
                                 help='输出目录 (默认: reports)')
        output_group.add_argument('--format', type=str, choices=['console', 'json', 'html'],
                                 help='报告格式 (默认: console)')
        
        args = parser.parse_args()
        
        # 应用命令行参数
        if args.config:
            self.load_from_file(args.config)
        
        # 更新配置
        arg_mapping = {
            'symbol': args.symbol,
            'timeframe': args.timeframe,
            'start_date': args.start,
            'end_date': args.end,
            'initial_capital': args.capital,
            'trade_amount': args.amount,
            'leverage': args.leverage,
            'stop_loss_pct': args.stop_loss,
            'take_profit_pct': args.take_profit,
            'fee_rate': args.fee,
            'slippage': args.slippage,
            'fractal_period': args.fractal_period,
            'confidence_threshold': args.confidence,
            'verbose': args.verbose,
            'plot_results': not args.no_plot,
            'save_results': not args.no_save,
            'report_format': args.format,
            'enable_entry_scoping': not args.no_entry_scoping,
            'enable_ai_analysis': args.enable_ai
        }
        
        for key, value in arg_mapping.items():
            if value is not None:
                self.config[key] = value
        
        # 处理日期参数
        if args.days and args.end:
            end_date = datetime.strptime(args.end, '%Y-%m-%d') if len(args.end) == 10 else datetime.strptime(args.end, '%Y-%m-%d %H:%M:%S')
            start_date = end_date - timedelta(days=args.days)
            self.config['start_date'] = start_date.strftime('%Y-%m-%d %H:%M:%S')
        elif args.days and not args.end:
            # 如果没有结束日期，使用当前时间
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            self.config['end_date'] = end_date.strftime('%Y-%m-%d %H:%M:%S')
            self.config['start_date'] = start_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # 加载环境变量（覆盖命令行参数）
        self.load_from_env()
        
        return args
    
    def interactive_setup(self):
        """交互式配置向导"""
        print("\n" + "="*60)
        print("🤖 缠论回测系统 - 交互式配置向导")
        print("="*60)
        
        print("\n📊 回测参数配置:")
        
        # 时间范围
        print("\n1. 时间范围设置:")
        use_default_dates = input("   使用默认时间范围(2024年1-6月)? (y/n): ").lower() == 'y'
        
        if not use_default_dates:
            start_date = input("   开始日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS): ")
            end_date = input("   结束日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS): ")
            
            if start_date:
                self.config['start_date'] = start_date
            if end_date:
                self.config['end_date'] = end_date
        
        # 交易参数
        print("\n2. 交易参数设置:")
        self.config['initial_capital'] = float(input(f"   初始资金 (默认: {self.config['initial_capital']}): ") or self.config['initial_capital'])
        self.config['trade_amount'] = float(input(f"   每次交易数量 (默认: {self.config['trade_amount']}): ") or self.config['trade_amount'])
        
        # 风险参数
        print("\n3. 风险参数设置:")
        self.config['stop_loss_pct'] = float(input(f"   止损百分比 (默认: {self.config['stop_loss_pct']}%): ") or self.config['stop_loss_pct'])
        self.config['take_profit_pct'] = float(input(f"   止盈百分比 (默认: {self.config['take_profit_pct']}%): ") or self.config['take_profit_pct'])
        
        # 缠论参数
        print("\n4. 缠论参数设置:")
        self.config['fractal_period'] = int(input(f"   分型周期 (默认: {self.config['fractal_period']}): ") or self.config['fractal_period'])
        self.config['confidence_threshold'] = float(input(f"   信号信心阈值 (默认: {self.config['confidence_threshold']}): ") or self.config['confidence_threshold'])
        
        # 模式选择
        print("\n5. 模式选择:")
        self.config['enable_entry_scoping'] = input("   启用5分钟精确入场? (y/n, 默认: y): ").lower() != 'n'
        self.config['verbose'] = input("   显示详细日志? (y/n, 默认: n): ").lower() == 'y'
        
        # 输出选项
        print("\n6. 输出选项:")
        self.config['plot_results'] = input("   生成图表? (y/n, 默认: y): ").lower() != 'n'
        self.config['save_results'] = input("   保存结果文件? (y/n, 默认: y): ").lower() != 'n'
        
        print("\n✅ 配置完成!")
        return self.config
    
    def display_config(self):
        """显示当前配置"""
        print("\n" + "="*60)
        print("📋 当前回测配置")
        print("="*60)
        
        groups = {
            '回测参数': ['symbol', 'timeframe', 'start_date', 'end_date'],
            '资金参数': ['initial_capital', 'trade_amount', 'leverage'],
            '风险参数': ['stop_loss_pct', 'take_profit_pct', 'fee_rate', 'slippage'],
            '缠论参数': ['fractal_period', 'central_pivot_lookback', 'confidence_threshold'],
            '模式参数': ['enable_entry_scoping', 'enable_ai_analysis', 'verbose']
        }
        
        for group_name, keys in groups.items():
            print(f"\n{group_name}:")
            for key in keys:
                if key in self.config:
                    value = self.config[key]
                    if isinstance(value, float):
                        if key in ['stop_loss_pct', 'take_profit_pct']:
                            print(f"  {key.replace('_', ' ').title()}: {value}%")
                        elif key in ['fee_rate', 'slippage']:
                            print(f"  {key.replace('_', ' ').title()}: {value*100:.2f}%")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("="*60)
    
    def save_config(self, filename: str = None):
        """保存配置到文件"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_config_{timestamp}.json"
        
        os.makedirs('configs', exist_ok=True)
        filepath = f"configs/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        print(f"✅ 配置已保存到: {filepath}")
        return filepath
    
    def get_config(self):
        """获取配置字典"""
        return self.config.copy()

# ====================== 简化版回测引擎 ======================
class SimpleBacktestEngine:
    """简化版回测引擎（仅用于演示配置）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 回测状态
        self.capital = config['initial_capital']
        self.btc_amount = 0.0
        self.trades = []
        self.equity_history = []
        
        print(f"🚀 初始化回测引擎")
        print(f"   交易对: {config['symbol']}")
        print(f"   时间框架: {config['timeframe']}")
        print(f"   时间范围: {config['start_date']} 到 {config['end_date']}")
    
    def run_simulation(self):
        """运行模拟回测（简化版本）"""
        print(f"\n🔍 运行回测分析...")
        
        # 模拟数据
        dates = pd.date_range(
            start=self.config['start_date'],
            end=self.config['end_date'],
            periods=100
        )
        
        # 生成模拟价格数据
        np.random.seed(42)
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.02)  # 2%日波动
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # 模拟回测逻辑
        position = None
        trade_count = 0
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # 简单交易策略：随机买入卖出
            if not position and np.random.random() < 0.05:  # 5%概率买入
                trade_amount = self.config['trade_amount']
                trade_value = trade_amount * price
                fee = trade_value * self.config['fee_rate']
                
                if trade_value + fee <= self.capital:
                    self.capital -= (trade_value + fee)
                    self.btc_amount += trade_amount
                    position = {
                        'entry_price': price,
                        'entry_time': date,
                        'amount': trade_amount,
                        'stop_loss': price * (1 - self.config['stop_loss_pct'] / 100)
                    }
                    
                    self.trades.append({
                        'timestamp': date,
                        'type': 'BUY',
                        'price': price,
                        'amount': trade_amount,
                        'fee': fee
                    })
                    trade_count += 1
                    
                    if self.config['verbose']:
                        print(f"  {date.strftime('%Y-%m-%d')}: 买入 {trade_amount:.4f} BTC @ ${price:.2f}")
            
            # 检查止损
            elif position and price <= position['stop_loss']:
                trade_amount = position['amount']
                trade_value = trade_amount * price
                fee = trade_value * self.config['fee_rate']
                
                self.capital += (trade_value - fee)
                self.btc_amount -= trade_amount
                
                pnl = (price - position['entry_price']) * trade_amount
                pnl_pct = (price - position['entry_price']) / position['entry_price'] * 100
                
                self.trades.append({
                    'timestamp': date,
                    'type': 'SELL',
                    'price': price,
                    'amount': trade_amount,
                    'fee': fee,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'reason': '止损'
                })
                
                position = None
                trade_count += 1
                
                if self.config['verbose']:
                    print(f"  {date.strftime('%Y-%m-%d')}: 止损卖出 {trade_amount:.4f} BTC @ ${price:.2f}, 盈亏: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            # 记录权益
            equity = self.capital + (self.btc_amount * price)
            self.equity_history.append({
                'timestamp': date,
                'equity': equity,
                'price': price,
                'capital': self.capital,
                'btc_amount': self.btc_amount
            })
        
        # 回测结束，平仓
        if self.btc_amount > 0 and prices:
            price = prices[-1]
            trade_value = self.btc_amount * price
            fee = trade_value * self.config['fee_rate']
            
            pnl = (price - position['entry_price']) * self.btc_amount if position else 0
            
            self.capital += (trade_value - fee)
            
            self.trades.append({
                'timestamp': dates[-1],
                'type': 'SELL',
                'price': price,
                'amount': self.btc_amount,
                'fee': fee,
                'pnl': pnl,
                'reason': '回测结束平仓'
            })
            
            self.btc_amount = 0
        
        print(f"\n✅ 回测完成!")
        print(f"   交易次数: {len(self.trades)}")
        print(f"   最终权益: ${self.equity_history[-1]['equity']:.2f}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        if not self.equity_history:
            return {}
        
        initial_equity = self.config['initial_capital']
        final_equity = self.equity_history[-1]['equity']
        total_return_pct = (final_equity - initial_equity) / initial_equity * 100
        
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
        
        report = {
            'summary': {
                'initial_capital': initial_equity,
                'final_equity': final_equity,
                'total_return_pct': total_return_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'total_trades': len(self.trades),
                'winning_trades': winning_trades,
                'win_rate': win_rate
            },
            'config': self.config,
            'trades': self.trades[-10:] if len(self.trades) > 10 else self.trades,
            'equity_history': self.equity_history
        }
        
        return report
    
    def plot_results(self, report: Dict[str, Any]):
        """绘制结果图表"""
        if not report or not self.equity_history:
            print("❌ 没有回测数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 权益曲线
        times = [h['timestamp'] for h in self.equity_history]
        equities = [h['equity'] for h in self.equity_history]
        prices = [h['price'] for h in self.equity_history]
        
        axes[0].plot(times, equities, 'b-', linewidth=2, label='总权益')
        axes[0].axhline(y=self.config['initial_capital'], color='r', linestyle='--', alpha=0.5, label='初始资金')
        axes[0].set_title(f'回测结果 - 总收益: {report["summary"]["total_return_pct"]:.2f}%', fontsize=14)
        axes[0].set_ylabel('权益 (USDT)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 价格曲线
        axes[1].plot(times, prices, 'g-', alpha=0.7, label='价格')
        axes[1].set_xlabel('时间')
        axes[1].set_ylabel('价格 (USDT)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 标记交易点
        buy_times = [t['timestamp'] for t in self.trades if t['type'] == 'BUY']
        buy_prices = [t['price'] for t in self.trades if t['type'] == 'BUY']
        sell_times = [t['timestamp'] for t in self.trades if t['type'] == 'SELL']
        sell_prices = [t['price'] for t in self.trades if t['type'] == 'SELL']
        
        if buy_times:
            axes[1].scatter(buy_times, buy_prices, color='green', s=100, marker='^', label='买入', zorder=5)
        if sell_times:
            axes[1].scatter(sell_times, sell_prices, color='red', s=100, marker='v', label='卖出', zorder=5)
        
        plt.tight_layout()
        
        if self.config['save_results']:
            os.makedirs('reports', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'reports/backtest_chart_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 图表已保存: {filename}")
        
        if self.config['plot_results']:
            plt.show()
        else:
            plt.close()
    
    def print_report(self, report: Dict[str, Any]):
        """打印回测报告"""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("📊 回测报告摘要")
        print("="*60)
        
        print(f"💰 资金统计:")
        print(f"   初始资金: ${summary['initial_capital']:,.2f}")
        print(f"   最终权益: ${summary['final_equity']:,.2f}")
        print(f"   总收益率: {summary['total_return_pct']:+.2f}%")
        
        print(f"\n📈 风险指标:")
        print(f"   最大回撤: {summary['max_drawdown_pct']:.2f}%")
        
        print(f"\n📊 交易统计:")
        print(f"   总交易次数: {summary['total_trades']}")
        print(f"   盈利交易: {summary['winning_trades']}")
        print(f"   胜率: {summary['win_rate']:.1f}%")
        
        if report['trades']:
            print(f"\n📋 最近交易:")
            for trade in report['trades']:
                time_str = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
                if trade['type'] == 'BUY':
                    print(f"   {time_str} | 买入 {trade['amount']:.4f} BTC @ ${trade['price']:.2f}")
                else:
                    pnl_str = f"(${trade['pnl']:+.2f})" if 'pnl' in trade else ""
                    reason_str = f" [{trade.get('reason', '')}]" if 'reason' in trade else ""
                    print(f"   {time_str} | 卖出 {trade['amount']:.4f} BTC @ ${trade['price']:.2f} {pnl_str}{reason_str}")
        
        print("="*60)

# ====================== 批处理回测 ======================
class BatchBacktest:
    """批处理回测 - 测试多组参数"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.results = []
    
    def run_batch(self, param_sets: List[Dict[str, Any]]):
        """运行批处理回测"""
        print(f"\n📊 开始批处理回测 ({len(param_sets)} 组参数)")
        
        for i, params in enumerate(param_sets, 1):
            print(f"\n🔧 测试参数组 {i}/{len(param_sets)}:")
            for key, value in params.items():
                print(f"   {key}: {value}")
            
            # 更新配置
            config = self.config_manager.get_config()
            config.update(params)
            
            # 运行回测
            engine = SimpleBacktestEngine(config)
            report = engine.run_simulation()
            
            if report:
                self.results.append({
                    'params': params,
                    'report': report['summary']
                })
        
        return self.analyze_batch_results()
    
    def analyze_batch_results(self):
        """分析批处理结果"""
        if not self.results:
            return None
        
        print("\n" + "="*60)
        print("📈 批处理回测结果分析")
        print("="*60)
        
        # 转换为DataFrame便于分析
        results_df = pd.DataFrame([
            {
                **r['params'],
                'total_return': r['report']['total_return_pct'],
                'max_drawdown': r['report']['max_drawdown_pct'],
                'win_rate': r['report']['win_rate'],
                'trade_count': r['report']['total_trades']
            }
            for r in self.results
        ])
        
        # 找出最佳参数
        if not results_df.empty:
            # 按收益率排序
            best_return = results_df.loc[results_df['total_return'].idxmax()]
            # 按风险调整收益排序
            results_df['risk_adjusted'] = results_df['total_return'] / abs(results_df['max_drawdown'])
            best_risk_adj = results_df.loc[results_df['risk_adjusted'].idxmax()]
            # 按胜率排序
            best_win_rate = results_df.loc[results_df['win_rate'].idxmax()]
            
            print(f"\n🏆 最佳收益率参数:")
            self._print_best_params(best_return)
            
            print(f"\n⚖️  最佳风险调整收益:")
            self._print_best_params(best_risk_adj)
            
            print(f"\n🎯 最高胜率参数:")
            self._print_best_params(best_win_rate)
            
            # 保存结果
            if self.config_manager.config['save_results']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_file = f'reports/batch_results_{timestamp}.csv'
                results_df.to_csv(results_file, index=False)
                print(f"\n💾 批处理结果已保存: {results_file}")
            
            return results_df
        
        return None
    
    def _print_best_params(self, best_row):
        """打印最佳参数"""
        print(f"   收益率: {best_row['total_return']:.2f}%")
        print(f"   最大回撤: {best_row['max_drawdown']:.2f}%")
        print(f"   胜率: {best_row['win_rate']:.1f}%")
        print(f"   交易次数: {best_row['trade_count']}")
        
        param_keys = ['stop_loss_pct', 'take_profit_pct', 'fractal_period', 'confidence_threshold']
        for key in param_keys:
            if key in best_row:
                print(f"   {key.replace('_', ' ').title()}: {best_row[key]}")

# ====================== 主函数 ======================
def main():
    """主函数"""
    print("="*60)
    print("🎛️  可配置缠论回测系统")
    print("="*60)
    
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 解析命令行参数
    args = config_manager.parse_args()
    
    # 如果没有指定配置，显示配置向导
    if len(sys.argv) == 1:  # 没有命令行参数
        print("\n🤔 没有指定配置参数，请选择配置方式:")
        print("1. 交互式配置向导")
        print("2. 使用默认配置")
        print("3. 运行批处理回测")
        
        choice = input("\n请输入选项 (1-3): ").strip()
        
        if choice == '1':
            config = config_manager.interactive_setup()
        elif choice == '3':
            # 批处理回测
            run_batch_backtest(config_manager)
            return
        else:
            print("使用默认配置")
    
    # 显示配置
    config_manager.display_config()
    
    # 询问是否保存配置
    save_config = input("\n💾 是否保存当前配置? (y/n): ").lower() == 'y'
    if save_config:
        config_name = input("请输入配置文件名 (默认: 自动生成): ")
        config_manager.save_config(config_name if config_name else None)
    
    # 开始回测
    start_backtest = input("\n🚀 是否开始回测? (y/n): ").lower() == 'y'
    if not start_backtest:
        print("回测已取消")
        return
    
    # 运行回测
    config = config_manager.get_config()
    engine = SimpleBacktestEngine(config)
    report = engine.run_simulation()
    
    if report:
        # 输出报告
        if config['report_format'] == 'json':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f'reports/backtest_report_{timestamp}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 JSON报告已保存: {report_file}")
        else:
            engine.print_report(report)
        
        # 生成图表
        if config['plot_results'] or config['save_results']:
            engine.plot_results(report)

def run_batch_backtest(config_manager: ConfigManager):
    """运行批处理回测"""
    print("\n" + "="*60)
    print("📊 批处理回测配置")
    print("="*60)
    
    # 定义参数网格
    print("\n请选择批处理测试类型:")
    print("1. 测试止损止盈参数")
    print("2. 测试缠论参数")
    print("3. 测试时间框架")
    print("4. 自定义参数网格")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == '1':
        # 测试止损止盈参数
        param_sets = []
        for sl in [1.0, 1.5, 2.0, 2.5]:
            for tp in [2.0, 3.0, 4.0, 5.0]:
                param_sets.append({
                    'stop_loss_pct': sl,
                    'take_profit_pct': tp
                })
    
    elif choice == '2':
        # 测试缠论参数
        param_sets = []
        for fp in [3, 5, 7]:
            for ct in [0.5, 0.6, 0.7, 0.8]:
                param_sets.append({
                    'fractal_period': fp,
                    'confidence_threshold': ct
                })
    
    elif choice == '3':
        # 测试时间框架
        param_sets = []
        for tf in ['15m', '30m', '1h', '4h']:
            param_sets.append({
                'timeframe': tf,
                'enable_entry_scoping': tf != '15m'  # 15分钟框架不使用精确入场
            })
    
    elif choice == '4':
        # 自定义参数网格
        print("\n🔧 自定义参数网格（每行格式: 参数名=值1,值2,值3）")
        print("例如: stop_loss_pct=1.0,1.5,2.0")
        print("输入空行结束")
        
        param_grid = {}
        while True:
            line = input("输入参数: ").strip()
            if not line:
                break
            
            try:
                key, values_str = line.split('=')
                key = key.strip()
                values = [v.strip() for v in values_str.split(',')]
                
                # 转换类型
                converted_values = []
                for v in values:
                    if v.replace('.', '').isdigit():
                        if '.' in v:
                            converted_values.append(float(v))
                        else:
                            converted_values.append(int(v))
                    elif v.lower() in ['true', 'false']:
                        converted_values.append(v.lower() == 'true')
                    else:
                        converted_values.append(v)
                
                param_grid[key] = converted_values
            except:
                print("❌ 格式错误，请使用: 参数名=值1,值2,值3")
        
        # 生成参数组合
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        param_sets = []
        for combination in product(*values):
            param_set = dict(zip(keys, combination))
            param_sets.append(param_set)
    
    else:
        print("❌ 无效选项")
        return
    
    print(f"\n📊 将测试 {len(param_sets)} 组参数")
    proceed = input("是否继续? (y/n): ").lower() == 'y'
    
    if not proceed:
        print("批处理回测已取消")
        return
    
    # 运行批处理回测
    batch = BatchBacktest(config_manager)
    batch.run_batch(param_sets)

# ====================== 配置文件生成器 ======================
def create_sample_configs():
    """创建示例配置文件"""
    sample_configs = {
        '保守配置': {
            'stop_loss_pct': 1.0,
            'take_profit_pct': 2.0,
            'trade_amount': 0.0005,
            'confidence_threshold': 0.7
        },
        '激进配置': {
            'stop_loss_pct': 2.0,
            'take_profit_pct': 5.0,
            'trade_amount': 0.002,
            'confidence_threshold': 0.5
        },
        '日内交易': {
            'timeframe': '15m',
            'entry_timeframe': '5m',
            'stop_loss_pct': 0.8,
            'take_profit_pct': 1.5,
            'enable_entry_scoping': True
        },
        '趋势跟踪': {
            'timeframe': '4h',
            'stop_loss_pct': 3.0,
            'take_profit_pct': 8.0,
            'fractal_period': 7,
            'enable_entry_scoping': False
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    
    for name, config in sample_configs.items():
        filename = f"configs/{name.lower().replace(' ', '_')}_config.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ 创建示例配置: {filename}")
    
    print("\n📝 创建示例配置文件完成！")
    print("使用方法:")
    print("  python backtest_configurable.py --config configs/保守配置_config.json")
    print("  python backtest_configurable.py --config configs/激进配置_config.json")

# ====================== 快速启动脚本 ======================
def quick_start():
    """快速启动脚本"""
    print("\n🚀 快速启动回测")
    print("-" * 40)
    
    # 预设配置
    presets = {
        '1': {'name': '2024上半年测试', 'start_date': '2024-01-01', 'end_date': '2024-06-30'},
        '2': {'name': '30分钟框架测试', 'timeframe': '30m', 'entry_timeframe': '5m'},
        '3': {'name': '1小时框架测试', 'timeframe': '1h', 'entry_timeframe': '15m'},
        '4': {'name': '小资金测试', 'initial_capital': 5000, 'trade_amount': 0.0005},
        '5': {'name': '保守策略', 'stop_loss_pct': 1.0, 'take_profit_pct': 2.0, 'confidence_threshold': 0.7}
    }
    
    print("请选择预设配置:")
    for key, preset in presets.items():
        print(f"  {key}. {preset['name']}")
    print("  6. 自定义配置")
    
    choice = input("\n请选择 (1-6): ").strip()
    
    config_manager = ConfigManager()
    
    if choice in presets:
        config = config_manager.get_config()
        config.update(presets[choice])
        config_manager.config = config
        
        print(f"\n✅ 使用预设配置: {presets[choice]['name']}")
        config_manager.display_config()
        
        # 运行回测
        engine = SimpleBacktestEngine(config)
        report = engine.run_simulation()
        
        if report:
            engine.print_report(report)
            engine.plot_results(report)
    else:
        # 自定义配置
        main()

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        main()
    else:
        print("\n请选择运行模式:")
        print("1. 快速启动（预设配置）")
        print("2. 完整配置向导")
        print("3. 创建示例配置文件")
        print("4. 查看帮助")
        
        mode = input("\n请输入选项 (1-4): ").strip()
        
        if mode == '1':
            quick_start()
        elif mode == '2':
            main()
        elif mode == '3':
            create_sample_configs()
        elif mode == '4':
            print("\n使用帮助:")
            print("  快速启动: python backtest_configurable.py")
            print("  完整配置: python backtest_configurable.py --help")
            print("  使用配置文件: python backtest_configurable.py --config my_config.json")
            print("  环境变量: export BT_SYMBOL='BTC/USDT'; export BT_INITIAL_CAPITAL='10000'")
        else:
            print("❌ 无效选项")