#!/usr/bin/env python3
"""
缠论交易策略回测命令行工具
支持丰富的命令行参数配置
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入真实回测引擎
from backtest_system import RealChanBacktestEngine as RealEngine
from backtest_system import RealBacktestConfig as RealConfig

# ====================== 核心回测引擎 ======================
class ChanBacktestEngine:
    """缠论回测引擎（简化版）"""
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_history = []
        
    def run(self):
        """运行回测"""
        print(f"\n🔍 运行回测: {self.config['symbol']} {self.config['timeframe']}")
        print(f"   时间: {self.config['start_date']} 到 {self.config['end_date']}")
        print(f"   资金: ${self.config['initial_capital']}")
        
        # 使用真实回测引擎
        if self.config['verbose']:
            print("   使用真实数据回测中...")
        
        # 转换配置到真实回测引擎的格式
        real_config = RealConfig()
        real_config.symbol = self.config['symbol']
        real_config.timeframe = self.config['timeframe']
        real_config.start_date = self.config['start_date']
        real_config.end_date = self.config['end_date']
        real_config.initial_capital = self.config['initial_capital']
        real_config.trade_amount = self.config['trade_amount']
        real_config.leverage = self.config['leverage']
        real_config.stop_loss_pct = self.config['stop_loss_pct']
        real_config.take_profit_pct = self.config['take_profit_pct']
        real_config.fee_rate = self.config['fee_rate']
        real_config.slippage = self.config['slippage']
        real_config.fractal_period = self.config['fractal_period']
        real_config.confidence_threshold = self.config['confidence_threshold']
        real_config.enable_entry_scoping = self.config['enable_entry_scoping']
        real_config.verbose = self.config['verbose']
        real_config.plot_results = self.config['plot_results']
        real_config.save_results = self.config['save_results']
        
        # 创建并运行真实回测引擎
        real_engine = RealEngine(real_config)
        real_results = real_engine.run_backtest()
        
        if real_results is None:
            print("❌ 回测失败")
            return None
        
        # 转换结果格式以兼容原有接口
        results = {
            'total_return': real_results['summary']['total_return_pct'],
            'max_drawdown': real_results['summary']['max_drawdown_pct'],
            'win_rate': real_results['summary']['win_rate'],
            'total_trades': real_results['summary']['total_trades'],
            'sharpe_ratio': 0.0  # 真实回测没有计算夏普比率，可以考虑后续添加
        }
        
        # 保存真实回测的详细结果
        self.real_results = real_results
        
        return results
    
    def generate_sample_results(self):
        """生成示例结果（实际使用时应替换为真实回测）"""
        return {
            'total_return': 15.5,
            'max_drawdown': -8.2,
            'win_rate': 65.3,
            'total_trades': 42,
            'sharpe_ratio': 1.8
        }
    
    def print_report(self, results):
        """打印报告"""
        print("\n" + "="*60)
        print("📊 回测结果报告")
        print("="*60)
        print(f"总收益率: {results['total_return']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2f}%")
        print(f"胜率: {results['win_rate']:.2f}%")
        print(f"交易次数: {results['total_trades']}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print("="*60)
    
    def save_report(self, results, output_dir):
        """保存报告"""
        # 确保输出目录在外层目录
        outer_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_dir)
        os.makedirs(outer_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"{outer_dir}/report_{timestamp}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': results,
            'summary': {
                'symbol': self.config['symbol'],
                'timeframe': self.config['timeframe'],
                'total_return': results['total_return'],
                'max_drawdown': results['max_drawdown']
            }
        }
        
        # 如果有真实回测的详细结果，也保存起来
        if hasattr(self, 'real_results') and self.real_results:
            report_data['real_results'] = self.real_results
            
            # 另外保存真实回测的详细报告
            real_report_file = f"{outer_dir}/real_report_{timestamp}.json"
            with open(real_report_file, 'w') as f:
                json.dump(self.real_results, f, indent=2, default=str)
            print(f"💾 真实回测详细报告已保存: {real_report_file}")
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"💾 报告已保存: {report_file}")
        return report_file

# ====================== 命令行参数解析 ======================
def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='缠论交易策略回测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本回测
  %(prog)s --start 2024-01-01 --end 2024-06-01
  
  # 指定交易对和资金
  %(prog)s --symbol BTC/USDT --capital 50000 --timeframe 1h
  
  # 设置风险参数
  %(prog)s --stop-loss 1.0 --take-profit 2.5 --fee 0.0008
  
  # 缠论参数配置
  %(prog)s --fractal-period 7 --confidence 0.7
  
  # 输出选项
  %(prog)s --verbose --output my_reports --format json
  
  # 批量测试
  %(prog)s --batch "sl=1.0,1.5,2.0;tp=2.0,3.0,4.0"
  
  # 使用配置文件
  %(prog)s --config my_config.json
        """
    )
    
    # 必需参数
    parser.add_argument('--start', '--start-date', required=True,
                       help='开始日期 (格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)')
    
    # 可选参数
    parser.add_argument('--end', '--end-date', 
                       help='结束日期 (默认: 当前日期)')
    parser.add_argument('--days', type=int,
                       help='回测天数 (从开始日期算起)')
    parser.add_argument('--symbol', default='BTC/USDT',
                       help='交易对 (默认: BTC/USDT)')
    parser.add_argument('--timeframe', default='30m',
                       help='K线时间框架 (默认: 30m)')
    parser.add_argument('--capital', '--initial-capital', type=float, default=10000.0,
                       help='初始资金 (默认: 10000.0)')
    parser.add_argument('--amount', '--trade-amount', type=float, default=0.001,
                       help='每次交易数量 (默认: 0.001)')
    
    # 风险参数
    risk_group = parser.add_argument_group('风险参数')
    risk_group.add_argument('--stop-loss', '--sl', type=float, default=1.5,
                           help='止损百分比 (默认: 1.5)')
    risk_group.add_argument('--take-profit', '--tp', type=float, default=3.0,
                           help='止盈百分比 (默认: 3.0)')
    risk_group.add_argument('--fee', '--fee-rate', type=float, default=0.001,
                           help='手续费率 (默认: 0.001)')
    risk_group.add_argument('--slippage', type=float, default=0.0005,
                           help='滑点 (默认: 0.0005)')
    risk_group.add_argument('--leverage', type=float, default=1.0,
                           help='杠杆倍数 (默认: 1.0)')
    
    # 策略参数
    strategy_group = parser.add_argument_group('策略参数')
    strategy_group.add_argument('--fractal-period', type=int, default=5,
                               help='缠论分型周期 (默认: 5)')
    strategy_group.add_argument('--pivot-lookback', type=int, default=20,
                               help='中枢观察周期 (默认: 20)')
    strategy_group.add_argument('--confidence', type=float, default=0.6,
                               help='信号信心阈值 (默认: 0.6)')
    strategy_group.add_argument('--entry-scoping', choices=['on', 'off'], default='on',
                               help='精确入场开关 (默认: on)')
    strategy_group.add_argument('--ai-analysis', choices=['on', 'off'], default='off',
                               help='AI分析开关 (默认: off)')
    
    # 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='详细输出模式')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='安静模式，只输出结果')
    output_group.add_argument('--output', '-o', default='reports',
                             help='输出目录 (默认: reports)')
    output_group.add_argument('--format', choices=['console', 'json', 'csv', 'html'],
                             default='console', help='输出格式 (默认: console)')
    output_group.add_argument('--plot', choices=['on', 'off'], default='on',
                             help='生成图表 (默认: on)')
    output_group.add_argument('--save', choices=['on', 'off'], default='on',
                             help='保存结果文件 (默认: on)')
    
    # 高级参数
    advanced_group = parser.add_argument_group('高级参数')
    advanced_group.add_argument('--config', help='配置文件路径')
    advanced_group.add_argument('--save-config', help='保存当前配置到文件')
    advanced_group.add_argument('--batch', help='批量测试参数网格')
    advanced_group.add_argument('--compare', nargs='+', help='比较多个配置文件')
    advanced_group.add_argument('--mode', choices=['single', 'walkforward', 'optimize'],
                               default='single', help='回测模式 (默认: single)')
    
    # 数据参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--data-source', choices=['binance', 'mock', 'file'],
                           default='binance', help='数据源 (默认: binance)')
    data_group.add_argument('--data-file', help='本地数据文件路径')
    data_group.add_argument('--cache', choices=['on', 'off'], default='on',
                           help='使用数据缓存 (默认: on)')
    
    return parser

# ====================== 配置管理器 ======================
class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def parse_args_to_config(args):
        """解析参数为配置字典"""
        # 处理结束日期
        end_date = args.end
        if not end_date:
            if args.days:
                start_dt = datetime.strptime(args.start, '%Y-%m-%d')
                end_dt = start_dt + timedelta(days=args.days)
                end_date = end_dt.strftime('%Y-%m-%d')
            else:
                end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 处理时间格式
        if len(args.start) == 10:
            start_date = f"{args.start} 00:00:00"
        else:
            start_date = args.start
            
        if len(end_date) == 10:
            end_date = f"{end_date} 23:59:59"
        
        # 构建配置字典
        config = {
            'symbol': args.symbol,
            'timeframe': args.timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': args.capital,
            'trade_amount': args.amount,
            'leverage': args.leverage,
            'stop_loss_pct': args.stop_loss,
            'take_profit_pct': args.take_profit,
            'fee_rate': args.fee,
            'slippage': args.slippage,
            'fractal_period': args.fractal_period,
            'central_pivot_lookback': args.pivot_lookback,
            'confidence_threshold': args.confidence,
            'enable_entry_scoping': args.entry_scoping == 'on',
            'enable_ai_analysis': args.ai_analysis == 'on',
            'verbose': args.verbose,
            'output_dir': args.output,
            'output_format': args.format,
            'plot_results': args.plot == 'on',
            'save_results': args.save == 'on',
            'data_source': args.data_source,
            'data_file': args.data_file,
            'use_cache': args.cache == 'on',
            'mode': args.mode
        }
        
        # 如果提供了配置文件，覆盖默认配置
        if args.config:
            ConfigManager.load_config_from_file(args.config, config)
        
        return config
    
    @staticmethod
    def load_config_from_file(config_file, base_config):
        """从文件加载配置"""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            base_config.update(file_config)
            print(f"✅ 从文件加载配置: {config_file}")
        except Exception as e:
            print(f"⚠️  配置文件加载失败: {e}")
    
    @staticmethod
    def save_config_to_file(config, filename):
        """保存配置到文件"""
        # 确保配置文件也在外层目录
        outer_configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
        os.makedirs(outer_configs_dir, exist_ok=True)
        if not filename.endswith('.json'):
            filename = f"{outer_configs_dir}/{filename}.json"
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"💾 配置已保存: {filename}")
        return filename
    
    @staticmethod
    def display_config(config):
        """显示配置信息"""
        print("\n" + "="*60)
        print("📋 当前配置")
        print("="*60)
        
        groups = {
            '回测参数': ['symbol', 'timeframe', 'start_date', 'end_date'],
            '资金参数': ['initial_capital', 'trade_amount', 'leverage'],
            '风险参数': ['stop_loss_pct', 'take_profit_pct', 'fee_rate', 'slippage'],
            '策略参数': ['fractal_period', 'confidence_threshold', 'enable_entry_scoping'],
            '输出参数': ['output_dir', 'output_format', 'plot_results', 'verbose']
        }
        
        for group, keys in groups.items():
            print(f"\n{group}:")
            for key in keys:
                if key in config:
                    value = config[key]
                    if isinstance(value, float):
                        if 'pct' in key or 'fee' in key:
                            print(f"  {key}: {value:.2f}%")
                        else:
                            print(f"  {key}: {value}")
                    elif isinstance(value, bool):
                        print(f"  {key}: {'是' if value else '否'}")
                    else:
                        print(f"  {key}: {value}")
        
        print("="*60)

# ====================== 批量测试工具 ======================
class BatchTester:
    """批量测试工具"""
    
    @staticmethod
    def parse_param_grid(param_str):
        """解析参数网格字符串"""
        param_grid = {}
        
        # 格式: "sl=1.0,1.5,2.0;tp=2.0,3.0,4.0"
        params = param_str.split(';')
        
        for param in params:
            if '=' in param:
                key, values_str = param.split('=', 1)
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
                    elif v.lower() in ['on', 'off']:
                        converted_values.append(v.lower() == 'on')
                    else:
                        converted_values.append(v)
                
                param_grid[key] = converted_values
        
        return param_grid
    
    @staticmethod
    def generate_param_sets(param_grid):
        """生成参数组合"""
        from itertools import product
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        param_sets = []
        for combination in product(*values):
            param_set = dict(zip(keys, combination))
            
            # 转换参数名
            param_set = BatchTester.convert_param_names(param_set)
            param_sets.append(param_set)
        
        return param_sets
    
    @staticmethod
    def convert_param_names(param_set):
        """转换参数名"""
        name_mapping = {
            'sl': 'stop_loss_pct',
            'tp': 'take_profit_pct',
            'fp': 'fractal_period',
            'ct': 'confidence_threshold',
            'capital': 'initial_capital',
            'amount': 'trade_amount'
        }
        
        converted = {}
        for key, value in param_set.items():
            if key in name_mapping:
                converted[name_mapping[key]] = value
            else:
                converted[key] = value
        
        return converted
    
    @staticmethod
    def run_batch_tests(base_config, param_sets):
        """运行批量测试"""
        results = []
        
        for i, params in enumerate(param_sets, 1):
            print(f"\n🔧 测试组合 {i}/{len(param_sets)}:")
            
            # 合并配置
            test_config = base_config.copy()
            test_config.update(params)
            
            # 显示参数
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            # 运行回测
            engine = ChanBacktestEngine(test_config)
            result = engine.run()
            
            if result:
                results.append({
                    'params': params,
                    'result': result
                })
        
        return BatchTester.analyze_batch_results(results)
    
    @staticmethod
    def analyze_batch_results(results):
        """分析批量测试结果"""
        if not results:
            return None
        
        print("\n" + "="*60)
        print("📊 批量测试结果分析")
        print("="*60)
        
        # 找出最佳结果
        best_return = max(results, key=lambda x: x['result']['total_return'])
        best_sharpe = max(results, key=lambda x: x['result'].get('sharpe_ratio', 0))
        
        print(f"\n🏆 最佳收益率:")
        print(f"   收益率: {best_return['result']['total_return']:.2f}%")
        print(f"   参数: {json.dumps(best_return['params'], indent=2)}")
        
        print(f"\n📈 最佳夏普比率:")
        print(f"   夏普比率: {best_sharpe['result'].get('sharpe_ratio', 0):.2f}")
        print(f"   参数: {json.dumps(best_sharpe['params'], indent=2)}")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 确保批量测试结果也在外层目录
        outer_reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
        os.makedirs(outer_reports_dir, exist_ok=True)
        results_file = f"{outer_reports_dir}/batch_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 批量测试结果已保存: {results_file}")
        
        return best_return, best_sharpe

# ====================== 主函数 ======================
def main():
    """主函数"""
    # 创建解析器
    parser = create_parser()
    args = parser.parse_args()
    
    # 处理安静模式
    if args.quiet:
        args.verbose = False
    
    # 解析配置
    config = ConfigManager.parse_args_to_config(args)
    
    # 保存配置（如果指定）
    if args.save_config:
        ConfigManager.save_config_to_file(config, args.save_config)
    
    # 显示配置
    if not args.quiet:
        ConfigManager.display_config(config)
    
    # 批量测试模式
    if args.batch:
        print("\n📊 进入批量测试模式")
        
        # 解析参数网格
        param_grid = BatchTester.parse_param_grid(args.batch)
        param_sets = BatchTester.generate_param_sets(param_grid)
        
        print(f"将测试 {len(param_sets)} 种参数组合")
        
        # 运行批量测试
        BatchTester.run_batch_tests(config, param_sets)
        
        return
    
    # 单次回测模式
    print("\n🚀 开始单次回测")
    
    # 确认开始
    if not args.quiet:
        confirm = input("\n确认开始回测? (y/n): ").lower()
        if confirm != 'y':
            print("回测已取消")
            return
    
    # 创建回测引擎
    engine = ChanBacktestEngine(config)
    
    # 运行回测
    results = engine.run()
    
    # 输出结果
    if results:
        engine.print_report(results)
        
        # 保存结果
        if config['save_results']:
            engine.save_report(results, config['output_dir'])
        
        print("\n✅ 回测完成!")

# ====================== 便捷运行脚本 ======================
if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs('reports', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 运行主函数
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)