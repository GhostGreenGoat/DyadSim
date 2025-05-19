"""
基准测试运行器，用于批量执行虚拟人物模拟并收集评估结果
"""

import os
import json
import time
import argparse
import random
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

from api.data.character_profiles import character_profiles, get_character_by_scenario
from api.data.conflict_scenarios import conflict_scenarios, get_scenario_by_id, get_situation_by_id
from character_simulator import CharacterSimulator


class BenchmarkRunner:
    """基准测试运行器类"""
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        log_dir: str = "logs",
        max_turns: int = 10,
        character_api: str = "deepseek",
        partner_api: str = "openrouter",
        expert_apis: Optional[List[str]] = None,
        character_model: Optional[str] = None,
        partner_model: Optional[str] = None,
        expert_model: Optional[str] = None,
        use_emotion_prediction: bool = True,
        use_expert_analysis: bool = True,
        num_experts: int = 3,
        chinese_font: Optional[FontProperties] = None,
        random_seed: Optional[int] = None
    ):
        """
        初始化基准测试运行器
        
        参数:
            output_dir (str): 输出目录
            log_dir (str): 日志目录
            max_turns (int): 最大对话轮次
            character_api (str): 虚拟人物使用的API类型
            partner_api (str): 对话伴侣使用的API类型
            expert_apis (List[str], optional): 专家分析使用的API类型列表
            character_model (str, optional): 虚拟人物使用的模型名称
            partner_model (str, optional): 对话伴侣使用的模型名称
            expert_model (str, optional): 专家分析使用的模型名称
            use_emotion_prediction (bool): 是否启用待测模型的情感预测
            use_expert_analysis (bool): 是否启用专家的情感分析
            num_experts (int): 专家数量
            chinese_font (FontProperties, optional): 中文字体属性
            random_seed (int, optional): 随机种子，设置后将固定随机选择结果
        """
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.max_turns = max_turns
        self.character_api = character_api
        self.partner_api = partner_api
        self.expert_apis = expert_apis
        self.character_model = character_model
        self.partner_model = partner_model
        self.expert_model = expert_model
        self.use_emotion_prediction = use_emotion_prediction
        self.use_expert_analysis = use_expert_analysis
        self.num_experts = num_experts
        self.random_seed = random_seed
        
        # 如果设置了随机种子，初始化随机数生成器
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 加载中文字体（用于生成图表）
        if chinese_font is None:
            # 检测操作系统，以使用合适的字体
            import platform
            system = platform.system()
            
            if system == "Windows":
                # Windows系统使用SimHei
                font_path = r"C:\Windows\Fonts\simhei.ttf"
                if os.path.exists(font_path):
                    self.chinese_font = FontProperties(fname=font_path)
                else:
                    self.chinese_font = None
            elif system == "Darwin":  # macOS
                # macOS系统尝试使用系统字体
                self.chinese_font = FontProperties(family='Heiti SC')
            else:  # Linux或其他系统
                # 尝试使用系统默认字体
                self.chinese_font = None
        else:
            self.chinese_font = chinese_font
    
    def generate_test_cases(
        self,
        num_characters: int = 5,
        scenario_ids: Optional[List[str]] = None,
        num_scenarios: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        生成测试用例（每个场景只生成主角为第一个情境、伴侣为第二个情境的唯一分配方式），并为每个用例生成唯一的初始情绪和初始情绪值
        """
        test_cases = []
        if not scenario_ids:
            scenario_ids = [scenario["id"] for scenario in conflict_scenarios]
        if num_scenarios is not None and num_scenarios > 0 and num_scenarios < len(scenario_ids):
            scenario_ids = random.sample(scenario_ids, num_scenarios)
        for scenario_id in scenario_ids:
            scenario = get_scenario_by_id(scenario_id)
            if not scenario:
                continue
            situations = scenario.get("situations", [])
            if len(situations) < 2:
                continue
            # 主角和伴侣profile/situation都锁定
            character = get_character_by_scenario(scenario_id, situations[0]["id"])
            partner = get_character_by_scenario(scenario_id, situations[1]["id"])
            if character and partner:
                sim = CharacterSimulator(
                    character_config=character,
                    scenario_id=scenario_id,
                    situation_id=situations[0]["id"],
                    character_api=self.character_api,
                    partner_api=self.partner_api,
                    expert_apis=self.expert_apis,
                    character_model=self.character_model,
                    partner_model=self.partner_model,
                    expert_model=self.expert_model,
                    max_turns=self.max_turns,
                    log_dir=self.log_dir,
                    use_emotion_prediction=self.use_emotion_prediction,
                    use_expert_analysis=self.use_expert_analysis,
                    num_experts=self.num_experts,
                    mode="default"
                )
                initial_emotion, initial_emotion_score = sim._calculate_initial_emotion(return_both=True)
                test_cases.append({
                    "character": character,
                    "partner": partner,
                    "scenario_id": scenario_id,
                    "situation_id": situations[0]["id"],
                    "partner_situation_id": situations[1]["id"],
                    "initial_emotion": initial_emotion,
                    "initial_emotion_score": initial_emotion_score
                })
            else:
                continue
        return test_cases
    
    def run_single_test(self, test_case: Dict[str, Any], mode: str = "default") -> Dict[str, Any]:
        """
        运行单个测试用例，支持mode
        """
        simulator = CharacterSimulator(
            character_config=test_case['character'],
            scenario_id=test_case['scenario_id'],
            situation_id=test_case['situation_id'],
            partner_profile=test_case.get('partner'),
            partner_situation_id=test_case.get('partner_situation_id'),
            character_api=self.character_api,
            partner_api=self.partner_api,
            expert_apis=self.expert_apis,
            character_model=self.character_model,
            partner_model=self.partner_model,
            expert_model=self.expert_model,
            max_turns=self.max_turns,
            log_dir=self.log_dir,
            use_emotion_prediction=self.use_emotion_prediction,
            use_expert_analysis=self.use_expert_analysis,
            num_experts=self.num_experts,
            mode=mode,
            initial_emotion=test_case.get('initial_emotion'),
            initial_emotion_score=test_case.get('initial_emotion_score')
        )
        try:
            result = simulator.run_simulation()
            prediction_accuracy = self._calculate_prediction_accuracy(result)
            expert_consensus = self._calculate_expert_consensus(result)
            summary = {
                "character_id": test_case['character']['id'],
                "character_name": test_case['character']['name'],
                "personality_type": test_case['character']['personality_type'],
                "relationship_belief": test_case['character']['relationship_belief'],
                "communication_type": test_case['character']['communication_type'],
                "attachment_style": test_case['character']['attachment_style'],
                "scenario_id": test_case['scenario_id'],
                "scenario_name": result['scenario']['scenario']['name'],
                "situation_name": result['scenario']['situation']['name'],
                "turns_completed": result['turns_completed'],
                "final_emotion_score": result['final_emotion_score'],
                "initial_emotion_score": result['emotion_history'][0]['score'] if result['emotion_history'] else 0,
                "emotion_change": (result['final_emotion_score'] - (result['emotion_history'][0]['score'] if result['emotion_history'] else 0)),
                "emotion_prediction_accuracy": prediction_accuracy,
                "expert_consensus": expert_consensus,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True,
                "mode": mode
            }
            return summary
        except Exception as e:
            summary = {
                "character_id": test_case['character']['id'],
                "character_name": test_case['character']['name'],
                "personality_type": test_case['character']['personality_type'],
                "relationship_belief": test_case['character']['relationship_belief'],
                "communication_type": test_case['character']['communication_type'],
                "attachment_style": test_case['character']['attachment_style'],
                "scenario_id": test_case['scenario_id'],
                "turns_completed": 0,
                "final_emotion_score": 0,
                "initial_emotion_score": 0,
                "emotion_change": 0,
                "emotion_prediction_accuracy": 0,
                "expert_consensus": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "error": str(e),
                "mode": mode
            }
            return summary
    
    def _calculate_prediction_accuracy(self, result: Dict[str, Any]) -> float:
        """
        计算情感预测的准确度
        
        参数:
            result (Dict[str, Any]): 模拟结果
            
        返回:
            float: 情感预测准确度（0-1）
        """
        if not result.get('emotion_prediction_history') or len(result.get('emotion_prediction_history', [])) < 2:
            return 0.0
        
        # 计算预测准确度
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(result['emotion_prediction_history'])):
            # 跳过预测结果不完整的项
            if not result['emotion_prediction_history'][i].get('predicted_emotion'):
                continue
                
            # 获取预测轮次
            turn = result['emotion_prediction_history'][i]['turn']
            
            # 查找下一轮的实际情绪结果
            next_turn = turn + 1
            actual_emotion = None
            
            for emotion_record in result['emotion_history']:
                if emotion_record['turn'] == next_turn:
                    actual_emotion = emotion_record['emotion_info'].get('primary_emotion')
                    break
            
            # 如果没有下一轮的情绪数据，跳过
            if not actual_emotion:
                continue
                
            # 比较预测和实际结果
            predicted_emotion = result['emotion_prediction_history'][i]['predicted_emotion']
            
            # 检查预测是否接近实际情绪
            if predicted_emotion == actual_emotion:
                correct_predictions += 1
                total_predictions += 1
            elif self._are_similar_emotions(predicted_emotion, actual_emotion):
                correct_predictions += 0.5  # 部分正确
                total_predictions += 1
            else:
                total_predictions += 1
        
        # 计算准确率
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _are_similar_emotions(self, emotion1: str, emotion2: str) -> bool:
        """
        检查两种情绪是否相似
        
        参数:
            emotion1 (str): 第一种情绪
            emotion2 (str): 第二种情绪
            
        返回:
            bool: 如果情绪相似则为True
        """
        # 定义相似情绪组
        similar_groups = [
            {"愤怒", "厌恶", "烦躁"},
            {"悲伤", "失落", "绝望"},
            {"恐惧", "焦虑", "担忧"},
            {"快乐", "愉悦", "满足"},
            {"信任", "依赖", "安心"},
            {"期待", "希望", "憧憬"}
        ]
        
        # 检查两种情绪是否在同一组
        for group in similar_groups:
            if emotion1 in group and emotion2 in group:
                return True
                
        return False
    
    def _calculate_expert_consensus(self, result: Dict[str, Any]) -> float:
        """
        计算专家分析的一致性
        
        参数:
            result (Dict[str, Any]): 模拟结果
            
        返回:
            float: 专家一致性（0-1）
        """
        if not result.get('expert_analysis_history') or not result['expert_analysis_history']:
            return 0.0
        
        # 计算专家间的情绪分析一致性
        agreement_scores = []
        
        for turn_analysis in result['expert_analysis_history']:
            analyses = turn_analysis.get('analyses', [])
            
            # 如果专家数量少于2，无法计算一致性
            if len(analyses) < 2:
                continue
                
            # 统计情绪分布
            emotion_counts = {}
            total_analyses = len(analyses)
            
            for analysis in analyses:
                emotion = analysis.get('primary_emotion')
                if emotion and emotion != "未知":
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 找出最常见的情绪
            most_common_emotion = None
            most_common_count = 0
            
            for emotion, count in emotion_counts.items():
                if count > most_common_count:
                    most_common_emotion = emotion
                    most_common_count = count
            
            # 计算一致性比例
            if most_common_emotion and total_analyses > 0:
                agreement = most_common_count / total_analyses
                agreement_scores.append(agreement)
        
        # 计算平均一致性
        return sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
    
    def run_benchmark(self, test_cases=None, num_characters=5, scenario_ids=None, parallel=False, max_workers=4, num_scenarios=None, modes=None, output_dir=None):
        """
        运行基准测试，支持多mode
        """
        start_time = time.time()
        if not test_cases:
            test_cases = self.generate_test_cases(
                num_characters=num_characters,
                scenario_ids=scenario_ids,
                num_scenarios=num_scenarios
            )
        if not modes:
            modes = ["default"]
        if not output_dir:
            output_dir = self.output_dir
        results = []
        for test_case in test_cases:
            for mode in modes:
                summary = self.run_single_test(test_case, mode=mode)
                results.append(summary)
                # 自动保存结果
                fname = f"result_{summary['character_id']}_{summary['scenario_id']}_{mode}.json"
                fpath = os.path.join(output_dir, fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
        total_time = time.time() - start_time
        print(f"\n===== 测试结果摘要 =====")
        print(f"总测试用例: {len(test_cases)} x {len(modes)} = {len(results)}")
        print(f"成功测试: {sum(1 for r in results if r.get('success', False))}")
        print(f"失败测试: {sum(1 for r in results if not r.get('success', False))}")
        print(f"总运行时间: {total_time:.2f}秒")
        return {
            "results": results,
            "total_time": total_time
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行LQBench基准测试')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='输出目录路径')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录路径')
    parser.add_argument('--max-turns', type=int, default=10, help='最大对话轮次')
    parser.add_argument('--character-api', type=str, default='deepseek', help='虚拟人物使用的API类型')
    parser.add_argument('--partner-api', type=str, default='openrouter', help='对话伴侣使用的API类型')
    parser.add_argument('--expert-apis', type=str, nargs='+', default=['deepseek'], help='专家分析使用的API类型列表')
    parser.add_argument('--character-model', type=str, default=None, help='虚拟人物使用的模型名称')
    parser.add_argument('--partner-model', type=str, default=None, help='对话伴侣使用的模型名称')
    parser.add_argument('--expert-model', type=str, default=None, help='专家分析使用的模型名称')
    parser.add_argument('--num-characters', type=int, default=3, help='随机生成的虚拟人物数量')
    parser.add_argument('--scenario-ids', type=str, nargs='+', help='要测试的场景ID列表')
    parser.add_argument('--num-scenarios', type=int, help='要随机选择的场景数量')
    parser.add_argument('--parallel', action='store_true', help='是否并行运行测试')
    parser.add_argument('--max-workers', type=int, default=4, help='并行运行时的最大工作线程数')
    parser.add_argument('--use-emotion-prediction', action='store_true', default=True, help='是否启用情感预测')
    parser.add_argument('--use-expert-analysis', action='store_true', default=True, help='是否启用专家分析')
    parser.add_argument('--num-experts', type=int, default=3, help='专家数量')
    parser.add_argument('--random-seed', type=int, default=None, help='随机种子，设置后将固定随机选择结果')
    parser.add_argument('--modes', type=str, nargs='+', default=['default', 'improved'], help='要测试的模式列表（如 default improved）')
    
    args = parser.parse_args()
    
    # 创建基准测试运行器
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        max_turns=args.max_turns,
        character_api=args.character_api,
        partner_api=args.partner_api,
        expert_apis=args.expert_apis,
        character_model=args.character_model,
        partner_model=args.partner_model,
        expert_model=args.expert_model,
        use_emotion_prediction=args.use_emotion_prediction,
        use_expert_analysis=args.use_expert_analysis,
        num_experts=args.num_experts,
        random_seed=args.random_seed
    )
    
    # 运行基准测试
    runner.run_benchmark(
        num_characters=args.num_characters,
        scenario_ids=args.scenario_ids,
        parallel=args.parallel,
        max_workers=args.max_workers,
        num_scenarios=args.num_scenarios,
        modes=args.modes,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 