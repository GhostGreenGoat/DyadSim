import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter, defaultdict
from tqdm import tqdm
import sys
import time

# 确保能import到llm.py
sys.path.append("api")
from llm import LLMClient

# 配置路径
LOG_DIR = "log-gemini_NVC"
RESULT_DIR = "benchmark_result_gemini_NVC"

# 情感词典
POS_WORDS = ['理解', '支持', '感谢', '喜欢', '爱', '包容', '信任', '合作', '共情', '鼓励', '安慰', '希望', '愿意', '一起', '努力', '开心', '幸福', '温暖']
NEG_WORDS = ['讨厌', '生气', '愤怒', '失望', '难过', '伤心', '烦', '崩溃', '不想', '不理', '冷漠', '指责', '攻击', '责怪', '无语', '绝望', '分手', '离开']

POS_WORDS_EN = {
    '理解': 'understand', '支持': 'support', '感谢': 'thanks', '喜欢': 'like', '爱': 'love', '包容': 'tolerate',
    '信任': 'trust', '合作': 'cooperate', '共情': 'empathy', '鼓励': 'encourage', '安慰': 'comfort',
    '希望': 'hope', '愿意': 'willing', '一起': 'together', '努力': 'strive', '开心': 'happy', '幸福': 'happy', '温暖': 'warm'
}
NEG_WORDS_EN = {
    '讨厌': 'hate', '生气': 'angry', '愤怒': 'angry', '失望': 'disappointed', '难过': 'sad', '伤心': 'sad',
    '烦': 'annoyed', '崩溃': 'breakdown', '不想': 'unwilling', '不理': 'ignore', '冷漠': 'indifferent',
    '指责': 'blame', '攻击': 'attack', '责怪': 'blame', '无语': 'speechless', '绝望': 'despair', '分手': 'breakup', '离开': 'leave'
}
ALL_WORDS_EN = {**POS_WORDS_EN, **NEG_WORDS_EN}

BEHAVIOR_LABELS_EN = {
    "建设性沟通": "Constructive Communication",
    "回避/冷处理": "Avoidance/Withdrawal",
    "攻击/指责": "Attack/Blame",
    "自我暴露/脆弱表达": "Vulnerability/Disclosure",
    "情绪主诉": "Emotion Expression",
    "修复承诺": "Repair Commitment"
}

def replace_chinese_with_english(text):
    for zh, en in ALL_WORDS_EN.items():
        text = text.replace(zh, en)
    return text

# 自定义颜色表
COLOR_PALETTE = [
    '#8dd3c7', # Teal
    '#ffffb3', # Yellow
    '#bebada', # Purple
    '#fb8072', # Red
    '#80b1d3', # Blue
    '#fdb462', # Orange
    '#b3de69', # Green
    '#fccde5'  # Pink
]

# 在COLOR_PALETTE后添加新的字典
EMOTION_TYPES_EN = {
    '不安': 'Anxiety',
    '不满': 'Dissatisfaction',
    '不确定': 'Uncertainty',
    '不适': 'Discomfort',
    '专注': 'Focus',
    '低落': 'Depression',
    '关切': 'Concern',
    '关心': 'Care',
    '关爱': 'Affection',
    '兴奋': 'Excitement',
    '内疚': 'Guilt',
    '压抑': 'Suppression',
    '反思与释然': 'Reflection & Relief',
    '喜悦': 'Joy',
    '困惑': 'Confusion',
    '困惑与温暖交织': 'Mixed Confusion & Warmth',
    '困惑与犹豫': 'Confusion & Hesitation',
    '坚定': 'Determination',
    '失望': 'Disappointment',
    '失落': 'Loss',
    '委屈': 'Grievance',
    '安心': 'Peace of Mind',
    '安慰': 'Comfort',
    '安抚': 'Reassurance',
    '尴尬': 'Embarrassment',
    '平淡': 'Plain',
    '平静': 'Calm',
    '平静中略带失落': 'Calm with Slight Loss',
    '幸福': 'Happiness',
    '开心': 'Joy',
    '忧虑': 'Worry',
    '忧郁': 'Melancholy',
    '怀念': 'Nostalgia',
    '怀疑': 'Doubt',
    '恐惧': 'Fear',
    '恐惧与绝望': 'Fear & Despair',
    '愉悦': 'Pleasure',
    '感动': 'Touched',
    '感动与安心': 'Touched & Peaceful',
    '感动与犹豫': 'Touched & Hesitant',
    '感激': 'Gratitude',
    '感激与压力交织': 'Mixed Gratitude & Pressure',
    '感激与希望': 'Gratitude & Hope',
    '感激与疲惫': 'Gratitude & Fatigue',
    '愤怒': 'Anger',
    '抑郁': 'Depression',
    '担忧': 'Concern',
    '担忧与关心': 'Concern & Care',
    '担忧与安慰': 'Concern & Comfort',
    '放松': 'Relaxation',
    '无奈': 'Helplessness',
    '无聊': 'Boredom',
    '期待': 'Expectation',
    '期待与不安交织': 'Mixed Expectation & Anxiety',
    '欣慰': 'Relief',
    '沮丧': 'Frustration',
    '淡漠': 'Indifference',
    '温暖': 'Warmth',
    '温暖但略带困惑': 'Warmth with Slight Confusion',
    '满意': 'Satisfaction',
    '满足': 'Content',
    '烦恼': 'Trouble',
    '烦躁': 'Irritation',
    '焦虑': 'Anxiety',
    '焦虑缓解': 'Anxiety Relief',
    '犹豫': 'Hesitation',
    '理解': 'Understanding',
    '理解与妥协': 'Understanding & Compromise',
    '甜蜜': 'Sweet',
    '疲惫': 'Fatigue',
    '疲惫与不被理解': 'Fatigue & Misunderstood',
    '疲惫与渴望倾诉': 'Fatigue & Need to Express',
    '疲惫与轻微沮丧': 'Fatigue & Slight Frustration',
    '矛盾': 'Contradiction',
    '积极': 'Positive',
    '空虚': 'Emptiness',
    '绝望': 'Despair',
    '缓和': 'Ease',
    '耐心': 'Patience',
    '舒缓': 'Soothing',
    '被理解': 'Being Understood',
    '谨慎的信任': 'Cautious Trust',
    '谨慎的期待': 'Cautious Expectation',
    '轻微不满': 'Slight Dissatisfaction',
    '迷茫': 'Lost',
    '释然': 'Relief',
    '难过': 'Sad',
    '震惊与受伤': 'Shocked & Hurt',
    '顺从': 'Compliance',
    '高兴': 'Happy'
}

# =========================
# 1. 信息读取与对齐模块
# =========================

def read_log_file(filepath):
    """读取单个log文件，标准化输出轮次、情绪分数、消息，并提取情绪种类"""
    with open(filepath, 'r') as f:
        js = json.load(f)
    turns = []
    emo_hist = js.get('emotion_history', [])
    dial_hist = js.get('dialogue_history', [])
    for i, turn in enumerate(dial_hist):
        t = {
            'turn': turn.get('turn', i+1),
            'character_message': turn.get('character_message', ''),
            'partner_message': turn.get('partner_message', ''),
            'character_score': None,
            'emotions': []
        }
        # 情绪分数优先 emotion_history
        if i < len(emo_hist):
            t['character_score'] = emo_hist[i].get('score', None)
            # 优先从emotion_history中找emotions
            if 'emotion_info' in emo_hist[i] and 'emotions' in emo_hist[i]['emotion_info']:
                t['emotions'] = emo_hist[i]['emotion_info']['emotions']
            elif 'emotions' in emo_hist[i]:
                t['emotions'] = emo_hist[i]['emotions']
        # fallback: 从dialogue_history中找
        if t['character_score'] is None and 'emotion_info' in turn:
            t['character_score'] = turn['emotion_info'].get('score', None)
        if not t['emotions'] and 'emotion_info' in turn and 'emotions' in turn['emotion_info']:
            t['emotions'] = turn['emotion_info']['emotions']
        turns.append(t)
    return {
        'turns': turns,
        'final_emotion_score': js.get('final_emotion_score', None),
        'turns_completed': js.get('turns_completed', len(turns)),
        'max_turns': js.get('max_turns', 15)
    }

def align_cases(log_dir):
    """自动配对default/improved，返回标准化case列表"""
    files = [f for f in os.listdir(log_dir) if f.endswith('.json') and 'prompts' not in f]
    case_dict = defaultdict(dict)
    for f in files:
        m = re.match(r'(.+?)_(default|improved)_\d+\.json', f)
        if m:
            case, mode = m.groups()
            case_dict[case][mode] = os.path.join(log_dir, f)
    cases = []
    for case, pair in case_dict.items():
        if 'default' in pair and 'improved' in pair:
            cases.append({
                'case': case,
                'default': read_log_file(pair['default']),
                'improved': read_log_file(pair['improved'])
            })
    return cases

def read_benchmark_result(result_dir):
    """读取benchmark结果，返回case->mode->(ini, fin)"""
    files = [f for f in os.listdir(result_dir) if f.endswith('.json')]
    case_stat = defaultdict(dict)
    for f in files:
        m = re.match(r'result_(.+)_(default|improved)\.json', f)
        if m:
            case, mode = m.groups()
            with open(os.path.join(result_dir, f), 'r') as fp:
                js = json.load(fp)
            ini = js.get('initial_emotion_score', None)
            fin = js.get('final_emotion_score', None)
            case_stat[case][f'{mode}_ini'] = ini
            case_stat[case][f'{mode}_fin'] = fin
    return case_stat

# =========================
# 2. LLM 行为类型标注模块
# =========================

def extract_all_messages(cases):
    """提取所有消息及其索引"""
    messages = []
    msg_idx = []
    for case in cases:
        for mode in ['default', 'improved']:
            for i, turn in enumerate(case[mode]['turns']):
                for role in ['character_message', 'partner_message']:
                    msg = turn.get(role, '')
                    if msg:
                        messages.append(msg)
                        msg_idx.append((case['case'], mode, i, role))
    return messages, msg_idx

def llm_batch_label(messages, batch_size=10, model="openai/gpt-4.1", cache_dir=None):
    cache_file = os.path.join(cache_dir, "llm_behavior_labels.json") if cache_dir else "llm_behavior_labels.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    client = LLMClient(api_type="openrouter", model_name=model)
    results = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        uncached = [msg for msg in batch if msg not in cache]
        if uncached:
            prompt = f"""你是心理学对话分析专家。请根据以下定义，判断每条消息属于哪些行为类型（可多选）：\n\n1. 建设性沟通（Constructive Communication）：主动表达感受、倾听、共情、提出解决方案、表达理解/支持、非暴力沟通等。\n2. 回避/冷处理（Avoidance/Withdrawal）：逃避问题、拒绝沟通、转移话题、沉默、敷衍、推脱责任。\n3. 攻击/指责（Attack/Blame）：批评、指责、贬低、情绪爆发、讽刺、威胁、翻旧账。\n4. 自我暴露/脆弱表达（Vulnerability/Disclosure）：表达真实情绪、需求、恐惧、脆弱点，寻求理解或安慰。\n\n请严格输出如下JSON格式：\n[\n  {{\"text\": \"消息内容1\", \"labels\": [\"建设性沟通\"]}},\n  {{\"text\": \"消息内容2\", \"labels\": [\"攻击/指责\", \"自我暴露/脆弱表达\"]}},\n  ...\n]\n只输出JSON，不要解释。\n\n待判别消息如下：\n{json.dumps(uncached, ensure_ascii=False)}\n"""
            for _ in range(3):
                try:
                    response, _ = client.call(prompt, model=model, temperature=0.2, max_tokens=1500)
                    json_str = response[response.find('['):response.rfind(']')+1]
                    batch_result = json.loads(json_str)
                    for item in batch_result:
                        cache[item["text"]] = item["labels"]
                    break
                except Exception as e:
                    print(f"LLM标注失败，重试: {e}")
                    time.sleep(2)
            else:
                for msg in uncached:
                    cache[msg] = ["未知"]
            with open(cache_file, 'w') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        for msg in batch:
            results.append(cache.get(msg, ["未知"]))
    return results

def add_behavior_labels_to_cases(cases, labels, msg_idx):
    """将LLM标签写回case结构"""
    case_map = {(c['case'], m): c[m] for c in cases for m in ['default', 'improved']}
    for idx, label in enumerate(labels):
        case_name, mode, i, role = msg_idx[idx]
        if 'behavior_labels' not in case_map[(case_name, mode)]['turns'][i]:
            case_map[(case_name, mode)]['turns'][i]['behavior_labels'] = {}
        case_map[(case_name, mode)]['turns'][i]['behavior_labels'][role] = label

# =========================
# 行为标签后处理模块
# =========================
def refine_behavior_labels(raw_labels):
    """
    对llm_batch_label返回的标签列表做优先级处理。
    优先保留'攻击/指责'和'回避/冷处理'，否则只保留第一个标签。
    """
    if not raw_labels or not isinstance(raw_labels, list):
        return []
    priority = ['攻击/指责', '回避/冷处理']
    for p in priority:
        if p in raw_labels:
            return [p]
    return [raw_labels[0]]

def llm_reclassify_unknown_behavior(messages, model="openai/gpt-4.1", cache_dir=None):
    cache_file = os.path.join(cache_dir, "llm_behavior_reclassify.json") if cache_dir else "llm_behavior_reclassify.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    client = LLMClient(api_type="openrouter", model_name=model)
    results = []
    uncached = [msg for msg in messages if msg not in cache]
    if uncached:
        for text in uncached:
            prompt = """你是心理学对话分析专家。请判断以下对话文本属于哪个类型（必须且只能选择一个）：\n1. 建设性沟通\n2. 回避/冷处理\n3. 攻击/指责\n4. 自我暴露/脆弱表达\n5. 情绪主诉\n6. 修复承诺\n\n对话文本：\n{text}\n\n请直接输出类型名称（比如\"建设性沟通\"），不要其他任何内容。必须从以上六个选项中选择一个，不能输出其他内容。""".format(text=text)
            valid_types = ["建设性沟通", "回避/冷处理", "攻击/指责", "自我暴露/脆弱表达", "情绪主诉", "修复承诺"]
            max_retries = 5
            for _ in range(max_retries):
                try:
                    response, _ = client.call(prompt, model=model, temperature=0.2, max_tokens=100)
                    label = response.strip().strip('"').strip()
                    if label in valid_types:
                        cache[text] = label
                        break
                    else:
                        print(f"LLM返回了无效的行为标签: {label}，重试...")
                except Exception as e:
                    print(f"LLM重分类失败，重试: {e}")
                    time.sleep(2)
            else:
                print(f"文本重分类失败多次，默认标记为'建设性沟通': {text[:50]}...")
                cache[text] = "建设性沟通"
        with open(cache_file, 'w') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    for msg in messages:
        results.append(cache.get(msg, "建设性沟通"))
    return results

def ensure_label_in_six_types(label):
    six_types = [
        "建设性沟通", "回避/冷处理", "攻击/指责", "自我暴露/脆弱表达", "情绪主诉", "修复承诺"
    ]
    return label if label in six_types else "未知"

# =========================
# 3. NVC技巧与冲突阶段 LLM批量标注模块
# =========================
def extract_turn_texts_improved(cases):
    """提取improved组每轮对话文本（拼接character_message和partner_message）"""
    turn_texts = []
    turn_idx = []
    for case in cases:
        for i, t in enumerate(case['improved']['turns']):
            text = (t.get('character_message', '') or '') + ' ' + (t.get('partner_message', '') or '')
            turn_texts.append(text.strip())
            turn_idx.append((case['case'], i))
    return turn_texts, turn_idx

def add_nvc_and_stage_to_cases(cases, stage_labels, turn_idx):
    """将对话阶段标注写回cases结构"""
    case_map = {c['case']: c['improved']['turns'] for c in cases}
    for idx, (case_name, i) in enumerate(turn_idx):
        case_map[case_name][i]['conflict_stage'] = stage_labels[idx]

def calculate_average_emotion_trend(cases):
    """计算improved组所有case的平均情绪分数趋势"""
    all_scores = []
    for case in cases:
        scores = [t.get('character_score') for t in case.get('improved', {}).get('turns', [])]
        all_scores.append(scores)

    if not all_scores:
        return [], []

    # 找到最长序列长度
    max_len = max(len(s) for s in all_scores)

    # 对齐并计算平均值
    padded_scores = []
    for scores in all_scores:
        padded = scores[:] # 复制列表
        # 向后用None填充到max_len
        padded.extend([None] * (max_len - len(padded)))
        padded_scores.append(padded)

    # 计算每列（每轮）的平均值，忽略None
    average_scores = []
    for i in range(max_len):
        col_scores = [s[i] for s in padded_scores if s[i] is not None]
        average_scores.append(np.mean(col_scores) if col_scores else None) # 如果该轮所有值都是None，平均值为None

    # 移除尾部连续的None，确定实际的最大轮次
    actual_max_len = max_len
    for i in range(max_len - 1, -1, -1):
        if average_scores[i] is None:
            actual_max_len = i
        else:
            break

    return list(range(1, actual_max_len + 1)), average_scores[:actual_max_len]

def analyze_stage_ranges(cases):
    """分析冲突阶段在各轮次的典型出现范围"""
    # 统计每个轮次各阶段出现的频率
    turn_stage_counts = defaultdict(Counter)
    max_turns = 0
    for case in cases:
        turns = case.get('improved', {}).get('turns', [])
        max_turns = max(max_turns, len(turns))
        for i, turn in enumerate(turns):
            stage = turn.get('conflict_stage', '未知')
            turn_stage_counts[i+1][stage] += 1 # 轮次从1开始

    # 确定每个阶段的典型起始和结束轮次 (这里简化为首次和最后一次出现频率>阈值)
    stage_ranges = {}
    total_cases = len(cases)
    # 调整阈值，避免某些阶段在少量case中出现就被认为是典型范围
    # 可以考虑至少2个case，或者10%的case
    stage_threshold_count = max(2, int(total_cases * 0.1))

    # 定义阶段顺序，确保图例和背景绘制顺序
    ordered_stages = ['引发阶段', '升温阶段', '高峰阶段', '缓和阶段', '修复阶段', '未知']

    for stage in ordered_stages:
        start_turn, end_turn = None, None
        for turn in range(1, max_turns + 1):
            if turn in turn_stage_counts and turn_stage_counts[turn][stage] >= stage_threshold_count:
                if start_turn is None:
                    start_turn = turn
                end_turn = turn # 持续更新结束轮次
        if start_turn is not None:
             stage_ranges[stage] = (start_turn, end_turn)

    return stage_ranges, max_turns

def collect_skill_trigger_impact_data(cases):
    """收集所有improved组case中，每个轮次触发技巧后的情绪变化数据"""
    trigger_data_points_raw = []
    for case in cases:
        improved_turns = case.get('improved', {}).get('turns', [])
        scores = [t.get('character_score') for t in improved_turns]

        for i in range(len(improved_turns)):
            current_score = scores[i]
            # 假设每个improved模式的轮次都触发了心理学技巧，
            # 计算到下一轮的情绪变化
            if current_score is not None and i + 1 < len(scores):
                 next_score = scores[i+1]
                 if next_score is not None:
                      emotional_change = next_score - current_score
                      trigger_data_points_raw.append({
                          'turn': i + 1, # 轮次从1开始
                          'raw_score_at_turn': current_score,
                          'change_to_next_turn': emotional_change,
                          'case_name': case.get('case_name', '未知') # 方便调试
                      })
    return trigger_data_points_raw

STAGE_LABELS_EN = {
    '引发阶段': 'Triggering',
    '升温阶段': 'Escalation',
    '高峰阶段': 'Peak',
    '缓和阶段': 'Cooling',
    '修复阶段': 'Repair'
}

def plot_nvc_skill_eventline_aggregated(cases, fig_save_dir=None):
    """
    绘制improved组聚合的NVC技巧、冲突阶段与情绪变化事件线
    (点表示每轮平均情绪分数变化，大小表示幅度，颜色表示方向，点在折线上)
    """
    print("Calculating average emotion trend...")
    avg_x, average_scores = calculate_average_emotion_trend(cases)
    print(f"Average scores calculated for turns: {len(avg_x)}")

    print("Analyzing stage ranges...")
    stage_ranges, max_turns_stages = analyze_stage_ranges(cases)
    print(f"Stage ranges analyzed: {stage_ranges}")

    # 计算每一轮的平均情绪分数变化
    diffs = np.diff(average_scores)
    diffs = np.append(diffs, np.nan)  # 补齐长度

    plt.figure(figsize=(14, 7))
    ax = plt.gca() # Get current axes

    # 新配色和顺序
    stage_colors = {
        '引发阶段': '#ff9800',   # 亮橙
        '升温阶段': '#e57373',   # 柔和红
        '高峰阶段': '#8e24aa',   # 紫色
        '缓和阶段': '#81c784',   # 温和绿
        '修复阶段': '#64b5f6',   # 温和蓝
    }
    # 按照用户要求的上色顺序，后画的覆盖前画的
    ordered_stages = ['引发阶段', '升温阶段', '高峰阶段', '缓和阶段', '修复阶段']
    for stage in ordered_stages:
        if stage in stage_ranges:
            start, end = stage_ranges[stage]
            plt.axvspan(start - 0.5, end + 0.5, ymin=0, ymax=1, color=stage_colors.get(stage, '#e0e0e0'), alpha=0.25, zorder=0)
    # 在色块上方标注英文名和轮次区间，分层避免重叠
    stage_label_ypos = {
        '引发阶段': 0.95,
        '升温阶段': 0.90,
        '高峰阶段': 0.85,
        '缓和阶段': 0.80,
        '修复阶段': 0.75
    }
    for stage in ordered_stages:
        if stage in stage_ranges:
            start, end = stage_ranges[stage]
            mid = (start + end) / 2
            stage_en = STAGE_LABELS_EN.get(stage, stage)
            label = f"{stage_en}\n({start}-{end})"
            ypos = stage_label_ypos.get(stage, 0.95)
            plt.text(mid, ypos, label, ha='center', va='top', fontsize=11, color=stage_colors[stage], alpha=0.85, transform=ax.get_xaxis_transform())

    # 绘制平均情绪分数折线
    if avg_x:
        plt.plot(avg_x, average_scores, marker='o', linestyle='-', color=COLOR_PALETTE[2], linewidth=2, label='Average Emotion Score', zorder=3)

    # 在折线上画点，颜色和大小反映diff
    point_colors = []
    for d in diffs:
        if np.isnan(d):
            point_colors.append(COLOR_PALETTE[7])  # 灰色
        elif d > 0.05:
            point_colors.append(COLOR_PALETTE[6])  # Green
        elif d < -0.05:
            point_colors.append(COLOR_PALETTE[3])  # Red
        else:
            point_colors.append(COLOR_PALETTE[7])  # Pink/灰色
    max_change_abs = np.nanmax(np.abs(diffs)) if np.any(~np.isnan(diffs)) else 1
    size_multiplier = 500 / max_change_abs if max_change_abs > 0 else 100
    point_sizes = [50 + abs(d) * size_multiplier if not np.isnan(d) else 50 for d in diffs]

    plt.scatter(avg_x, average_scores, s=point_sizes, c=point_colors, alpha=0.8, edgecolors='w', linewidth=0.5, zorder=4, label='Emotion Change')

    plt.xlabel('Dialogue Turn')
    plt.ylabel('Emotion Score')
    plt.title('Aggregated NVC Skill Eventline: Stages, Emotion Trend, and Change')
    plt.xticks(range(1, 16))
    plt.xlim(0.5, 15.5)
    plt.ylim(-5, 1.1)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # 图例
    from matplotlib.lines import Line2D
    legend_handles = []
    for stage in ordered_stages:
        if stage in stage_ranges:
            legend_handles.append(Line2D([0], [0], linestyle='none', marker='s', alpha=0.25, markersize=10, color=stage_colors[stage], label=STAGE_LABELS_EN.get(stage, stage)))
    legend_handles.append(Line2D([0], [0], marker='o', color=COLOR_PALETTE[6], label='+ Emotion Change', linestyle='None', markersize=8))
    legend_handles.append(Line2D([0], [0], marker='o', color=COLOR_PALETTE[3], label='- Emotion Change', linestyle='None', markersize=8))
    legend_handles.append(Line2D([0], [0], marker='o', color=COLOR_PALETTE[7], label='Little/No Change', linestyle='None', markersize=8))
    legend_handles.append(Line2D([0], [0], color=COLOR_PALETTE[2], label='Average Emotion Score', linewidth=2))
    plt.legend(handles=legend_handles, loc='best')

    plt.tight_layout()
    save_path = os.path.join(fig_save_dir, "aggregated_nvc_skill_emotion_change_eventline.png") if fig_save_dir else "aggregated_nvc_skill_emotion_change_eventline.png"
    plt.savefig(save_path)
    plt.show()

# =========================
# 4. 可视化与统计模块
# =========================

def plot_emotion_trends(cases, fig_save_dir=None):
    """情绪趋势均值+标准差阴影图"""
    all_default, all_improved = [], []
    for case in cases:
        default_scores = [t['character_score'] for t in case['default']['turns'] if t['character_score'] is not None]
        improved_scores = [t['character_score'] for t in case['improved']['turns'] if t['character_score'] is not None]
        if default_scores:
            all_default.append(default_scores)
        if improved_scores:
            all_improved.append(improved_scores)
    if not (all_default or all_improved):
        print("无有效情绪分数数据，跳过情绪趋势图。")
        return
    max_len = max([len(x) for x in all_default+all_improved])
    def pad(l): return l + [l[-1]]*(max_len-len(l)) if l else [None]*max_len
    plt.figure(figsize=(12, 7))
    # Default
    if all_default:
        arr = np.array([pad(x) for x in all_default])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = np.arange(1, max_len+1)
        plt.plot(x, mean, color=COLOR_PALETTE[0], linewidth=3, label='default average')
        plt.fill_between(x, mean-std, mean+std, color=COLOR_PALETTE[0], alpha=0.2, label='default std')

    # Improved
    if all_improved:
        arr = np.array([pad(x) for x in all_improved])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = np.arange(1, max_len+1)
        plt.plot(x, mean, color=COLOR_PALETTE[6], linewidth=3, label='improved average')
        plt.fill_between(x, mean-std, mean+std, color=COLOR_PALETTE[6], alpha=0.2, label='improved std')
    plt.xlabel('dialogue turn')
    plt.ylabel('emotion score')
    plt.title('emotion change trend comparison (mean ± std)')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(fig_save_dir, 'emotion_trend.png') if fig_save_dir else 'emotion_trend.png'
    plt.savefig(save_path)
    plt.show()

def plot_emotion_improve_rate(case_stat, fig_save_dir=None):
    improved_cnt, default_cnt, total = 0, 0, 0
    for case, stat in case_stat.items():
        if 'default_ini' in stat and 'default_fin' in stat:
            if stat['default_fin'] is not None and stat['default_ini'] is not None and stat['default_fin'] > stat['default_ini'] + 1e-6:
                default_cnt += 1
        if 'improved_ini' in stat and 'improved_fin' in stat:
            if stat['improved_fin'] is not None and stat['improved_ini'] is not None and stat['improved_fin'] > stat['improved_ini'] + 1e-6:
                improved_cnt += 1
        if 'default_ini' in stat and 'default_fin' in stat and 'improved_ini' in stat and 'improved_fin' in stat:
             total += 1
    default_rate = default_cnt / total if total > 0 else 0
    improved_rate = improved_cnt / total if total > 0 else 0
    plt.figure(figsize=(5, 5))
    plt.bar(['default', 'improved'], [default_rate, improved_rate], color=[COLOR_PALETTE[0], COLOR_PALETTE[6]])
    plt.ylabel("Emotion Improvement Rate")
    plt.title("Emotion Improvement Rate Comparison")
    plt.tight_layout()
    save_path = os.path.join(fig_save_dir, "emotion_improve_rate.png") if fig_save_dir else "emotion_improve_rate.png"
    plt.savefig(save_path)
    plt.show()

def count_sentiment_words(dialogues):
    pos, neg = 0, 0
    for msg in dialogues:
        for w in POS_WORDS:
            if w in msg: pos += 1
        for w in NEG_WORDS:
            if w in msg: neg += 1
    return pos, neg

def llm_batch_translate_to_english(texts, batch_size=10, model="openai/gpt-4.1", cache_file="llm_translate_en.json"):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    client = LLMClient(api_type="openrouter", model_name=model)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        uncached = [txt for txt in batch if txt not in cache]
        if uncached:
            prompt = (
                "Please translate the following Chinese sentences into fluent English. "
                "Only output a JSON list of the translations, in order, no explanations.\n"
                f"{json.dumps(uncached, ensure_ascii=False)}"
            )
            for _ in range(3):
                try:
                    response, _ = client.call(prompt, model=model, temperature=0.2, max_tokens=1500)
                    json_str = response[response.find('['):response.rfind(']')+1]
                    batch_result = json.loads(json_str)
                    for zh, en in zip(uncached, batch_result):
                        cache[zh] = en
                    break
                except Exception as e:
                    print(f"LLM翻译失败，重试: {e}")
                    time.sleep(2)
            else:
                for zh in uncached:
                    cache[zh] = zh  # fallback: 保留原文
            with open(cache_file, 'w') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        for txt in batch:
            results.append(cache.get(txt, txt))
    return results

def plot_sentiment_word_ratio(cases, fig_save_dir=None):
    default_dialogues, improved_dialogues = [], []
    for case in cases:
        for mode, group in [('default', default_dialogues), ('improved', improved_dialogues)]:
            for t in case[mode]['turns']:
                if t.get('character_message'): group.append(t['character_message'])
                if t.get('partner_message'): group.append(t['partner_message'])
    default_pos, default_neg = count_sentiment_words(default_dialogues)
    improved_pos, improved_neg = count_sentiment_words(improved_dialogues)
    plt.figure(figsize=(7, 5))
    plt.bar(['default positive', 'default negative', 'improved positive', 'improved negative'],
        [default_pos, default_neg, improved_pos, improved_neg],
                color=[COLOR_PALETTE[0], COLOR_PALETTE[3], COLOR_PALETTE[6], COLOR_PALETTE[4]])
    plt.title("Sentiment Word Ratio Comparison")
    plt.tight_layout()
    save_path = os.path.join(fig_save_dir, "sentiment_word_ratio.png") if fig_save_dir else "sentiment_word_ratio.png"
    plt.savefig(save_path)
    plt.show()
    # 词云
    for group, dialogues in [('default', default_dialogues), ('improved', improved_dialogues)]:
        translated_dialogues = llm_batch_translate_to_english(dialogues)
        text = ' '.join(translated_dialogues)
        wc = WordCloud(font_path=None, width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{group} wordcloud")
        plt.tight_layout()
        save_path = os.path.join(fig_save_dir, f"{group}_wordcloud.png") if fig_save_dir else f"{group}_wordcloud.png"
        plt.savefig(save_path)
        plt.show()
        plt.close()

def plot_behavior_type_bar(cases, fig_save_dir=None):
    behavior_counter = {'default': Counter(), 'improved': Counter()}
    total_count = {'default': 0, 'improved': 0}
    for case in cases:
        for mode in ['default', 'improved']:
            for t in case[mode]['turns']:
                labels_dict = t.get('behavior_labels', {})
                for role in ['character_message', 'partner_message']:
                    labels = labels_dict.get(role, [])
                    for tag in labels:
                        behavior_counter[mode][tag] += 1
                        total_count[mode] += 1
    df_behavior = pd.DataFrame([
        {
            'Behavior Type': BEHAVIOR_LABELS_EN.get(k, k),
            'Proportion': v / total_count[mode] if total_count[mode] > 0 else 0,
            'Group': mode
        }
        for mode in ['default', 'improved']
        for k, v in behavior_counter[mode].items()
        if v > 0
    ])
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_behavior, x='Behavior Type', y='Proportion', hue='Group',
                     palette=[COLOR_PALETTE[0], COLOR_PALETTE[6]])
    plt.title('Behavior Type Proportion Comparison')
    plt.yscale('log')
    plt.ylim(0.0001, 1)
    plt.xticks(rotation=20, ha='right')
    for container in ax.containers:
        labels = [f'{v*100:.1f}%' for v in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(fig_save_dir, "behavior_type_bar.png") if fig_save_dir else "behavior_type_bar.png"
    plt.savefig(save_path)
    plt.show()

def plot_skill_trigger_scatter(cases, fig_save_dir=None):
    trigger_points = []
    for case in cases:
        turn_labels = defaultdict(set)
        for i, t in enumerate(case['improved']['turns']):
            labels_dict = t.get('behavior_labels', {})
            for role in ['character_message', 'partner_message']:
                labels = labels_dict.get(role, [])
                turn_labels[i].update(labels)
        for turn in sorted(turn_labels):
            if any(t in turn_labels[turn] for t in ['建设性沟通', '自我暴露/脆弱表达']):
                trigger_points.append({'case': case['case'], 'turn': turn+1})
                break
    if trigger_points:
        plt.figure(figsize=(10, 6))
        plt.scatter([x['turn'] for x in trigger_points], 
                   range(len(trigger_points)), 
                   c=COLOR_PALETTE[4])
        plt.xlabel('技巧首次触发轮次')
        plt.ylabel('case编号')
        plt.title('improved组心理学技巧触发点分布 (首次)')
        case_names = [p['case'] for p in trigger_points]
        plt.yticks(range(len(trigger_points)), case_names)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = os.path.join(fig_save_dir, "skill_trigger_scatter.png") if fig_save_dir else "skill_trigger_scatter.png"
        plt.savefig(save_path)
        plt.show()
    else:
        print("无符合条件的技巧触发点，跳过绘制散点图。")

def llm_reclassify_unknown_stage(turn_texts, model="openai/gpt-4.1", cache_dir=None):
    cache_file = os.path.join(cache_dir, "llm_stage_reclassify.json") if cache_dir else "llm_stage_reclassify.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    
    client = LLMClient(api_type="openrouter", model_name=model)
    results = []
    uncached = [txt for txt in turn_texts if txt not in cache]
    
    if uncached:
        # 每次处理一条文本，避免批量处理时的JSON解析错误
        for text in uncached:
            prompt = """你是对话冲突阶段分析专家。请判断以下对话文本属于哪个阶段（必须且只能选择一个）：
1. 引发阶段：冲突或分歧刚刚出现
2. 升温阶段：情绪或矛盾加剧
3. 高峰阶段：冲突最激烈，情绪最强烈
4. 缓和阶段：情绪缓和，冲突减弱
5. 表达与聆听阶段：双方开始倾听和表达
6. 修复阶段：尝试修复关系，重建连接

对话文本：
{text}

请直接输出阶段名称（比如"引发阶段"），不要其他任何内容。必须从以上六个选项中选择一个，不能输出其他内容。""".format(text=text)
            
            valid_stages = ["引发阶段", "升温阶段", "高峰阶段", "缓和阶段", "表达与聆听阶段", "修复阶段"]
            max_retries = 5  # 增加重试次数，确保得到有效标签
            
            for _ in range(max_retries):
                try:
                    response, _ = client.call(prompt, model=model, temperature=0.2, max_tokens=100)
                    stage = response.strip().strip('"').strip()
                    if stage in valid_stages:
                        cache[text] = stage
                        break
                    else:
                        print(f"LLM返回了无效的阶段标签: {stage}，重试...")
                except Exception as e:
                    print(f"LLM重分类失败，重试: {e}")
                    time.sleep(2)
            else:
                # 如果多次重试后仍未得到有效标签，选择最安全的标签
                print(f"文本重分类失败多次，默认标记为'引发阶段': {text[:50]}...")
                cache[text] = "引发阶段"
                
        # 保存新的缓存
        with open(cache_file, 'w') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            
    for txt in turn_texts:
        results.append(cache.get(txt, "引发阶段"))  # 默认使用最安全的标签
    return results

def ensure_stage_in_six_types(stage):
    """
    确保阶段标签在六大类型之内
    """
    six_types = [
        "引发阶段", "升温阶段", "高峰阶段", 
        "缓和阶段", "表达与聆听阶段", "修复阶段"
    ]
    return stage if stage in six_types else "未知"

def classify_emotion(emotion_en):
    """
    将情感分类为正面、负面和中性
    """
    negative_emotions = {
        'Anxiety', 'Depression', 'Dissatisfaction', 'Disappointment', 'Loss', 
        'Grievance', 'Fear', 'Fear & Despair', 'Anger', 'Frustration', 
        'Irritation', 'Despair', 'Emptiness', 'Sad', 'Shocked & Hurt',
        'Worry', 'Melancholy', 'Doubt', 'Trouble', 'Helplessness', 'Boredom'
    }
    
    positive_emotions = {
        'Joy', 'Happiness', 'Pleasure', 'Sweet', 'Peace of Mind', 'Comfort',
        'Reassurance', 'Calm', 'Warmth', 'Satisfaction', 'Content', 'Understanding',
        'Being Understood', 'Relief', 'Positive', 'Happy', 'Excitement', 'Care',
        'Affection', 'Gratitude', 'Hope', 'Relaxation', 'Soothing', 'Determination'
    }
    
    if emotion_en in negative_emotions:
        return 'negative'
    elif emotion_en in positive_emotions:
        return 'positive'
    else:
        return 'neutral'

def color_func_factory(emotion_type):
    """
    根据情感类型创建颜色函数
    """
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        emotion_class = classify_emotion(word)
        if emotion_class == 'negative':
            # 负面情感使用红色系
            return f"hsl(0, {random_state.randint(60, 100)}%, {random_state.randint(40, 60)}%)"
        elif emotion_class == 'positive':
            # 正面情感使用绿色系
            return f"hsl(120, {random_state.randint(30, 70)}%, {random_state.randint(30, 50)}%)"
        else:
            # 中性情感使用灰色系
            return f"hsl(0, 0%, {random_state.randint(60, 80)}%)"
    return color_func

def plot_emotion_type_wordclouds(cases, fig_save_dir=None):
    """
    为default组和improved组分别生成情感类型词云，
    使用不同颜色表示不同性质的情感
    """
    emotion_counter = {'default': Counter(), 'improved': Counter()}
    total_emotions = {'default': 0, 'improved': 0}
    
    # 统计情感类型频率
    for case in cases:
        for mode in ['default', 'improved']:
            for t in case[mode]['turns']:
                emotions = t.get('emotions', [])
                for emo in emotions:
                    emotion_counter[mode][emo] += 1
                    total_emotions[mode] += 1
    
    # 创建词频字典（使用英文，并计算比例）
    emotion_freq = {'default': {}, 'improved': {}}
    for mode in ['default', 'improved']:
        if total_emotions[mode] > 0:
            emotion_freq[mode] = {
                EMOTION_TYPES_EN.get(emo, emo): count / total_emotions[mode]
                for emo, count in emotion_counter[mode].items()
            }
    
    # 分别为两组生成词云
    for mode, title in [('default', 'Default Group'), ('improved', 'Improved Group')]:
        if emotion_freq[mode]:
            plt.figure(figsize=(12, 8))
            
            wc = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                prefer_horizontal=0.6,  # 减小水平偏好，让垂直词更多
                min_font_size=10,  # 稍微增加最小字号
                max_font_size=60,  # 减小最大字号，使差异不那么大
                color_func=color_func_factory(mode),
                random_state=42,
                margin=1,  # 进一步减小词间距
                relative_scaling=0.3,  # 减小相对缩放，使大小差异更小
                collocations=False,
                repeat=True,  # 允许重复词以填满空间
                scale=2  # 增加整体密度
            ).generate_from_frequencies(emotion_freq[mode])
            
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Emotion Distribution - {title}', fontsize=14, pad=20)
            
            # 添加图例，使用十六进制颜色代码
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc='#ff4444', label='Negative'),  # 鲜艳的红色
                plt.Rectangle((0, 0), 1, 1, fc='#4caf50', label='Positive'),  # 温和的绿色
                plt.Rectangle((0, 0), 1, 1, fc='#9e9e9e', label='Neutral')   # 中性灰色
            ]
            plt.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.1, 1.1))
            
            plt.tight_layout()
            save_path = os.path.join(fig_save_dir, f'emotion_wordcloud_{mode}.png') if fig_save_dir else f'emotion_wordcloud_{mode}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

def plot_behavior_distribution_by_turn(cases, fig_save_dir=None):
    """
    绘制行为类型随对话轮次的分布堆叠面积图，
    主图顶部用错开文字块展示矛盾阶段（按轮次区间，横向排开，无色带），
    下方显示每轮对话总数（最大y轴为65）。
    面积图配色用COLOR_PALETTE，柱状图为灰色。
    面积图左右两端填满。
    """
    # 统计每轮次的行为类型
    behavior_by_turn = {'default': defaultdict(lambda: defaultdict(int)), 
                       'improved': defaultdict(lambda: defaultdict(int))}
    turn_totals = {'default': defaultdict(int), 'improved': defaultdict(int)}
    case_counts = {'default': defaultdict(int), 'improved': defaultdict(int)}
    max_turn = 0
    
    # 收集数据
    for case in cases:
        for mode in ['default', 'improved']:
            for i, turn in enumerate(case[mode]['turns']):
                turn_num = i + 1
                max_turn = max(max_turn, turn_num)
                case_counts[mode][turn_num] += 1  # 记录每轮的case数量
                labels_dict = turn.get('behavior_labels', {})
                for role in ['character_message', 'partner_message']:
                    labels = labels_dict.get(role, [])
                    for label in labels:
                        behavior_by_turn[mode][turn_num][label] += 1
                        turn_totals[mode][turn_num] += 1

    # 获取矛盾阶段区间
    stage_ranges, _ = analyze_stage_ranges(cases)
    ordered_stages = ['引发阶段', '修复阶段', '缓和阶段', '高峰阶段', '升温阶段']
    # 文字纵向错开y坐标
    y_texts = [0.85, 0.88, 0.91, 0.94, 0.95]

    for mode in ['default', 'improved']:
        # 创建具有两个子图的画布，比例为8:2
        fig = plt.figure(figsize=(15, 9))
        gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.15)
        
        # 上方子图：行为分布堆叠图
        ax1 = plt.subplot(gs[0])
        
        # 准备堆叠面积图数据（左右两端填满）
        behaviors = list(BEHAVIOR_LABELS_EN.keys())
        x = [0.5] + list(range(1, max_turn + 1)) + [max_turn + 0.5]
        y_data = []
        for behavior in behaviors:
            proportions = []
            for turn in range(1, max_turn + 1):
                total = turn_totals[mode][turn] or 1
                count = behavior_by_turn[mode][turn][behavior]
                proportions.append(count / total)
            # 首尾补首尾值
            y_data.append([proportions[0]] + proportions + [proportions[-1]])
        # 使用COLOR_PALETTE循环配色
        n_colors = len(COLOR_PALETTE)
        area_colors = [COLOR_PALETTE[i % n_colors] for i in range(len(behaviors))]
        ax1.stackplot(x, y_data, 
                     labels=[BEHAVIOR_LABELS_EN[b] for b in behaviors],
                     alpha=0.7, colors=area_colors)

        # 主图顶部只保留错开文字块展示矛盾阶段（无色带）
        for idx, stage in enumerate(ordered_stages):
            if stage in stage_ranges:
                start, end = stage_ranges[stage]
                stage_en = STAGE_LABELS_EN.get(stage, stage)
                # 居中标注，纵向错开
                mid = (start + end) / 2
                label = f"{stage_en} ({start}-{end})"
                y_text = y_texts[idx % len(y_texts)]
                ax1.text(mid, y_text, label, ha='center', va='center', fontsize=10, color='black', alpha=0.95, transform=ax1.get_xaxis_transform(), zorder=6)

        ax1.set_ylabel('Behavior Type Proportion')
        ax1.set_title(f'Behavior Distribution by Turn ({mode.capitalize()} Group)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.5, max_turn + 0.5)
        ax1.set_ylim(0, 1)
        
        # 下方子图：对话数量柱状图
        ax2 = plt.subplot(gs[1])
        x_bar = list(range(1, max_turn + 1))
        counts = [case_counts[mode][turn] for turn in x_bar]
        bars = ax2.bar(x_bar, counts, alpha=0.8, color='gray')
        
        # 在柱子上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        ax2.set_xlabel('Dialogue Turn')
        ax2.set_ylabel('Dialogue Count')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xlim(0.5, max_turn + 0.5)
        ax2.set_ylim(0, 65)
        
        plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.18)
        save_path = os.path.join(fig_save_dir, f'behavior_distribution_{mode}.png') if fig_save_dir else f'behavior_distribution_{mode}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

def plot_behavior_transition_matrix(cases, mode='improved', fig_save_dir=None):
    """
    绘制行为类型转移概率热力图（默认improved组，可选default组），图片保存到fig_save_dir
    """
    import numpy as np
    import seaborn as sns
    behaviors = list(BEHAVIOR_LABELS_EN.keys())
    n = len(behaviors)
    trans_counts = np.zeros((n, n), dtype=int)
    for case in cases:
        turns = case[mode]['turns']
        prev_label = None
        for t in turns:
            labels = t.get('behavior_labels', {})
            curr = None
            for label in labels.get('character_message', []):
                if label in behaviors:
                    curr = label
                    break
            if prev_label is not None and curr is not None:
                i = behaviors.index(prev_label)
                j = behaviors.index(curr)
                trans_counts[i, j] += 1
            prev_label = curr
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = np.divide(trans_counts, row_sums, where=row_sums!=0)
    behavior_en = [BEHAVIOR_LABELS_EN[b] for b in behaviors]
    plt.figure(figsize=(10, 8))
    sns.heatmap(trans_probs, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=behavior_en, yticklabels=behavior_en, cbar_kws={'label': 'Transition Probability'})
    plt.xlabel('Next Behavior Type')
    plt.ylabel('Previous Behavior Type')
    plt.title(f'Behavior Transition Matrix ({mode.capitalize()} Group)')
    plt.tight_layout()
    if fig_save_dir is not None:
        os.makedirs(fig_save_dir, exist_ok=True)
        save_path = os.path.join(fig_save_dir, f'behavior_transition_matrix_{mode}.png')
    else:
        save_path = f'behavior_transition_matrix_{mode}.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

def llm_batch_conflict_stage(turn_texts, batch_size=10, model="openai/gpt-4.1", cache_dir=None):
    """
    用LLM批量标注每轮对话属于的矛盾阶段（6类），带缓存，缓存目录可控。
    """
    import json, os, time
    cache_file = os.path.join(cache_dir, "llm_conflict_stage.json") if cache_dir else "llm_conflict_stage.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    client = LLMClient(api_type="openrouter", model_name=model)
    results = []
    for i in range(0, len(turn_texts), batch_size):
        batch = turn_texts[i:i+batch_size]
        uncached = [txt for txt in batch if txt not in cache]
        if uncached:
            prompt = (
                "你是对话冲突阶段分析专家。请判断每条对话文本属于以下哪个阶段（单选）：\n"
                "1. 引发阶段：冲突或分歧刚刚出现\n"
                "2. 升温阶段：情绪或矛盾加剧\n"
                "3. 高峰阶段：冲突最激烈，情绪最强烈\n"
                "4. 缓和阶段：情绪缓和，冲突减弱\n"
                "5. 表达与聆听阶段：双方开始倾听和表达\n"
                "6. 修复阶段：尝试修复关系，重建连接\n"
                "请严格输出如下JSON格式：\n"
                "[{\"text\": \"对话文本1\", \"stage\": \"高峰阶段\"}, ...]\n"
                "只输出JSON，不要解释。\n"
                "待判别文本如下：\n"
                f"{json.dumps(uncached, ensure_ascii=False)}"
            )
            for _ in range(3):
                try:
                    response, _ = client.call(prompt, model=model, temperature=0.2, max_tokens=1500)
                    json_str = response[response.find('['):response.rfind(']')+1]
                    batch_result = json.loads(json_str)
                    for item in batch_result:
                        cache[item["text"]] = item["stage"]
                    break
                except Exception as e:
                    print(f"LLM标注阶段失败，重试: {e}")
                    time.sleep(2)
            else:
                for txt in uncached:
                    cache[txt] = "未知"
            with open(cache_file, 'w') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        for txt in batch:
            results.append(cache.get(txt, "未知"))
    return results

# =========================
# 主流程
# =========================

def main():
    # 目录集中定义
    log_dir = "log-gemini_gentle"
    result_dir = "benchmark_result_gemini_gentle"
    fig_save_dir = "figures_gemini_gentle"
    cache_dir = "cache_gemini_gentle"
    os.makedirs(fig_save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("1. 读取并对齐所有case...")
    cases = align_cases(log_dir)
    print(f"共{len(cases)}组case。")

    print("2. LLM批量标注行为类型...")
    messages, msg_idx = extract_all_messages(cases)
    print(f"共{len(messages)}条消息，开始LLM批量判别行为类型...")
    labels = llm_batch_label(messages, batch_size=10, model="openai/gpt-4.1", cache_dir=cache_dir)
    print("标签后处理...")
    refined_labels = [refine_behavior_labels(l) for l in labels]
    
    # 检查并重分类"未知"标签
    unknown_idx = [i for i, label in enumerate(refined_labels) if not label or label[0] == "未知"]
    if unknown_idx:
        unknown_msgs = [messages[i] for i in unknown_idx]
        reclassified = llm_reclassify_unknown_behavior(unknown_msgs, cache_dir=cache_dir)
        for idx, new_label in zip(unknown_idx, reclassified):
            refined_labels[idx] = [new_label]
    
    add_behavior_labels_to_cases(cases, refined_labels, msg_idx)

    print("3. 读取benchmark结果...")
    case_stat = read_benchmark_result(result_dir)

    print("4. LLM批量标注对话阶段...")
    turn_texts, turn_idx = extract_turn_texts_improved(cases)
    stage_labels = llm_batch_conflict_stage(turn_texts, cache_dir=cache_dir)
    
    # 检查并重分类"未知"阶段
    unknown_idx = [i for i, label in enumerate(stage_labels) if label == "未知"]
    if unknown_idx:
        unknown_texts = [turn_texts[i] for i in unknown_idx]
        reclassified = llm_reclassify_unknown_stage(unknown_texts, cache_dir=cache_dir)
        for idx, new_label in zip(unknown_idx, reclassified):
            stage_labels[idx] = new_label
    
    add_nvc_and_stage_to_cases(cases, stage_labels, turn_idx)
    print("5. 绘制情绪改善率柱状图...")
    plot_emotion_improve_rate(case_stat, fig_save_dir=fig_save_dir)

    print("6. 绘制情绪趋势折线图...")
    plot_emotion_trends(cases, fig_save_dir=fig_save_dir)

    # print("7. 绘制积极/消极用语比例及词云...")
    # plot_sentiment_word_ratio(cases, fig_save_dir=fig_save_dir)

    print("8. 绘制行为类型分布柱状图...")
    plot_behavior_type_bar(cases, fig_save_dir=fig_save_dir)

    print("9. 绘制improved组技巧首次触发点散点图...")
    plot_skill_trigger_scatter(cases, fig_save_dir=fig_save_dir)

    print("10. 绘制聚合的NVC技巧、冲突阶段与情绪变化事件线...")
    plot_nvc_skill_eventline_aggregated(cases, fig_save_dir=fig_save_dir)

    # print("11. 绘制情感类型频率柱状图...")
    # plot_emotion_type_frequency_bar(cases, fig_save_dir=fig_save_dir)
    
    print("11. 生成情感类型词云...")
    plot_emotion_type_wordclouds(cases, fig_save_dir=fig_save_dir)

    print("12. 绘制行为类型随轮次的分布...")
    plot_behavior_distribution_by_turn(cases, fig_save_dir=fig_save_dir)

    print("13. 绘制行为类型转移矩阵（default组）...")
    plot_behavior_transition_matrix(cases, mode='default', fig_save_dir=fig_save_dir)

    print("分析完成，所有图表已保存。")

if __name__ == "__main__":
    main()