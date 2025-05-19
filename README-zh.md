# LQBench - 情侣对话模拟基准测试框架

## 项目概述

LQBench 是一个用于情侣对话模拟的基准测试框架。它旨在模拟具有不同性格特质、关系信念、沟通方式和依恋类型的虚拟人物在各种冲突场景下的交互过程和情绪变化。框架通过大型语言模型 (LLM) 实现角色扮演，支持多种对话场景测试和结果分析。

## 系统架构与关键文件

*   `benchmark_runner.py`: 基准测试的核心运行脚本，用于配置和启动模拟过程。
*   `character_simulator.py`: 虚拟人物模拟器，管理单个对话的流程和情绪评估。
*   `analyze_benchmark.py`: 用于分析 `benchmark_runner.py` 运行后产生的基准测试结果。
*   `config.json`: 项目的配置文件，用于存放 API 密钥等敏感信息（请参考 `config.json.example`）。
*   `requirements.txt`: 项目的依赖库列表。
*   `api/`: 存放与 LLM 交互及数据定义相关的模块。
    *   `api/llm.py`: LLM 接口客户端。
    *   `api/data/`: 存放各种人物特质、场景、提示词模板等数据定义。
        *   `api/data/prompt_templates.py`: 包含 LLM 提示词模板，包括用于特定模式的心理学技巧指令。

## 环境设置

在运行项目之前，请确保你已经安装了 Python 3，并按照以下步骤设置环境：

1.  **克隆项目仓库** (如果尚未完成)
    ```bash
    git clone <你的仓库地址>
    cd LQBench
    ```
    *(请将 `<你的仓库地址>` 替换为实际的项目仓库地址)*

2.  **安装依赖**
    使用 `requirements.txt` 文件安装项目所需的 Python 库：
    ```bash
    pip install -r requirements.txt
    ```
    或者使用 `pip3` 如果你的系统同时存在 Python 2 和 Python 3：
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **配置 API 密钥**
    复制 `config.json.example` 文件，并将其重命名为 `config.json`。
    ```bash
    cp config.json.example config.json
    ```
    编辑 `config.json` 文件，填入你的大语言模型 API 密钥（目前支持 DeepSeek 和 OpenRouter）。请确保妥善保管你的 API 密钥。

## 运行基准测试

使用 `benchmark_runner.py` 脚本来执行基准测试。该脚本支持多种命令行参数来控制测试的行为，例如指定使用的 API、模型、对话轮次、参与人物数量、是否并行等。

### 运行模式 (Modes)

`benchmark_runner.py` 支持通过 `--modes` 参数指定不同的运行模式。每种模式可能会影响虚拟人物的扮演方式或引入特定的行为逻辑。

1.  **普通模式 (Default Mode)**
    这是基准测试的默认扮演模式。在这种模式下，虚拟人物会主要根据其配置中定义的性格、关系信念、沟通方式和依恋类型来扮演角色，进行自由对话模拟。

    你可以通过不指定 `--modes` 参数来运行默认模式，或者更明确地通过 `--modes default` 来指定。

    **示例命令 (运行默认模式):**

    ```bash
    python3 benchmark_runner.py --character-api deepseek --character-model deepseek-chat --partner-api deepseek --partner-model deepseek-chat --num-experts 1 --num_characters 2 --parallel
    ```
    *(此示例使用了 DeepSeek API 和模型，设置专家数量为 1，人物数量为 2，并启用并行)*

2.  **技巧增强模式 (Improved Mode)**
    在这种模式下，除了基本的人物设定外，框架还会引入 `api/data/prompt_templates.py` 中定义的心理学或沟通技巧来指导虚拟人物的扮演。例如，`improved` 模式会尝试使用"温和启动（Gentle Start-Up）"等技巧来处理冲突对话。

    要运行此模式，请在 `--modes` 参数中包含 `improved`。你可以单独运行 `improved` 模式 (`--modes improved`)，或者同时运行 `default` 和 `improved` 模式 (`--modes default improved`)，框架会对每个场景分别进行这两种模式的模拟。

    **示例命令 (同时运行 default 和 improved 模式):**

    ```bash
    python3 benchmark_runner.py --modes default improved --character-api deepseek --character-model deepseek-chat --partner-api deepseek --partner-model deepseek-chat --num-experts 1 --num_characters 2 --parallel
    ```

### 其他常用参数

*   `--output_dir`: 指定结果输出目录（默认: `benchmark_results`）。
*   `--log_dir`: 指定日志输出目录（默认: `logs`）。
*   `--max_turns`: 最大对话轮次（默认: 10）。
*   `--character_api`, `--partner_api`, `--expert_apis`: 指定不同角色使用的 API 类型。
*   `--character_model`, `--partner_model`, `--expert_model`: 指定不同角色使用的模型名称。
*   `--num_experts`: 专家数量（默认: 3）。
*   `--num_characters`: 生成测试用例时的主角人物数量（默认: 5）。
*   `--scenario_ids`: 指定要运行的场景 ID 列表。
*   `--num_scenarios`: 随机选择的场景数量。
*   `--parallel`: 启用并行运行。
*   `--max_workers`: 并行运行时最大 worker 数量（默认: 4）。
*   `--random_seed`: 设置随机种子以固定测试用例的选择。

请根据你的需求调整命令行参数。

## 结果分析

基准测试运行完成后，结果会保存在 `--output_dir` 指定的目录下。你可以使用 `analyze_benchmark.py` 脚本来对这些结果进行进一步的分析和可视化。

```bash
python3 analyze_benchmark.py <结果文件路径>
```
*(请将 `<结果文件路径>` 替换为 benchmark_runner.py 生成的实际结果文件路径)*

具体分析功能和参数请参考 `analyze_benchmark.py` 的代码或运行 `python3 analyze_benchmark.py --help` 查看帮助信息。

## 新增功能详解

### 1. 待测模型情感预测

该功能允许待测模型在每轮对话后，基于历史对话预测虚拟人物在下一轮对话中可能的情感状态。

#### 核心实现

- **预测时机**：每轮对话结束后、下一轮开始前
- **预测内容**：情绪类型、强度、评分和解释
- **评估方法**：将预测结果与实际情绪状态对比，计算准确度

#### 使用示例

```python
# 启用情感预测功能
simulator = CharacterSimulator(
    character_config=character,
    scenario_id=scenario_id,
    use_emotion_prediction=True,
    partner_api="openrouter"  # 待测模型API
)

# 运行模拟
result = simulator.run_simulation()

# 查看情感预测历史
prediction_history = result['emotion_prediction_history']
for prediction in prediction_history:
    print(f"轮次 {prediction['turn']}:")
    print(f"预测情绪: {prediction['predicted_emotion']}")
    print(f"情绪强度: {prediction['intensity']}")
    print(f"情绪评分: {prediction['emotion_score']}")
    print(f"预测解释: {prediction['explanation']}")
    print("---")
```

### 2. 多专家实时情感分析

该功能使用多个专家模型同时分析虚拟人物与待测模型的对话，实时评估虚拟人物的情感状态。

#### 核心实现

- **分析时机**：每轮对话结束后立即进行
- **专家数量**：默认使用3个专家模型（可配置）
- **分析内容**：主要情绪、强度、评分、关键触发点和简要分析
- **一致性计算**：评估多个专家之间分析结果的一致程度

#### 使用示例

```python
# 启用多专家分析功能
simulator = CharacterSimulator(
    character_config=character,
    scenario_id=scenario_id,
    use_expert_analysis=True,
    num_experts=3,
    expert_api="deepseek"  # 专家模型API
)

# 运行模拟
result = simulator.run_simulation()

# 查看专家分析历史
expert_history = result['expert_analysis_history']
for turn_analysis in expert_history:
    print(f"轮次 {turn_analysis['turn']}:")
    for analysis in turn_analysis['analyses']:
        print(f"专家 {analysis['expert_id']}:")
        print(f"主要情绪: {analysis['primary_emotion']}")
        print(f"情绪强度: {analysis['intensity']}")
        print(f"情绪评分: {analysis['emotion_score']}")
        print(f"关键触发点: {', '.join(analysis['key_triggers'])}")
        print(f"简要分析: {analysis['analysis']}")
    print("---")
```

## 运行方法

### 基本使用

```python
from LQBench.character_simulator import CharacterSimulator

# 创建模拟器
simulator = CharacterSimulator(
    character_api="deepseek",
    partner_api="openrouter",
    expert_api="deepseek",
    use_emotion_prediction=True,
    use_expert_analysis=True,
    num_experts=3
)

# 运行模拟
result = simulator.run_simulation()

# 打印结果
print(f"对话轮次: {result['turns_completed']}")
print(f"最终情绪分数: {result['final_emotion_score']}")
```

### 命令行运行

```bash
python -m LQBench.benchmark_runner --num-characters 3 --max-turns 8 --use-emotion-prediction --use-expert-analysis --num-experts 3
```

## 数据流向图

```
角色特质 → 提示词生成 → LLM接口 → 对话生成 → 情绪解析 → 对话流程控制 → 结果分析
  |                                 ↑
  |                                 |
场景选择 ---------------------------|
```

## 结果解读

每次测试的结果包含：

1. **基本信息**：角色ID、名称、性格特质、场景等
2. **对话统计**：完成轮次、初始情绪分、最终情绪分、情绪变化
3. **情绪曲线**：每轮对话的情绪分数变化
4. **详细对话**：完整的对话历史和情绪变化记录

批量测试的结果包含：

1. **CSV报告**：所有测试的汇总统计数据
2. **性格对比图**：不同性格特质的情绪变化对比
3. **场景对比图**：不同场景下的情绪变化对比
4. **依恋类型图**：不同依恋类型的情绪变化对比

## 开发扩展

### 添加新性格特质

在`api/data/personality_types.py`中添加新的性格类型：

```python
personality_types.append({
    "id": "your_new_type_id",
    "name": "新性格类型名称",
    "description": "详细描述...",
    "interaction_style": "交互风格描述..."
})
```

### 添加新场景

在`api/data/conflict_scenarios.py`中添加新的冲突场景：

```python
conflict_scenarios.append({
    "id": "new_scenario_id",
    "name": "新场景名称",
    "description": "场景描述...",
    "situations": [
        {
            "id": "situation_1",
            "name": "情境1名称",
            "description": "情境描述...",
            "example": "情境示例...",
            "typical_triggers": ["触发因素1", "触发因素2"]
        }
    ]
})
```

### 自定义情绪评估

修改`api/data/emotions.py`中的情绪评分阈值：

```python
emotion_scoring = {
    "threshold": {
        "improvement": 5,  # 情绪好转阈值
        "worsening": -5,   # 情绪恶化阈值
        "critical": -8     # 临界阈值
    }
}
```

## 注意事项

1. 必须提供有效的API密钥才能运行测试
2. 大模型API请求可能产生费用，请控制测试规模
3. 测试结果可能因模型版本和参数而异
4. 情绪解析基于大模型输出，准确性有限
5. 大规模并行测试可能受API调用限制影响 

## 评估模块详解

### 5. 对话评估模块 (metrics.py)

metrics.py 文件提供了一套全面的对话质量评估和角色特征分析工具，用于评估和分析生成的对话日志。该模块可以分析对话中的沟通模式、依恋类型特征、情感变化等多个维度，生成详细的评估报告。

#### 主要类和功能

1. **ConsistencyEvaluator 类**：角色特征归类器
   - 分析对话内容，识别角色的沟通模式和依恋类型
   - 支持基于关键词的规则式分析和API辅助分析
   - 能够识别六种沟通类型：直接表达、间接表达、情感表达、理性分析、问题导向、解决导向
   - 能够识别四种依恋类型：安全型、焦虑型、回避型、混乱型
   - 检测违反预设规则的行为

2. **DialogueEvaluator 类**：对话质量评估器
   - 提供全面的对话统计数据：消息数量、长度、词汇多样性等
   - 分析情感变化：情感范围、波动性、变化轨迹
   - 评估主题连贯性和互动质量
   - 分析说话风格和表达清晰度
   - 评估积极/消极因素的平衡
   - 生成综合的一致性得分和质量得分

3. **批量分析功能**
   - `analyze_dialogue_log`：分析单个对话日志文件
   - `batch_analyze_logs`：批量分析目录中的所有对话日志
   - 汇总统计数据：平均得分、沟通类型分布、依恋类型分布等
   
#### 使用方法

metrics.py 可以通过命令行参数进行配置，支持以下功能：

```bash
# 评估单个对话日志文件
python metrics.py -f 日志文件路径 -o 输出目录

# 批量评估目录中的所有日志文件
python metrics.py -d 日志目录 -o 输出目录

# 启用API辅助分析（用于规则式方法无法判断的情况）
python metrics.py -d 日志目录 --use-api --api-type deepseek

# 使用配置文件
python metrics.py -c 配置文件.json
```

配置文件示例（JSON格式）：
```json
{
    "input_directory": "logs_test",
    "output_dir": "logs_test_eval",
    "output": "logs_test_eval",
    "use_api": true,
    "api_type": "deepseek",
    "model_name": "deepseek-chat",
    "debug": false
}
```

#### 项目文件夹说明

- **logs_test/**：包含对话示例文件，用于测试和分析
- **logs_test_eval/**：存放评估结果的目录，包含对logs_test中对话的分析结果
- **metrics.py**：对话评估工具，用于分析对话质量、沟通模式和依恋特征

#### 示例输出

评估结果以JSON格式存储，包含以下主要信息：
- 基本元数据：文件名、角色信息、场景信息
- 对话统计：总消息数、平均长度、词汇多样性等
- 情感统计：情感范围、波动性、主导情感等
- 一致性分析：主题连贯性、互动质量、说话风格等
- 质量评估：倾听清晰度、积极/消极因素平衡等
- 角色特征：沟通类型、依恋类型、违禁行为数量等
- 情感分布：各类情感的出现频率

#### 评估维度说明

1. **沟通类型**：基于对话中出现的关键词和表达方式，将沟通模式分为六种类型
2. **依恋类型**：基于依恋理论，从对话中识别安全型、焦虑型、回避型和混乱型的特征
3. **一致性得分**：0-10分，评估对话的连贯性、互动质量和风格一致性
4. **质量得分**：0-10分，综合评估对话的质量，包括词汇多样性、主题连贯性、倾听能力等

通过这些评估，可以深入了解不同人物特质在对话中的表现差异，为虚拟角色的设计和优化提供数据支持。 