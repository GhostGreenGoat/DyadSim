# LQBench - Couple Dialogue Simulation Benchmark Framework

## Project Overview

LQBench is a benchmark framework for simulating couple dialogues. It is designed to simulate the interaction process and emotional changes of virtual characters with different personality traits, relationship beliefs, communication styles, and attachment types in various conflict scenarios. The framework uses Large Language Models (LLMs) for role-playing and supports testing various dialogue scenarios and analyzing results.

## System Architecture and Key Files

*   `benchmark_runner.py`: The core benchmark running script, used to configure and start the simulation process.
*   `character_simulator.py`: The virtual character simulator, managing the flow and emotional evaluation of a single dialogue.
*   `analyze_benchmark.py`: Used to analyze the benchmark test results generated after running `benchmark_runner.py`.
*   `config.json`: The project's configuration file, used to store API keys and other sensitive information (refer to `config.json.example`).
*   `requirements.txt`: List of project dependencies.
*   `api/`: Contains modules related to LLM interaction and data definitions.
    *   `api/llm.py`: LLM interface client.
    *   `api/data/`: Stores data definitions for various character traits, scenarios, prompt templates, etc.
        *   `api/data/prompt_templates.py`: Contains LLM prompt templates, including instructions for psychological techniques used in specific modes.

## Environment Setup

Before running the project, ensure you have Python 3 installed and set up the environment by following these steps:

1.  **Clone the Project Repository** (If not already done)
    ```bash
    git clone <Your Repository URL>
    cd LQBench
    ```
    *(Please replace `<Your Repository URL>` with the actual project repository URL)*

2.  **Install Dependencies**
    Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    Or use `pip3` if your system has both Python 2 and Python 3:
    ```bash
    pip3 install -r requirements.txt
    ```

3.  **Configure API Keys**
    Copy the `config.json.example` file and rename it to `config.json`.
    ```bash
    cp config.json.example config.json
    ```
    Edit the `config.json` file and enter your Large Language Model API keys (currently supports DeepSeek and OpenRouter). Please ensure your API keys are securely stored.

## Running Benchmark Tests

Use the `benchmark_runner.py` script to execute benchmark tests. The script supports various command-line arguments to control the test behavior, such as specifying the API, model, number of turns, number of characters, whether to run in parallel, etc.

### Running Modes (Modes)

`benchmark_runner.py` supports specifying different running modes via the `--modes` parameter. Each mode may affect the virtual characters' role-playing style or introduce specific behavioral logic.

1.  **Default Mode**
    This is the default role-playing mode for the benchmark test. In this mode, virtual characters will primarily play their roles based on the personality, relationship beliefs, communication styles, and attachment types defined in their configuration, conducting free dialogue simulations.

    You can run the default mode by not specifying the `--modes` parameter, or more explicitly by specifying `--modes default`.

    **Example Command (Running Default Mode):**

    ```bash
    python3 benchmark_runner.py --character-api deepseek --character-model deepseek-chat --partner-api deepseek --partner-model deepseek-chat --num-experts 1 --num_characters 2 --parallel
    ```
    *(This example uses DeepSeek API and model, sets the number of experts to 1, number of characters to 2, and enables parallelism)*

2.  **Improved Mode**
    In this mode, in addition to basic character settings, the framework will introduce psychological or communication techniques defined in `api/data/prompt_templates.py` to guide the virtual characters' role-playing. For example, the `improved` mode will attempt to use techniques like "Gentle Start-Up" to handle conflict dialogues.

    To run this mode, include `improved` in the `--modes` parameter. You can run the `improved` mode alone (`--modes improved`), or run both `default` and `improved` modes simultaneously (`--modes default improved`). The framework will simulate each scenario twice, once for each mode.

    **Example Command (Running both default and improved modes):**

    ```bash
    python3 benchmark_runner.py --modes default improved --character-api deepseek --character-model deepseek-chat --partner-api deepseek --partner-model deepseek-chat --num-experts 1 --num_characters 2 --parallel
    ```

### Other Common Parameters

*   `--output_dir`: Specifies the output directory for results (default: `benchmark_results`).
*   `--log_dir`: Specifies the output directory for logs (default: `logs`).
*   `--max_turns`: Maximum number of dialogue turns (default: 10).
*   `--character_api`, `--partner_api`, `--expert_apis`: Specifies the API types used by different roles.
*   `--character_model`, `--partner_model`, `--expert_model`: Specifies the model names used by different roles.
*   `--num_experts`: Number of experts (default: 3).
*   `--num_characters`: Number of protagonist characters when generating test cases (default: 5).
*   `--scenario_ids`: List of scenario IDs to run.
*   `--num_scenarios`: Number of scenarios to randomly select.
*   `--parallel`: Enables parallel execution.
*   `--max_workers`: Maximum number of workers when running in parallel (default: 4).
*   `--random_seed`: Sets the random seed to fix test case selection.

Please adjust the command-line parameters according to your needs.

## Result Analysis

After the benchmark test runs, the results will be saved in the directory specified by `--output_dir`. You can use the `analyze_benchmark.py` script to perform further analysis and visualization of these results.

```bash
python3 analyze_benchmark.py <Results File Path>
```
*(Please replace `<Results File Path>` with the actual path to the results file generated by benchmark_runner.py)*

For specific analysis functions and parameters, please refer to the `analyze_benchmark.py` code or run `python3 analyze_benchmark.py --help` to see the help information.

## New Feature Details

### 1. Test Model Emotion Prediction

This feature allows the test model to predict the virtual character's potential emotional state in the next turn based on the dialogue history after each turn.

#### Core Implementation

- **Prediction Timing**: After each dialogue turn, before the next turn starts.
- **Prediction Content**: Emotion type, intensity, score, and explanation.
- **Evaluation Method**: Compare the predicted results with the actual emotional state to calculate accuracy.

#### Usage Example

```python
# Enable emotion prediction feature
simulator = CharacterSimulator(
    character_config=character,
    scenario_id=scenario_id,
    use_emotion_prediction=True,
    partner_api="openrouter"  # Test model API
)

# Run simulation
result = simulator.run_simulation()

# View emotion prediction history
prediction_history = result['emotion_prediction_history']
for prediction in prediction_history:
    print(f"Turn {prediction['turn']}:")
    print(f"Predicted Emotion: {prediction['predicted_emotion']}")
    print(f"Intensity: {prediction['intensity']}")
    print(f"Emotion Score: {prediction['emotion_score']}")
    print(f"Prediction Explanation: {prediction['explanation']}")
    print("---")
```

### 2. Multi-Expert Real-time Emotion Analysis

This feature uses multiple expert models to simultaneously analyze the dialogue between the virtual character and the test model, evaluating the virtual character's emotional state in real-time.

#### Core Implementation

- **Analysis Timing**: Immediately after each dialogue turn.
- **Number of Experts**: Defaults to 3 expert models (configurable).
- **Analysis Content**: Primary emotion, intensity, score, key triggers, and brief analysis.
- **Consistency Calculation**: Evaluates the degree of consistency in the analysis results from multiple experts.

#### Usage Example

```python
# Enable multi-expert analysis feature
simulator = CharacterSimulator(
    character_config=character,
    scenario_id=scenario_id,
    use_expert_analysis=True,
    num_experts=3,
    expert_api="deepseek"  # Expert model API
)

# Run simulation
result = simulator.run_simulation()

# View expert analysis history
expert_history = result['expert_analysis_history']
for turn_analysis in expert_history:
    print(f"Turn {turn_analysis['turn']}:")
    for analysis in turn_analysis['analyses']:
        print(f"Expert {analysis['expert_id']}:")
        print(f"Primary Emotion: {analysis['primary_emotion']}")
        print(f"Intensity: {analysis['intensity']}")
        print(f"Emotion Score: {analysis['emotion_score']}")
        print(f"Key Triggers: {', '.join(analysis['key_triggers'])}")
        print(f"Brief Analysis: {analysis['analysis']}")
    print("---")
```

## Running Methods

### Basic Usage

```python
from LQBench.character_simulator import CharacterSimulator

# Create simulator
simulator = CharacterSimulator(
    character_api="deepseek",
    partner_api="openrouter",
    expert_api="deepseek",
    use_emotion_prediction=True,
    use_expert_analysis=True,
    num_experts=3
)

# Run simulation
result = simulator.run_simulation()

# Print results
print(f"Turns Completed: {result['turns_completed']}")
print(f"Final Emotion Score: {result['final_emotion_score']}")
```

### Command Line Execution

```bash
python -m LQBench.benchmark_runner --num-characters 3 --max-turns 8 --use-emotion-prediction --use-expert-analysis --num-experts 3
```

## Data Flow Diagram
