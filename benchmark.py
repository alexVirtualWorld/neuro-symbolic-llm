from llama_cpp import Llama
import json
import time
import numpy as np
import ast

# 1. 初始化模型 (与之前保持一致)
MODEL_PATH = r"D:\newllm\models\qwen--Qwen2.5-3B-Instruct-GGUF\snapshots\master\qwen2.5-3b-instruct-q6_k.gguf"
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=1024, verbose=False)

SYSTEM_PROMPT = """你是一个高精度的神经符号路由器。识别自然语言意图，并提取所需参数。绝对不要计算。
可用意图：
1. calculate_matrix_determinant: 参数为 "matrix" (二维数组)
2. query_database_average: 参数为 "table_name" (表名), "column_name" (列名)"""

def route_intent(user_query: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_query}]
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        response_format={"type": "json_object", "schema": {"type": "object", "properties": {"intent": {"type": "string"}, "parameters": {"type": "object"}}, "required": ["intent", "parameters"]}},
        temperature=0.1, max_tokens=128
    )
    llm_latency = time.time() - start_time
    result_text = response["choices"][0]["message"]["content"]
    return json.loads(result_text), llm_latency

def symbolic_engine(routed_data: dict) -> tuple:
    intent = routed_data.get("intent")
    params = routed_data.get("parameters", {})
    start_time = time.time()
    
    try:
        if intent == "calculate_matrix_determinant":
            matrix_data = params.get("matrix", [])
            if isinstance(matrix_data, str):
                matrix_data = ast.literal_eval(matrix_data)
            matrix = np.array(matrix_data, dtype=float)
            result = np.linalg.det(matrix)
            engine_latency = time.time() - start_time
            return round(result, 4), engine_latency
        elif intent == "query_database_average":
            table = params.get("table_name")
            col = params.get("column_name")
            engine_latency = time.time() - start_time
            return f"SELECT AVG({col}) FROM {table};", engine_latency
        else:
            return "unsupported", time.time() - start_time
    except Exception:
        return "error", time.time() - start_time

# 2. 结构化测试集 (包含标准用例与干扰变体)
TEST_CASES = [
    {"query": "帮我算一下这个二维矩阵的行列式：[[4, 2], [1, 3]]", "expected_intent": "calculate_matrix_determinant", "expected_result": 10.0},
    {"query": "计算矩阵 [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 的行列式的值", "expected_intent": "calculate_matrix_determinant", "expected_result": 1.0},
    {"query": "在 users 表里面，把 age 这一列的平均值查出来", "expected_intent": "query_database_average", "expected_result": "SELECT AVG(age) FROM users;"},
    {"query": "请统计 sales_data 数据表中 revenue 字段的平均数", "expected_intent": "query_database_average", "expected_result": "SELECT AVG(revenue) FROM sales_data;"},
    {"query": "求解矩阵 [[5, 2], [-3, 4]] 的行列式", "expected_intent": "calculate_matrix_determinant", "expected_result": 26.0}
]

# 3. 执行自动化测试
if __name__ == "__main__":
    print(f"开始执行自动化基准测试，共 {len(TEST_CASES)} 个测试用例...\n")
    
    correct_routing = 0
    correct_execution = 0
    total_llm_time = 0
    total_engine_time = 0
    
    for i, test in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] 测试输入: {test['query']}")
        
        # 运行路由
        routed_data, llm_time = route_intent(test["query"])
        total_llm_time += llm_time
        
        is_route_correct = routed_data.get("intent") == test["expected_intent"]
        if is_route_correct:
            correct_routing += 1
            
        # 运行符号引擎
        engine_result, engine_time = symbolic_engine(routed_data)
        total_engine_time += engine_time
        
        is_exec_correct = engine_result == test["expected_result"]
        if is_exec_correct:
            correct_execution += 1
            
        print(f"  -> 路由耗时: {llm_time*1000:.2f}ms | 引擎耗时: {engine_time*1000:.2f}ms | 最终结果正确: {is_exec_correct}\n")

    # 4. 输出论文所需的汇总统计数据
    print("="*40)
    print("基准测试结果汇总 (论文数据源):")
    print(f"路由准确率: {correct_routing / len(TEST_CASES) * 100:.2f}%")
    print(f"端到端执行准确率: {correct_execution / len(TEST_CASES) * 100:.2f}%")
    print(f"平均 LLM 路由耗时: {(total_llm_time / len(TEST_CASES))*1000:.2f} ms")
    print(f"平均符号引擎计算耗时: {(total_engine_time / len(TEST_CASES))*1000:.2f} ms")
    print("="*40)