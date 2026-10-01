from llama_cpp import Llama
import json

# 1. 加载 GGUF 模型 (精确指向你下载的 q6_k 权重文件)
MODEL_PATH = r"D:\newllm\models\qwen--Qwen2.5-3B-Instruct-GGUF\snapshots\master\qwen2.5-3b-instruct-q6_k.gguf"

# 初始化模型引擎
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1, # -1 表示将所有计算层完全卸载到 RTX 4070 显存中
    n_ctx=1024,      # 路由任务只需短上下文，缩小 KV Cache 极大地节省显存开销
    verbose=False    # 关闭底层 C++ 运行日志，保持控制台整洁
)

# 2. 定义神经计算边界提示词
SYSTEM_PROMPT = """你是一个高精度的神经符号路由器（Neuro-Symbolic Router）。
你的唯一任务是分析用户的自然语言输入，识别其意图，并提取所需参数。绝对不要进行计算。
可用意图：
1. calculate_matrix_determinant: 参数为 "matrix" (二维数组)
2. query_database_average: 参数为 "table_name" (表名), "column_name" (列名)"""

def route_intent(user_query: str) -> dict:
    """神经路由层：使用 JSON Schema 强制规范 LLM 输出"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    # 启用底层 response_format 约束，拦截幻觉
    response = llm.create_chat_completion(
        messages=messages,
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "parameters": {"type": "object"}
                },
                "required": ["intent", "parameters"]
            }
        },
        temperature=0.1,
        max_tokens=128
    )
    
    result_text = response["choices"][0]["message"]["content"]
    return json.loads(result_text)

def symbolic_engine(routed_data: dict):
    """符号执行层：添加防御性数据反序列化"""
    intent = routed_data.get("intent")
    params = routed_data.get("parameters", {})
    
    if intent == "calculate_matrix_determinant":
        import numpy as np
        import ast  # 引入抽象语法树解析库
        
        matrix_data = params.get("matrix", [])
        
        # 核心修复：如果 LLM 将数组幻觉成了字符串，强制将其安全解析为 Python 列表
        if isinstance(matrix_data, str):
            try:
                matrix_data = ast.literal_eval(matrix_data)
            except Exception as e:
                return f"符号引擎异常：无法将字符串解析为矩阵。错误详情: {e}"
                
        # 此时 matrix_data 已确保是 list 类型，可以安全送入计算引擎
        try:
            matrix = np.array(matrix_data, dtype=float) # 显式声明浮点类型防止精度异常
            result = np.linalg.det(matrix)
            return f"符号引擎计算结果 (行列式): {result}"
        except np.linalg.LinAlgError as e:
            return f"符号引擎计算拒绝：非方阵或维度错误。错误详情: {e}"

    elif intent == "query_database_average":
        table = params.get("table_name")
        col = params.get("column_name")
        return f"符号引擎执行 SQL: SELECT AVG({col}) FROM {table};"
    else:
        return "符号引擎抛出异常：无法匹配规则库。"

# 3. 运行基准测试
if __name__ == "__main__":
    print("正在启动神经符号混合引擎...\n")
    user_input = "帮我算一下这个二维矩阵的行列式：[[4, 2], [1, 3]]"
    
    print(f"[User Input] {user_input}")
    
    # 阶段 A：RTX 4070 意图抽取
    structured_intent = route_intent(user_input)
    print(f"\n[GPU LLM Router Output]\n{json.dumps(structured_intent, indent=2, ensure_ascii=False)}")
    
    # 阶段 B：CPU 符号计算
    final_result = symbolic_engine(structured_intent)
    print(f"\n[CPU Symbolic Engine Output]\n{final_result}")