import json
import os
import time
import random  # 新增：用于随机选择任务
from tqdm import tqdm
from openai import OpenAI

# --- 1. 配置 ---
INPUT_FILE = '../data/processed/raw_chunks.jsonl'
OUTPUT_FILE = '../data/processed/domain_expert_sft.jsonl'

# 硅基流动 API
API_KEY = "sk-lrdpxzsnhsbckhjrzekbrtccomruhcwzyrlwbroqwojtwtsw"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 2. 定义多样化提示词池 (Diversity Task Pool) ---

# 任务 A: 复杂案情分析 (侧重推理)
PROMPT_CASE_ANALYSIS = """
你是一位资深律师。请阅读以下法条：
【法条】：{content}

请根据该法条，构造一个**疑难复杂的法律咨询案例**。
要求：
1. **User**: 描述一个包含多方冲突或模糊地带的具体案情。
2. **Assistant**: 必须使用 CoT (Chain of Thought) 模式回答。
   - 必须先输出 <思考过程>：分析法律关系、争议焦点、适用法条。
   - 再输出 <法律建议>：给出最终结论和操作建议。
3. **格式**: 返回 JSON，包含 "instruction" 和 "output" (必须包含思考过程和建议)。
"""

# 任务 B: 法律文书起草 (侧重生成)
PROMPT_DOCUMENT_DRAFTING = """
你是一位擅长文书写作的律师。请阅读以下法条：
【法条】：{content}

请根据该法条，构造一个**起草法律文书的需求**。
要求：
1. **User**: 要求起草一份基于该法条的文书（如合同条款、起诉状片段、律师函）。
2. **Assistant**: 
   - <思考过程>：构思文书的关键要素、风险点。
   - <文书正文>：输出正式、严谨的法律文书内容。
3. **格式**: 返回 JSON，包含 "instruction" 和 "output"。
"""

# 任务 C: 法律概念辨析 (侧重知识)
PROMPT_CONCEPT_EXPLAIN = """
你是一位法学教授。请阅读以下法条：
【法条】：{content}

请根据该法条，构造一个**小白用户提出的概念性问题**。
要求：
1. **User**: 询问该法条中某个难懂的法律概念或适用范围（例如：“这条法律里的XX是什么意思？”）。
2. **Assistant**: 
   - <思考过程>：拆解概念定义的内涵和外延。
   - <通俗解释>：用大白话+举例的方式解释给用户听。
3. **格式**: 返回 JSON，包含 "instruction" 和 "output"。
"""

# 任务权重配置 (实现数据多样性平衡)
# 60% 案情分析, 20% 文书起草, 20% 概念解释
TASK_POOL = [
    ("case_analysis", PROMPT_CASE_ANALYSIS, 0.6),
    ("doc_drafting", PROMPT_DOCUMENT_DRAFTING, 0.2),
    ("concept_explain", PROMPT_CONCEPT_EXPLAIN, 0.2)
]

# --- 3. 核心生成函数 ---
def generate_sft_data(chunk):
    content = chunk.get('content', '')
    if chunk.get('type') != 'legal_article':
        return None

    # [核心技术点] 数据多样性平衡：轮盘赌选择任务
    rand = random.random()
    cumulative_prob = 0
    selected_task_name = ""
    selected_prompt_tpl = ""
    
    for name, tpl, prob in TASK_POOL:
        cumulative_prob += prob
        if rand <= cumulative_prob:
            selected_task_name = name
            selected_prompt_tpl = tpl
            break
    
    # 填充提示词
    user_prompt = selected_prompt_tpl.format(content=content)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个构建法律专家数据集的助手。请务必输出合法的 JSON 格式，不要包含 Markdown 标记。"},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8, # 稍微调高温度以增加多样性
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        
        # 清洗 Markdown 标记 (如 ```json)
        if result.strip().startswith("```"):
            result = result.replace("```json", "").replace("```", "")
            
        data = json.loads(result)
        
        # [核心技术点] CoT 推理增强：强制格式化 Output
        # 我们检查模型返回的 Key，如果它是按要求返回了思考过程，我们就拼接
        # 如果模型直接返回了 output，我们就直接用
        
        raw_output = data.get("output", "")
        formatted_output = ""

        # 尝试智能解析（取决于模型怎么理解 prompt 中的 key）
        # 这里做一个通用的清洗逻辑，确保 CoT 显性化
        if isinstance(raw_output, dict):
            # 如果模型很听话，返回了 {"思考过程": "...", "结论": "..."}
            thought = raw_output.get("思考过程") or raw_output.get("thought") or raw_output.get("analysis") or ""
            answer = raw_output.get("法律建议") or raw_output.get("文书正文") or raw_output.get("通俗解释") or raw_output.get("conclusion") or ""
            
            formatted_output = f"#### 🧠 思考过程\n{thought}\n\n#### 📝 专家建议\n{answer}"
        elif isinstance(raw_output, str):
            # 如果模型直接返回了字符串，我们假设它已经包含了 CoT（因为 Prompt 要求了）
            formatted_output = raw_output
        else:
            formatted_output = str(raw_output)

        return {
            "instruction": data.get("instruction", ""),
            "input": "", 
            "output": formatted_output,
            "task_type": selected_task_name, # 记录任务类型，方便后续分析分布
            "source_doc": chunk.get('source', 'unknown'),
            "domain": "legal"
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# --- 4. 主程序 ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到文件: {INPUT_FILE}")
        return

    print(f"🚀 开始生成 SFT 数据 (多样性增强版)...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # 过滤只取法律条文
        raw_chunks = [json.loads(line) for line in f if json.loads(line).get('type') == 'legal_article']
    
    print(f"📄 共加载 {len(raw_chunks)} 条法条片段。")

    raw_chunks = raw_chunks[:10]

    generated_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for chunk in tqdm(raw_chunks):
            sft_item = generate_sft_data(chunk)
            if sft_item:
                f_out.write(json.dumps(sft_item, ensure_ascii=False) + '\n')
                f_out.flush()
                generated_count += 1
            # 避免并发过高
            time.sleep(0.1)

    print(f"\n✅ 生成完毕！共 {generated_count} 条。")
    print(f"   包含了：案情分析(60%)、文书起草(20%)、概念解释(20%)")
    print(f"   已实现显式 CoT 格式化存储。")

if __name__ == "__main__":
    main()