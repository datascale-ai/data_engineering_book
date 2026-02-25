import os
import base64
from byaldi import RAGMultiModalModel
from openai import OpenAI

# === 配置区域 ===
# 1. 硅基流动 API 配置
API_KEY = "sk-lrdpxzsnhsbckhjrzekbrtccomruhcwzyrlwbroqwojtwtsw"  # 你的密钥
BASE_URL = "https://api.siliconflow.cn/v1"

# 使用 72B 模型以获得最佳的图表分析能力
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct" 

# 2. 本地索引配置
INDEX_NAME = "finance_report_2024" 

# 3. 检索配置 (关键修改)
# 增加检索页数，防止只命中目录页。建议 3-5 页。
RETRIEVAL_K = 4 

# 4. 强制离线设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 正在初始化...")
print(f"📡 连接云端模型: {MODEL_NAME}")

# === 初始化客户端 ===
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# === 加载本地检索器 ===
print(f"📂 正在加载本地索引: {INDEX_NAME} ...")
try:
    RAG = RAGMultiModalModel.from_index(INDEX_NAME)
    print("✅ 检索器加载成功！")
except Exception as e:
    print(f"❌ 检索器加载失败: {e}")
    exit()

def run_chat():
    print("\n" + "="*50)
    print("  多模态财报助手 (API版 - 多页增强模式)")
    print(f"  每次检索: Top-{RETRIEVAL_K} 页 (解决目录跳转问题)")
    print("="*50)

    while True:
        user_query = input("\n>>> 请提问: ")
        if user_query.lower() in ['quit', 'exit']:
            break
        
        if not user_query.strip():
            continue

        print(f"🔍 正在检索 Top-{RETRIEVAL_K} 个相关页面...")
        
        # 1. 检索 (Local Retrieval)
        try:
            results = RAG.search(user_query, k=RETRIEVAL_K)
        except Exception as e:
            print(f"❌ 检索出错: {e}")
            continue

        if not results:
            print("⚠️ 未找到相关文档页面。")
            continue

        # 2. 构建多模态输入 (Multi-Image Payload)
        # 我们将检索到的 K 张图片全部喂给大模型
        content_payload = []
        
        # 先加入文字 Prompt
        content_payload.append({
            "type": "text", 
            "text": f"你是一个专业的CFO助手。我给你提供了 {len(results)} 张财报截图。请注意：其中可能包含目录页，请忽略目录，直接根据包含具体数据的页面回答问题：{user_query}。\n如果包含图表，请详细解读数据趋势。"
        })

        print(f"📄 命中页码: ", end="")
        for i, res in enumerate(results):
            page_num = res.page_num
            print(f"[{page_num}] ", end="")
            
            # 将每一页图片加入 Payload
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{res.base64}", 
                    "detail": "high" # 高清模式
                }
            })
        print("\n🚀 正在发送给大模型进行综合分析...")

        # 3. 生成 (Cloud Generation)
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": content_payload
                    }
                ],
                temperature=0.1,
                max_tokens=2048 # 增加输出长度，因为分析多页内容可能需要更多字数
            )
            
            answer = response.choices[0].message.content
            print("\n🤖 财报助手回答:")
            print("-" * 40)
            print(answer)
            print("-" * 40)

        except Exception as e:
            print(f"❌ API 调用失败: {e}")

if __name__ == "__main__":
    run_chat()