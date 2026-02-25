import os
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

from byaldi import RAGMultiModalModel

MODEL_PATH = "/home/xuxin123/book/project_5_rag/models/colpali-v1_2-merged"
PDF_PATH = "/home/xuxin123/book/project_5_rag/data/annual_report_2024_cn.pdf"
INDEX_NAME = "finance_report_2024"

def build_index():
    print(f"🚀 正在从本地加载 ColPali-v1.2-merged 模型...")
    print(f"📂 模型路径: {MODEL_PATH}")
    
    # 检查模型路径是否存在，防止报错
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到模型文件夹！请确认路径是否正确：{MODEL_PATH}")
        print("提示：如果你是在 src 目录下解压的，可能需要把文件夹移动到 models 目录下。")
        return

    # 从本地路径加载
    # 如果运行通过但显存爆了(OOM)，可以在这里加上 load_in_4bit=True
    try:
        RAG = RAGMultiModalModel.from_pretrained(
            MODEL_PATH,
            verbose=1
        )
    except Exception as e:
        print(f"❌ 模型加载失败。错误信息: {e}")
        return

    print(f"📖 开始索引 PDF 文件: {PDF_PATH}")
    print("提示：此过程会将每页转为视觉向量，请耐心等待...")
    
    # 执行索引
    RAG.index(
        input_path=PDF_PATH,
        index_name=INDEX_NAME,
        store_collection_with_index=True,
        overwrite=True
    )
    
    print(f"✅ 索引构建成功！保存位置：.byaldi/{INDEX_NAME}")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ 错误：找不到 PDF 文件 {PDF_PATH}")
    else:
        build_index()