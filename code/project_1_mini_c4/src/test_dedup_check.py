import json
from datasketch import MinHash, MinHashLSH

# 1. 准备三条数据：A 和 B 一模一样，C 是完全不同的
data_a = "Deep learning is a subset of machine learning using neural networks."
data_b = "Deep learning is a subset of machine learning using neural networks." # 完全重复
data_c = "Cooking pasta requires boiling water and adding salt." # 完全不同

# 2. 计算 MinHash
def get_minhash(text):
    m = MinHash(num_perm=128)
    for w in text.split():
        m.update(w.encode('utf8'))
    return m

mh_a = get_minhash(data_a)
mh_b = get_minhash(data_b)
mh_c = get_minhash(data_c)

# 3. 放入 LSH
lsh = MinHashLSH(threshold=0.8, num_perm=128)
lsh.insert("doc_a", mh_a)

# 4. 查询
print(f"查询 B (应该重复): {lsh.query(mh_b)}") 
print(f"查询 C (不该重复): {lsh.query(mh_c)}")

# 预期输出：
# 查询 B: ['doc_a']  <--- 成功抓到！
# 查询 C: []         <--- 安全忽略