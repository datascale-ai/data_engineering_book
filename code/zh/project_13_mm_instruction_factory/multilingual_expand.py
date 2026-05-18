def expand_multilingual(data):
    expanded = []
    for item in data:
        expanded.append(item)
        zh_item = item.copy()
        zh_item["instruction_zh"] = "模拟中文指令扩展: " + item["instruction"]
        expanded.append(zh_item)
    return expanded
