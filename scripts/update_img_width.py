 #!/usr/bin/env python3
"""
批量修改 docs 目录下所有 md 文件中的 img 样式 width 字段
规则：
1. 当前取值乘以 1.3
2. 如果取值超过 100%，则改为 95%
"""

import os
import re
import glob

def update_img_width(match):
    """更新 img 标签中的 width 值"""
    # 提取当前的 width 值
    width_attr = match.group(1)
    width_value = match.group(2)
    
    try:
        # 提取数字部分
        width_num = int(width_value.replace('%', ''))
        
        # 乘以 1.3
        new_width = int(width_num * 1.3)
        
        # 如果超过 100%，则改为 95%
        if new_width > 100:
            new_width = 95
        
        # 返回更新后的 img 标签
        return f'{width_attr}="{new_width}%"'
    
    except ValueError:
        # 如果无法解析数字，保持原样
        return match.group(0)

def process_file(file_path):
    """处理单个文件"""
    print(f"处理文件: {file_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式查找并替换 width 属性
    # 匹配 width="数字%" 的模式
    pattern = r'(width=)"(\d+)%"'
    
    # 记录修改次数
    original_content = content
    content = re.sub(pattern, update_img_width, content)
    
    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ 已更新")
    else:
        print(f"  - 无需更新")

def main():
    """主函数"""
    # 获取 docs 目录下所有的 md 文件
    docs_dir = "docs"
    md_files = []
    
    # 递归查找所有 md 文件，但排除 node_modules
    for root, dirs, files in os.walk(docs_dir):
        # 排除 node_modules 目录
        if 'node_modules' in dirs:
            dirs.remove('node_modules')
        
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    
    print(f"找到 {len(md_files)} 个 md 文件")
    print("开始处理...")
    print("-" * 50)
    
    # 处理每个文件
    for file_path in md_files:
        process_file(file_path)
    
    print("-" * 50)
    print("处理完成！")

if __name__ == "__main__":
    main()