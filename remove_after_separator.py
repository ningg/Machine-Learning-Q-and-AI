import os

pages_dir = 'docs'

for filename in os.listdir(pages_dir):
    if not filename.endswith('.md'):
        continue

    md_path = os.path.join(pages_dir, filename)
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 查找分隔线 --------------------- 所在行
    separator_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('::: {.book-promotion'):
            separator_idx = i
            break

    # 如果找到，删除该行及之后所有内容
    if separator_idx is not None:
        new_lines = lines[:separator_idx]
        with open(md_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Processed {filename}")
    else:
        print(f"No separator found in {filename}, skipped.")

print("Done.")