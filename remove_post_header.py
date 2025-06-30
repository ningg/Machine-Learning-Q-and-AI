import os

pages_dir = 'pages'

for filename in os.listdir(pages_dir):
    if not filename.endswith('.md'):
        continue

    md_path = os.path.join(pages_dir, filename)
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 查找 ::: post-header 行
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('::: post-header'):
            header_idx = i
            break

    # 如果找到，删除该行及之前所有内容
    if header_idx is not None:
        new_lines = lines[header_idx+1:]
        with open(md_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Processed {filename}")
    else:
        print(f"No '::: post-header' found in {filename}, skipped.")

print("Done.")