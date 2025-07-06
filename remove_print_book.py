import os

pages_dir = 'docs'

for filename in os.listdir(pages_dir):
    if not filename.endswith('.md'):
        continue

    md_path = os.path.join(pages_dir, filename)
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 查找 **ðŸ"˜ Print Book:** 所在行
    idx = None
    for i, line in enumerate(lines):
        if '**ðŸ"˜ Print Book:**' in line:
            idx = i
            break

    # 如果找到，删除该行以及之后7行内容
    if idx is not None:
        new_lines = lines[:idx] + lines[idx+8:]
        with open(md_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"Processed {filename}")
    else:
        print(f"No '**ðŸ\"˜ Print Book:**' found in {filename}, skipped.")

print("Done.")