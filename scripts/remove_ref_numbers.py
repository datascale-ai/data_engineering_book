import re
import os
import glob

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find ## 参考文献
    match = re.search(r'## 参考文献\n', content)
    if not match:
        return False

    pos = match.end()
    before = content[:pos]
    after = content[pos:]

    # Remove [N] prefix (with optional leading whitespace) from lines
    # Pattern: optional whitespace + [digits] + whitespace
    after = re.sub(r'^(\s*)\[\d+\]\s+', r'\1', after, flags=re.MULTILINE)

    new_content = before + after

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True


def main():
    base_dir = r'D:\r\Mindspore\书1\code\data_engineering_book\docs\zh'
    parts = ['part4', 'part5', 'part6']

    count = 0
    for part in parts:
        part_dir = os.path.join(base_dir, part)
        if not os.path.isdir(part_dir):
            print(f'Directory not found: {part_dir}')
            continue
        for md_file in glob.glob(os.path.join(part_dir, '*.md')):
            try:
                if process_file(md_file):
                    print(f'OK: {os.path.basename(md_file)}')
                    count += 1
                else:
                    print(f'Skip (no 参考文献): {os.path.basename(md_file)}')
            except Exception as e:
                print(f'Error {os.path.basename(md_file)}: {e}')

    print(f'\nDone. Processed {count} files.')


if __name__ == '__main__':
    main()
