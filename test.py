#!/usr/bin/env python3
import os
import argparse

def tree(dir_path, prefix="", level=0, max_depth=None, show_hidden=False):
    """递归打印目录结构"""
    if max_depth is not None and level > max_depth:
        return
    
    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        print(prefix + "└── [权限拒绝]")
        return

    entries = [e for e in entries if show_hidden or not e.startswith('.')]
    entries_count = len(entries)

    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        connector = "└── " if i == entries_count - 1 else "├── "
        print(prefix + connector + entry)
        
        if os.path.isdir(path):
            extension = "    " if i == entries_count - 1 else "│   "
            tree(path, prefix + extension, level + 1, max_depth, show_hidden)

def main():
    parser = argparse.ArgumentParser(description="Python 实现的 tree 命令")
    parser.add_argument("directory", nargs="?", default=".", help="要显示的目录，默认为当前目录")
    parser.add_argument("-L", "--level", type=int, default=None, help="限制目录深度")
    parser.add_argument("-a", "--all", action="store_true", help="显示隐藏文件")
    args = parser.parse_args()

    print(args.directory)
    tree(args.directory, max_depth=args.level, show_hidden=args.all)

if __name__ == "__main__":
    main()