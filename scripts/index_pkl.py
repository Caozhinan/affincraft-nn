#!/usr/bin/env python3  
import pickle  
import os  
import sys  
from pathlib import Path  
  
def build_pkl_index(pkl_file_path, index_file_path=None):  
    """构建PKL文件的位置索引"""  
    pkl_file_path = Path(pkl_file_path).expanduser().absolute()  
      
    if not pkl_file_path.exists():  
        raise FileNotFoundError(f"PKL文件不存在: {pkl_file_path}")  
      
    # 如果没有指定索引文件路径，自动生成  
    if index_file_path is None:  
        index_file_path = pkl_file_path.with_suffix('.idx')  
    else:  
        index_file_path = Path(index_file_path).expanduser().absolute()  
      
    print(f"开始构建索引...")  
    print(f"源文件: {pkl_file_path}")  
    print(f"索引文件: {index_file_path}")  
    print(f"文件大小: {pkl_file_path.stat().st_size / (1024**3):.2f} GB")  
      
    positions = []  
      
    with open(pkl_file_path, 'rb') as f:  
        count = 0  
        try:  
            while True:  
                pos = f.tell()  
                pickle.load(f)  # 加载对象以移动文件指针  
                positions.append(pos)  
                count += 1  
                  
                # 每10000个对象报告一次进度  
                if count % 10000 == 0:  
                    print(f"已处理 {count:,} 个对象...")  
                      
        except EOFError:  
            print(f"文件读取完成，共处理 {count:,} 个对象")  
      
    # 保存索引到文件  
    index_data = {  
        'positions': positions,  
        'total_objects': len(positions),  
        'file_size': pkl_file_path.stat().st_size,  
        'source_file': str(pkl_file_path)  
    }  
      
    with open(index_file_path, 'wb') as f:  
        pickle.dump(index_data, f)  
      
    print(f"索引构建完成：{len(positions):,} 个对象")  
    print(f"索引文件大小: {index_file_path.stat().st_size / (1024**2):.2f} MB")  
    print(f"索引文件保存到: {index_file_path}")  
      
    return index_file_path  
  
def main():  
    if len(sys.argv) < 2:  
        print("用法: python build_index.py <pkl_file_path> [index_file_path]")  
        print("示例: python build_index.py ~/pkl_affincraft/train.pkl")  
        print("示例: python build_index.py ~/pkl_affincraft/train.pkl ~/pkl_affincraft/train.idx")  
        sys.exit(1)  
      
    pkl_file_path = sys.argv[1]  
    index_file_path = sys.argv[2] if len(sys.argv) > 2 else None  
      
    try:  
        result_path = build_pkl_index(pkl_file_path, index_file_path)  
        print(f"\\n✅ 索引构建成功！")  
        print(f"可以在训练时使用: --train-pkl-index {result_path}")  
    except Exception as e:  
        print(f"❌ 索引构建失败: {e}")  
        sys.exit(1)  
  
if __name__ == "__main__":  
    main()