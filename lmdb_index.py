#!/usr/bin/env python3  
"""  
为现有LMDB数据库添加元数据  
修复转换时元数据写入失败的问题  
"""  
import lmdb  
import pickle  
from pathlib import Path  
import sys  
  
def add_metadata_to_lmdb(lmdb_path):  
    """  
    为LMDB添加缺失的元数据  
      
    Args:  
        lmdb_path: LMDB数据库路径  
    """  
    lmdb_path = Path(lmdb_path).expanduser().absolute()  
      
    if not lmdb_path.exists():  
        print(f"❌ 错误: LMDB路径不存在: {lmdb_path}")  
        return False  
      
    print(f"正在处理: {lmdb_path}")  
    print(f"数据库大小: {sum(f.stat().st_size for f in lmdb_path.glob('*.mdb')) / (1024**3):.2f} GB")  
      
    # 以读写模式打开LMDB  
    env = lmdb.open(  
        str(lmdb_path),  
        readonly=False,  
        subdir=True,  
        lock=True,  
        map_size=int(8e12)  # 8TB,确保有足够空间  
    )  
      
    # 检查是否已有元数据  
    with env.begin() as txn:  
        existing_meta = txn.get(b'__metadata__')  
        if existing_meta:  
            meta = pickle.loads(existing_meta)  
            print(f"⚠️ 元数据已存在: {meta}")  
            response = input("是否覆盖? (y/n): ")  
            if response.lower() != 'y':  
                env.close()  
                return False  
      
    # 统计样本数量  
    print("正在统计样本数量...")  
    count = 0  
    total_size = 0  
      
    with env.begin() as txn:  
        cursor = txn.cursor()  
        for key, value in cursor:  
            # 跳过元数据键本身  
            if key == b'__metadata__':  
                continue  
            count += 1  
            total_size += len(value)  
              
            # 每10万个样本报告一次进度  
            if count % 100000 == 0:  
                print(f"  已统计 {count:,} 个样本...")  
      
    print(f"✅ 统计完成: 找到 {count:,} 个样本")  
    print(f"   总数据大小: {total_size / (1024**3):.2f} GB")  
      
    # 写入元数据  
    print("正在写入元数据...")  
    with env.begin(write=True) as txn:  
        metadata = {  
            'total_samples': count,  
            'source_file': 'converted_from_pkl',  
            'total_size_bytes': total_size  
        }  
        txn.put(b'__metadata__', pickle.dumps(metadata))  
      
    env.close()  
      
    print(f"✅ 元数据已成功添加!")  
    print(f"   样本总数: {count:,}")  
      
    # 验证元数据  
    print("\n验证元数据...")  
    env = lmdb.open(str(lmdb_path), readonly=True, subdir=True, lock=False)  
    with env.begin() as txn:  
        meta_bytes = txn.get(b'__metadata__')  
        if meta_bytes:  
            meta = pickle.loads(meta_bytes)  
            print(f"✅ 验证成功: {meta}")  
        else:  
            print("❌ 验证失败: 无法读取元数据")  
    env.close()  
      
    return True  
  
def main():  
    if len(sys.argv) < 2:  
        print("用法: python add_metadata.py <lmdb_path>")  
        print("\n示例:")  
        print("  python add_metadata.py /data/run01/scw6f3q/zncao/lmdb_affincraft/train.lmdb")  
        print("  python add_metadata.py /data/run01/scw6f3q/zncao/lmdb_affincraft/valid.lmdb")  
        sys.exit(1)  
      
    lmdb_path = sys.argv[1]  
      
    try:  
        success = add_metadata_to_lmdb(lmdb_path)  
        if success:  
            print("\n✅ 处理完成!")  
        else:  
            print("\n⚠️ 处理已取消")  
    except Exception as e:  
        print(f"\n❌ 处理失败: {e}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)  
  
if __name__ == "__main__":  
    main()