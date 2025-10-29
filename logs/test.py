#!/usr/bin/env python3  
"""  
将AffinCraft PKL文件转换为LMDB格式  
保留所有字段信息,无损转换,并添加元数据  
"""  
import pickle  
import lmdb  
import numpy as np  
from pathlib import Path  
import sys  
from tqdm import tqdm  
  
def convert_pkl_to_lmdb(pkl_file_path, lmdb_path, map_size=int(4e12)):  
    """  
    将PKL文件转换为LMDB格式  
      
    Args:  
        pkl_file_path: 输入的PKL文件路径  
        lmdb_path: 输出的LMDB数据库路径  
        map_size: LMDB最大容量(字节),默认4TB  
    """  
    pkl_file_path = Path(pkl_file_path).expanduser().absolute()  
    lmdb_path = Path(lmdb_path).expanduser().absolute()  
      
    if not pkl_file_path.exists():  
        raise FileNotFoundError(f"PKL文件不存在: {pkl_file_path}")  
      
    print(f"开始转换...")  
    print(f"源文件: {pkl_file_path}")  
    print(f"目标LMDB: {lmdb_path}")  
    print(f"文件大小: {pkl_file_path.stat().st_size / (1024**3):.2f} GB")  
    print(f"LMDB map_size: {map_size / (1024**4):.2f} TB")  
      
    # 创建LMDB环境  
    env = lmdb.open(  
        str(lmdb_path),  
        map_size=map_size,  # 4TB  
        subdir=True,  
        readonly=False,  
        metasync=False,  
        sync=False,  
        map_async=True,  
        writemap=True,  
        meminit=False,  
        max_readers=1  
    )  
      
    count = 0  
    total_size = 0  
      
    with open(pkl_file_path, 'rb') as f:  
        txn = env.begin(write=True)  
        try:  
            while True:  
                # 读取一个对象  
                pkl_data = pickle.load(f)  
                  
                # 序列化为bytes  
                serialized = pickle.dumps(pkl_data, protocol=pickle.HIGHEST_PROTOCOL)  
                  
                # 存储到LMDB,使用索引作为key  
                key = f'{count}'.encode('ascii')  
                txn.put(key, serialized)  
                  
                total_size += len(serialized)  
                count += 1  
                  
                # 每1000个对象报告一次进度  
                if count % 1000 == 0:  
                    print(f"已处理 {count:,} 个对象, 累计大小: {total_size / (1024**3):.2f} GB")  
                  
                # 每10000个对象提交一次事务(提高性能)  
                if count % 10000 == 0:  
                    txn.commit()  
                    txn = env.begin(write=True)  
                      
        except EOFError:  
            print(f"文件读取完成")  
        finally:  
            # 确保最后一批数据被提交  
            txn.commit()  
      
    # 存储元数据  
    print("正在写入元数据...")  
    with env.begin(write=True) as txn:  
        metadata = {  
            'total_samples': count,  
            'source_file': str(pkl_file_path),  
            'total_size_bytes': total_size  
        }  
        txn.put(b'__metadata__', pickle.dumps(metadata))  
      
    env.close()  
      
    print(f"\n✅ 转换完成!")  
    print(f"总样本数: {count:,}")  
    print(f"LMDB大小: {total_size / (1024**3):.2f} GB")  
    print(f"LMDB路径: {lmdb_path}")  
      
    return lmdb_path  
  
def verify_lmdb(lmdb_path, num_samples=5):  
    """验证LMDB数据完整性"""  
    print(f"\n验证LMDB数据...")  
    env = lmdb.open(str(lmdb_path), readonly=True, subdir=True)  
      
    with env.begin() as txn:  
        # 读取元数据  
        metadata_bytes = txn.get(b'__metadata__')  
        if metadata_bytes:  
            metadata = pickle.loads(metadata_bytes)  
            print(f"元数据: {metadata}")  
        else:  
            print("⚠️ 警告: 未找到元数据")  
            env.close()  
            return  
          
        # 随机检查几个样本  
        print(f"\n检查前{num_samples}个样本:")  
        for i in range(min(num_samples, metadata['total_samples'])):  
            key = f'{i}'.encode('ascii')  
            data = txn.get(key)  
            if data is None:  
                print(f"  样本 {i}: ❌ 缺失")  
            else:  
                obj = pickle.loads(data)  
                pdbid = obj.get('pdbid', 'unknown')  
                print(f"  样本 {i}: ✅ pdbid={pdbid}, 字段数={len(obj)}")  
      
    env.close()  
    print("验证完成!")  
  
def add_metadata_to_existing_lmdb(lmdb_path):  
    """为已存在的LMDB添加元数据(如果缺失)"""  
    lmdb_path = Path(lmdb_path).expanduser().absolute()  
      
    if not lmdb_path.exists():  
        print(f"❌ 错误: LMDB路径不存在: {lmdb_path}")  
        return False  
      
    print(f"正在处理: {lmdb_path}")  
      
    # 以读写模式打开LMDB  
    env = lmdb.open(  
        str(lmdb_path),  
        readonly=False,  
        subdir=True,  
        lock=True,  
        map_size=int(4e12)  # 4TB  
    )  
      
    # 检查是否已有元数据  
    with env.begin() as txn:  
        existing_meta = txn.get(b'__metadata__')  
        if existing_meta:  
            meta = pickle.loads(existing_meta)  
            print(f"⚠️ 元数据已存在: {meta}")  
            env.close()  
            return True  
      
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
      
    return True  
  
def main():  
    if len(sys.argv) < 2:  
        print("用法:")  
        print("  转换: python convert_pkl_to_lmdb.py <pkl_file> [lmdb_output_dir]")  
        print("  添加元数据: python convert_pkl_to_lmdb.py --add-metadata <lmdb_path>")  
        print("\n示例:")  
        print("  python convert_pkl_to_lmdb.py ~/pkl_affincraft/train.pkl ~/lmdb_affincraft/train.lmdb")  
        print("  python convert_pkl_to_lmdb.py --add-metadata ~/lmdb_affincraft/train.lmdb")  
        sys.exit(1)  
      
    # 添加元数据模式  
    if sys.argv[1] == '--add-metadata':  
        if len(sys.argv) < 3:  
            print("❌ 错误: 请指定LMDB路径")  
            sys.exit(1)  
        lmdb_path = sys.argv[2]  
        try:  
            add_metadata_to_existing_lmdb(lmdb_path)  
        except Exception as e:  
            print(f"❌ 处理失败: {e}")  
            import traceback  
            traceback.print_exc()  
            sys.exit(1)  
        return  
      
    # 转换模式  
    pkl_file = sys.argv[1]  
      
    # 如果未指定输出路径,自动生成  
    if len(sys.argv) > 2:  
        lmdb_path = sys.argv[2]  
    else:  
        pkl_path = Path(pkl_file)  
        lmdb_path = pkl_path.parent / f"{pkl_path.stem}.lmdb"  
      
    try:  
        result_path = convert_pkl_to_lmdb(pkl_file, lmdb_path)  
        verify_lmdb(result_path)  
    except Exception as e:  
        print(f"❌ 转换失败: {e}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)  
  
if __name__ == "__main__":  
    main()