#!/usr/bin/env python3  
"""  
将 LMDB 验证集随机分割为验证集和测试集  
保持数据格式不变  
"""  
import lmdb  
import pickle  
import random  
from pathlib import Path  
import sys  
  
def split_lmdb(source_lmdb, valid_lmdb, test_lmdb, seed=42):  
    """  
    将源 LMDB 随机分割为验证集和测试集  
      
    Args:  
        source_lmdb: 源 LMDB 路径  
        valid_lmdb: 输出验证集 LMDB 路径  
        test_lmdb: 输出测试集 LMDB 路径  
        seed: 随机种子  
    """  
    source_lmdb = Path(source_lmdb).expanduser().absolute()  
    valid_lmdb = Path(valid_lmdb).expanduser().absolute()  
    test_lmdb = Path(test_lmdb).expanduser().absolute()  
      
    if not source_lmdb.exists():  
        raise FileNotFoundError(f"❌ 源 LMDB 不存在: {source_lmdb}")  
      
    print(f"📂 源 LMDB: {source_lmdb}")  
    print(f"📂 验证集输出: {valid_lmdb}")  
    print(f"📂 测试集输出: {test_lmdb}")  
    print("=" * 60)  
      
    # 1. 读取源 LMDB 的所有样本索引  
    print("\n1️⃣ 读取源 LMDB 元数据...")  
    source_env = lmdb.open(  
        str(source_lmdb),  
        readonly=True,  
        lock=False,  
        readahead=False,  
        meminit=False,  
        max_readers=256  
    )  
      
    with source_env.begin() as txn:  
        metadata_bytes = txn.get(b'__metadata__')  
        if metadata_bytes is None:  
            raise ValueError("❌ 源 LMDB 缺少元数据")  
          
        metadata = pickle.loads(metadata_bytes)  
        total_samples = metadata['total_samples']  
        print(f"✅ 总样本数: {total_samples:,}")  
      
    # 2. 生成随机打乱的索引  
    print("\n2️⃣ 生成随机索引...")  
    random.seed(seed)  
    indices = list(range(total_samples))  
    random.shuffle(indices)  
      
    # 平均分割  
    split_point = total_samples // 2  
    valid_indices = set(indices[:split_point])  
    test_indices = set(indices[split_point:])  
      
    print(f"✅ 验证集样本数: {len(valid_indices):,}")  
    print(f"✅ 测试集样本数: {len(test_indices):,}")  
      
    # 3. 创建输出 LMDB 环境  
    print("\n3️⃣ 创建输出 LMDB...")  
    valid_env = lmdb.open(  
        str(valid_lmdb),  
        map_size=int(0.3e12),  # 4TB  
        subdir=True,  
        readonly=False,  
        metasync=False,  
        sync=False,  
        map_async=True,  
        writemap=True,  
        meminit=False,  
        max_readers=1  
    )  
      
    test_env = lmdb.open(  
        str(test_lmdb),  
        map_size=int(0.3e12),  # 4TB  
        subdir=True,  
        readonly=False,  
        metasync=False,  
        sync=False,  
        map_async=True,  
        writemap=True,  
        meminit=False,  
        max_readers=1  
    )  
      
    # 4. 写入数据  
    print("\n4️⃣ 写入数据...")  
    valid_count = 0  
    test_count = 0  
    valid_size = 0  
    test_size = 0  
      
    valid_txn = valid_env.begin(write=True)  
    test_txn = test_env.begin(write=True)  
      
    try:  
        with source_env.begin() as source_txn:  
            for idx in range(total_samples):  
                # 读取原始数据  
                key = f'{idx}'.encode('ascii')  
                data_bytes = source_txn.get(key)  
                  
                if data_bytes is None:  
                    print(f"⚠️  警告: 索引 {idx} 数据缺失,跳过")  
                    continue  
                  
                # 根据索引分配到验证集或测试集  
                if idx in valid_indices:  
                    new_key = f'{valid_count}'.encode('ascii')  
                    valid_txn.put(new_key, data_bytes)  
                    valid_size += len(data_bytes)  
                    valid_count += 1  
                      
                    # 每 10000 个样本提交一次  
                    if valid_count % 10000 == 0:  
                        valid_txn.commit()  
                        valid_txn = valid_env.begin(write=True)  
                        print(f"  验证集已写入 {valid_count:,} 个样本")  
                else:  
                    new_key = f'{test_count}'.encode('ascii')  
                    test_txn.put(new_key, data_bytes)  
                    test_size += len(data_bytes)  
                    test_count += 1  
                      
                    # 每 10000 个样本提交一次  
                    if test_count % 10000 == 0:  
                        test_txn.commit()  
                        test_txn = test_env.begin(write=True)  
                        print(f"  测试集已写入 {test_count:,} 个样本")  
          
        # 最终提交  
        valid_txn.commit()  
        test_txn.commit()  
          
    except Exception as e:  
        print(f"❌ 写入过程出错: {e}")  
        valid_txn.abort()  
        test_txn.abort()  
        raise  
      
    # 5. 写入元数据  
    print("\n5️⃣ 写入元数据...")  
    with valid_env.begin(write=True) as txn:  
        valid_metadata = {  
            'total_samples': valid_count,  
            'source_file': str(source_lmdb),  
            'total_size_bytes': valid_size,  
            'split_seed': seed  
        }  
        txn.put(b'__metadata__', pickle.dumps(valid_metadata))  
      
    with test_env.begin(write=True) as txn:  
        test_metadata = {  
            'total_samples': test_count,  
            'source_file': str(source_lmdb),  
            'total_size_bytes': test_size,  
            'split_seed': seed  
        }  
        txn.put(b'__metadata__', pickle.dumps(test_metadata))  
      
    # 6. 关闭环境  
    source_env.close()  
    valid_env.close()  
    test_env.close()  
      
    # 7. 输出统计信息  
    print("\n" + "=" * 60)  
    print("✅ 分割完成!")  
    print(f"\n📊 验证集:")  
    print(f"   - 样本数: {valid_count:,}")  
    print(f"   - 大小: {valid_size / (1024**3):.2f} GB")  
    print(f"   - 路径: {valid_lmdb}")  
    print(f"\n📊 测试集:")  
    print(f"   - 样本数: {test_count:,}")  
    print(f"   - 大小: {test_size / (1024**3):.2f} GB")  
    print(f"   - 路径: {test_lmdb}")  
    print(f"\n🎲 随机种子: {seed}")  
  
if __name__ == "__main__":  
    if len(sys.argv) < 4:  
        print("用法: python split_lmdb.py <source_lmdb> <valid_lmdb> <test_lmdb> [seed]")  
        print("示例: python split_lmdb.py /ssd/home/scw6f3q/lmdb/valid.lmdb /ssd/home/scw6f3q/valid_lmdb /ssd/home/scw6f3q/test_lmdb 42")  
        sys.exit(1)  
      
    source = sys.argv[1]  
    valid = sys.argv[2]  
    test = sys.argv[3]  
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42  
      
    try:  
        split_lmdb(source, valid, test, seed)  
    except Exception as e:  
        print(f"❌ 分割失败: {e}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)