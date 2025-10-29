#!/usr/bin/env python3  
"""  
将 AffinCraft PKL 文件转换为 LMDB 格式  
在写入前过滤包含NaN的样本  
"""  
  
import pickle  
import lmdb  
import numpy as np  
from pathlib import Path  
import sys  
from tqdm import tqdm  
  
def has_nan_values(pkl_obj):  
    """  
    检测样本是否包含NaN值  
      
    Args:  
        pkl_obj: PKL对象(字典)  
          
    Returns:  
        bool: 如果包含NaN返回True,否则返回False  
    """  
    critical_fields = ['node_feat', 'edge_feat', 'coords', 'masif_desc_straight']  
      
    for field in critical_fields:  
        if field in pkl_obj:  
            data = pkl_obj[field]  
            if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):  
                if np.isnan(data).any():  
                    return True  
    return False  
  
def convert_pkl_to_lmdb_with_nan_filter(pkl_file_path, lmdb_path, map_size=int(0.5e12)):  
    """  
    将PKL文件转换为LMDB格式,过滤包含NaN的样本  
      
    Args:  
        pkl_file_path: 输入的PKL文件路径  
        lmdb_path: 输出的LMDB数据库路径  
        map_size: LMDB最大容量(字节), 默认 4TB  
    """  
    pkl_file_path = Path(pkl_file_path).expanduser().absolute()  
    lmdb_path = Path(lmdb_path).expanduser().absolute()  
  
    if not pkl_file_path.exists():  
        raise FileNotFoundError(f"❌ PKL文件不存在: {pkl_file_path}")  
  
    print(f"开始转换 AffinCraft PKL → LMDB (带NaN过滤)...")  
    print(f"源文件: {pkl_file_path}")  
    print(f"目标LMDB: {lmdb_path}")  
    print(f"源文件大小: {pkl_file_path.stat().st_size / (1024**3):.2f} GB")  
  
    # 创建LMDB环境  
    env = lmdb.open(  
        str(lmdb_path),  
        map_size=map_size,  
        subdir=True,  
        readonly=False,  
        lock=True,  
        metasync=False,  
        sync=False,  
        map_async=True,  
        writemap=True,  
        meminit=False,  
        max_readers=1  
    )  
  
    count = 0  
    skipped_count = 0  
    total_size = 0  
    skipped_pdbids = []  # 记录被跳过的pdbid  
      
    with open(pkl_file_path, "rb") as f:  
        txn = env.begin(write=True)  
        try:  
            while True:  
                # 从PKL流式取出对象  
                pkl_obj = pickle.load(f)  
                  
                # 检测NaN  
                if has_nan_values(pkl_obj):  
                    pdbid = pkl_obj.get('pdbid', f'unknown_{count + skipped_count}')  
                    skipped_pdbids.append(pdbid)  
                    skipped_count += 1  
                      
                    # 每100个跳过的样本打印一次  
                    if skipped_count % 100 == 0:  
                        print(f"已跳过 {skipped_count} 个包含NaN的样本")  
                    continue  
  
                # 序列化为bytes  
                serialized = pickle.dumps(pkl_obj, protocol=pickle.HIGHEST_PROTOCOL)  
  
                # 存进LMDB  
                key = f"{count}".encode("ascii")  
                txn.put(key, serialized)  
  
                total_size += len(serialized)  
                count += 1  
  
                # 每隔1000个对象打印一次  
                if count % 1000 == 0:  
                    print(f"已处理 {count:,} 个有效样本，跳过 {skipped_count} 个NaN样本")  
  
                # 每10000个提交一次事务，提高写入性能  
                if count % 10000 == 0:  
                    txn.commit()  
                    txn = env.begin(write=True)  
        except EOFError:  
            print("✅ PKL文件读取完成")  
        finally:  
            txn.commit()  
  
    # 添加元数据  
    metadata = {  
        "total_samples": count,  
        "skipped_samples": skipped_count,  
        "source_file": str(pkl_file_path),  
        "total_size_bytes": total_size,  
        "map_size_bytes": map_size,  
        "skipped_pdbids": skipped_pdbids[:1000]  # 只保存前1000个,避免元数据过大  
    }  
  
    with env.begin(write=True) as txn:  
        txn.put(b"__metadata__", pickle.dumps(metadata))  
  
    env.close()  
  
    print(f"\n✅ 转换完成！")  
    print(f"有效样本总数: {count:,}")  
    print(f"跳过的NaN样本: {skipped_count:,}")  
    print(f"跳过比例: {skipped_count / (count + skipped_count) * 100:.2f}%")  
    print(f"LMDB逻辑容量: {map_size / (1024**4):.2f} TB")  
    print(f"实际数据: {total_size / (1024**3):.2f} GB")  
    print(f"输出路径: {lmdb_path}")  
      
    # 输出前20个被跳过的pdbid  
    if skipped_pdbids:  
        print(f"\n前20个被跳过的样本pdbid:")  
        for pdbid in skipped_pdbids[:20]:  
            print(f"  - {pdbid}")  
        if len(skipped_pdbids) > 20:  
            print(f"  ... 还有 {len(skipped_pdbids) - 20} 个")  
  
    return lmdb_path  
  
  
def main():  
    if len(sys.argv) < 3:  
        print("用法:")  
        print("  python convert_pkl_to_lmdb_with_nan_filter.py <input_pkl> <output_lmdb>")  
        sys.exit(1)  
  
    pkl_file = sys.argv[1]  
    lmdb_path = sys.argv[2]  
    convert_pkl_to_lmdb_with_nan_filter(pkl_file, lmdb_path)  
  
  
if __name__ == "__main__":  
    main()