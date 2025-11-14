#!/usr/bin/env python3  
"""  
清理 LMDB 数据库中包含 NaN 的样本  
检查所有字段: node_feat, edge_index, edge_feat, coords, angle, dists, attn_bias, masif_features  
"""  
  
import pickle  
import lmdb  
import numpy as np  
import torch  
from pathlib import Path  
import sys  
from tqdm import tqdm  
  
  
def has_nan_in_sample(sample):  
    """全面检查样本中所有字段是否包含 NaN"""  
    # 需要检查的浮点数字段  
    float_fields = ['edge_feat', 'coords', 'angle', 'dists', 'attn_bias']  
      
    for field in float_fields:  
        if field not in sample:  
            continue  
              
        data = sample[field]  
          
        if isinstance(data, torch.Tensor):  
            if data.dtype in [torch.float32, torch.float64, torch.float16]:  
                if torch.isnan(data).any().item():  
                    return True  
        elif isinstance(data, np.ndarray):  
            if np.issubdtype(data.dtype, np.floating):  
                if np.isnan(data).any():  
                    return True  
      
    # 检查 masif_features  
    if 'masif_features' in sample:  
        masif_data = sample['masif_features']  
        if isinstance(masif_data, dict):  
            for key, value in masif_data.items():  
                if isinstance(value, torch.Tensor):  
                    if value.dtype in [torch.float32, torch.float64, torch.float16]:  
                        if torch.isnan(value).any().item():  
                            return True  
                elif isinstance(value, np.ndarray):  
                    if np.issubdtype(value.dtype, np.floating):  
                        if np.isnan(value).any():  
                            return True  
      
    # 检查其他浮点数字段  
    for key, value in sample.items():  
        if isinstance(value, (float, np.floating)):  
            if np.isnan(value):  
                return True  
      
    return False  
  
  
def clean_lmdb(source_lmdb, target_lmdb, map_size=int(4e12)):  
    """清理 LMDB 数据库,移除包含 NaN 的样本"""  
    source_path = Path(source_lmdb).expanduser().absolute()  
    target_path = Path(target_lmdb).expanduser().absolute()  
      
    if not source_path.exists():  
        raise FileNotFoundError(f"❌ 源 LMDB 不存在: {source_path}")  
      
    if target_path.exists():  
        response = input(f"⚠️  目标路径已存在: {target_path}\n是否覆盖? (y/n): ")  
        if response.lower() != 'y':  
            print("操作已取消")  
            return  
        import shutil  
        shutil.rmtree(target_path)  
      
    print(f"=" * 60)  
    print(f"🧹 开始清理 LMDB 数据库")  
    print(f"源路径: {source_path}")  
    print(f"目标路径: {target_path}")  
    print(f"=" * 60)  
      
    # 打开源 LMDB  
    source_env = lmdb.open(  
        str(source_path),  
        readonly=True,  
        lock=False,  
        readahead=False,  
        meminit=False,  
        max_readers=256  
    )  
      
    # 读取原始元数据  
    original_metadata = None  
    with source_env.begin() as txn:  
        metadata_bytes = txn.get(b'__metadata__')  
        if metadata_bytes is None:  
            print("❌ 源 LMDB 缺少元数据")  
            source_env.close()  
            return  
          
        original_metadata = pickle.loads(metadata_bytes)  
        total_samples = original_metadata.get('total_samples', 0)  
        print(f"源数据库样本总数: {total_samples:,}")  
        print(f"原始元数据字段: {list(original_metadata.keys())}\n")  
      
    # 创建目标 LMDB  
    target_env = lmdb.open(  
        str(target_path),  
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
      
    valid_count = 0  
    nan_count = 0  
    error_count = 0  
    total_size = 0  
    nan_pdbids = []  
    nan_fields_stats = {}  # 统计哪些字段包含 NaN  
      
    print("开始处理样本...")  
      
    with source_env.begin() as source_txn:  
        target_txn = target_env.begin(write=True)  
          
        try:  
            for idx in tqdm(range(total_samples), desc="处理进度"):  
                try:  
                    key = f'{idx}'.encode('ascii')  
                    data_bytes = source_txn.get(key)  
                      
                    if data_bytes is None:  
                        error_count += 1  
                        continue  
                      
                    sample = pickle.loads(data_bytes)  
                      
                    # 检查 NaN  
                    if has_nan_in_sample(sample):  
                        pdbid = sample.get('pdbid', f'sample_{idx}')  
                        nan_pdbids.append(pdbid)  
                        nan_count += 1  
                          
                        # 统计哪个字段有 NaN (用于调试)  
                        for field in ['edge_feat', 'coords', 'angle', 'dists', 'attn_bias']:  
                            if field in sample:  
                                data = sample[field]  
                                has_nan = False  
                                if isinstance(data, torch.Tensor):  
                                    if data.dtype in [torch.float32, torch.float64, torch.float16]:  
                                        has_nan = torch.isnan(data).any().item()  
                                elif isinstance(data, np.ndarray):  
                                    if np.issubdtype(data.dtype, np.floating):  
                                        has_nan = np.isnan(data).any()  
                                  
                                if has_nan:  
                                    nan_fields_stats[field] = nan_fields_stats.get(field, 0) + 1  
                          
                        if nan_count % 100 == 0:  
                            print(f"\n已跳过 {nan_count} 个包含 NaN 的样本")  
                        continue  
                      
                    # 写入目标 LMDB  
                    new_key = f'{valid_count}'.encode('ascii')  
                    target_txn.put(new_key, data_bytes)  
                      
                    total_size += len(data_bytes)  
                    valid_count += 1  
                      
                    # 定期提交事务  
                    if valid_count % 10000 == 0:  
                        target_txn.commit()  
                        target_txn = target_env.begin(write=True)  
                  
                except Exception as e:  
                    print(f"\n❌ 处理样本 {idx} 时出错: {e}")  
                    error_count += 1  
                    continue  
          
        finally:  
            target_txn.commit()  
      
    # 写入新的元数据 (保留原有格式)  
    new_metadata = original_metadata.copy()  
    new_metadata['total_samples'] = valid_count  
    new_metadata['total_size_bytes'] = total_size  
    new_metadata['cleaned_from'] = str(source_path)  
    new_metadata['removed_nan_samples'] = nan_count  
    new_metadata['removed_error_samples'] = error_count  
    new_metadata['nan_pdbids_sample'] = nan_pdbids[:100]  
    new_metadata['nan_fields_stats'] = nan_fields_stats  
      
    with target_env.begin(write=True) as txn:  
        txn.put(b'__metadata__', pickle.dumps(new_metadata))  
      
    source_env.close()  
    target_env.close()  
      
    # 输出统计信息  
    print(f"\n{'=' * 60}")  
    print(f"✅ 清理完成!")  
    print(f"{'=' * 60}")  
    print(f"原始样本数: {total_samples:,}")  
    print(f"有效样本数: {valid_count:,}")  
    print(f"移除 NaN 样本: {nan_count:,} ({nan_count/total_samples*100:.2f}%)")  
    print(f"移除错误样本: {error_count:,}")  
    print(f"数据大小: {total_size / (1024**3):.2f} GB")  
    print(f"输出路径: {target_path}")  
    print(f"\n保留的原始元数据字段: {list(original_metadata.keys())}")  
      
    if nan_fields_stats:  
        print(f"\nNaN 字段统计:")  
        for field, count in sorted(nan_fields_stats.items(), key=lambda x: x[1], reverse=True):  
            print(f"  {field}: {count} 个样本")  
      
    if nan_pdbids:  
        print(f"\n前 20 个被移除的样本 pdbid:")  
        for pdbid in nan_pdbids[:20]:  
            print(f"  - {pdbid}")  
        if len(nan_pdbids) > 20:  
            print(f"  ... 还有 {len(nan_pdbids) - 20} 个")  
  
  
def main():  
    if len(sys.argv) != 3:  
        print("用法:")  
        print("  python clean_lmdb_nan.py <source_lmdb> <target_lmdb>")  
        print("\n示例:")  
        print("  python clean_lmdb_nan.py /ssd/home/scw6f3q/train_lmdb /ssd/home/scw6f3q/new_train_lmdb")  
        sys.exit(1)  
      
    source_lmdb = sys.argv[1]  
    target_lmdb = sys.argv[2]  
      
    try:  
        clean_lmdb(source_lmdb, target_lmdb)  
    except Exception as e:  
        print(f"\n❌ 清理失败: {e}")  
        import traceback  
        traceback.print_exc()  
        sys.exit(1)  
  
  
if __name__ == "__main__":  
    main()