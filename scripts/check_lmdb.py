#!/usr/bin/env python3  
"""  
LMDB 数据集完整性检查脚本  
检查 AffinCraft LMDB 数据库的格式和数据完整性  
"""  
  
import lmdb  
import pickle  
import sys  
from pathlib import Path  
  
def check_lmdb(lmdb_path):  
    """检查 LMDB 数据库的完整性"""  
      
    lmdb_path = Path(lmdb_path).expanduser().absolute()  
      
    # 检查路径是否存在  
    if not lmdb_path.exists():  
        print(f"❌ 错误: LMDB 路径不存在: {lmdb_path}")  
        return False  
      
    print(f"📂 检查 LMDB: {lmdb_path}")  
    print("=" * 60)  
      
    try:  
        # 打开 LMDB 环境  
        env = lmdb.open(  
            str(lmdb_path),  
            readonly=True,  
            lock=False,  
            readahead=False,  
            meminit=False,  
            max_readers=256  
        )  
          
        with env.begin() as txn:  
            # 1. 检查元数据  
            print("\n1️⃣ 检查元数据...")  
            metadata_bytes = txn.get(b'__metadata__')  
              
            if metadata_bytes is None:  
                print("❌ 错误: 缺少 __metadata__ 键")  
                return False  
              
            try:  
                metadata = pickle.loads(metadata_bytes)  
                total_samples = metadata.get('total_samples', 0)  
                print(f"✅ 元数据正常")  
                print(f"   - 声明的样本总数: {total_samples:,}")  
            except Exception as e:  
                print(f"❌ 元数据反序列化失败: {e}")  
                return False  
              
            # 2. 检查实际样本数量  
            print("\n2️⃣ 检查实际样本数量...")  
            actual_count = 0  
            cursor = txn.cursor()  
              
            for key, _ in cursor:  
                if key != b'__metadata__':  
                    actual_count += 1  
              
            print(f"   - 实际样本数量: {actual_count:,}")  
              
            if actual_count != total_samples:  
                print(f"⚠️  警告: 实际样本数({actual_count})与元数据不符({total_samples})")  
            else:  
                print(f"✅ 样本数量一致")  
              
            # 3. 随机抽样检查数据完整性  
            print("\n3️⃣ 抽样检查数据完整性...")  
            check_indices = [0, total_samples // 2, total_samples - 1] if total_samples > 0 else []  
            nan_count = 0  
            error_count = 0  
              
            for idx in check_indices:  
                if idx >= total_samples:  
                    continue  
                      
                key = f'{idx}'.encode('ascii')  
                data_bytes = txn.get(key)  
                  
                if data_bytes is None:  
                    print(f"❌ 索引 {idx}: 数据缺失")  
                    error_count += 1  
                    continue  
                  
                try:  
                    pkl_data = pickle.loads(data_bytes)  
                      
                    # 检查必要字段  
                    required_fields = ['pdbid', 'ligand_coords', 'protein_coords']  
                    missing_fields = [f for f in required_fields if f not in pkl_data]  
                      
                    if missing_fields:  
                        print(f"⚠️  索引 {idx}: 缺少字段 {missing_fields}")  
                      
                    # 检查 NaN  
                    import numpy as np  
                    has_nan = False  
                    for field in ['ligand_coords', 'protein_coords']:  
                        if field in pkl_data:  
                            arr = np.array(pkl_data[field])  
                            if np.isnan(arr).any():  
                                has_nan = True  
                                nan_count += 1  
                                break  
                      
                    if has_nan:  
                        pdbid = pkl_data.get('pdbid', f'sample_{idx}')  
                        print(f"⚠️  索引 {idx} (pdbid: {pdbid}): 包含 NaN 值")  
                    else:  
                        print(f"✅ 索引 {idx}: 数据正常")  
                          
                except Exception as e:  
                    print(f"❌ 索引 {idx}: 反序列化失败 - {e}")  
                    error_count += 1  
              
            # 4. 统计信息  
            print("\n" + "=" * 60)  
            print("📊 检查总结:")  
            print(f"   - 总样本数: {total_samples:,}")  
            print(f"   - 实际样本数: {actual_count:,}")  
            print(f"   - 抽样检查: {len(check_indices)} 个样本")  
            print(f"   - 发现 NaN: {nan_count} 个")  
            print(f"   - 错误样本: {error_count} 个")  
              
            if error_count == 0 and actual_count == total_samples:  
                print("\n✅ LMDB 数据库格式正常!")  
                return True  
            else:  
                print("\n⚠️  LMDB 数据库存在问题,请检查!")  
                return False  
                  
        env.close()  
          
    except Exception as e:  
        print(f"\n❌ 检查过程中出错: {e}")  
        import traceback  
        traceback.print_exc()  
        return False  
  
if __name__ == "__main__":  
    if len(sys.argv) != 2:  
        print("用法: python check_lmdb.py <lmdb_path>")  
        print("示例: python check_lmdb.py /ssd/home/scw6f3q/lmdb/valid.lmdb")  
        sys.exit(1)  
      
    lmdb_path = sys.argv[1]  
    success = check_lmdb(lmdb_path)  
    sys.exit(0 if success else 1)