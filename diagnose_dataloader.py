#!/usr/bin/env python3  
"""  
DataLoader 崩溃诊断脚本 - 用于 AffinCraft LMDB 数据集  
"""  
import os  
import sys  
import traceback  
import torch  
from torch.utils.data import DataLoader  
import psutil  
import pickle  
import lmdb  
  
# 正确设置 Python 路径  
GRAPHORMER_PATH = "/data/run01/scw6f3q/zncao/affincraft-nn/graphormer"  
if GRAPHORMER_PATH not in sys.path:  
    sys.path.insert(0, GRAPHORMER_PATH)  
  
# 直接导入,不触发相对导入  
from data.affincraft_dataset import LMDBAffinCraftDataset  
  
def shm_status():  
    """打印 /dev/shm 与系统内存使用情况"""  
    os.system("echo '\n[SHM STATUS]' && df -h /dev/shm 2>/dev/null || echo '/dev/shm not available'")  
    vmem = psutil.virtual_memory()  
    print(f"[Memory] total={vmem.total/1e9:.1f}GB used={vmem.used/1e9:.1f}GB avail={vmem.available/1e9:.1f}GB percent={vmem.percent}%")  
  
def tensor_stats(t):  
    """获取 tensor 统计信息"""  
    if not torch.is_tensor(t):  
        return f"not tensor (type={type(t).__name__})"  
    try:  
        has_nan = torch.isnan(t).any().item() if t.dtype in [torch.float32, torch.float64, torch.float16] else False  
        has_inf = torch.isinf(t).any().item() if t.dtype in [torch.float32, torch.float64, torch.float16] else False  
        return (f"shape={tuple(t.shape)}, dtype={t.dtype}, "  
                f"min={float(t.min()):.4e}, max={float(t.max()):.4e}, "  
                f"mean={float(t.mean()):.4e}, NaN={has_nan}, Inf={has_inf}")  
    except Exception as e:  
        return f"shape={tuple(t.shape)}, dtype={t.dtype}, ERROR: {e}"  
  
def sample_summary(sample):  
    """生成样本摘要"""  
    info = []  
    if isinstance(sample, dict):  
        for k, v in sample.items():  
            try:  
                if torch.is_tensor(v):  
                    info.append(f"  {k}: {tensor_stats(v)}")  
                else:  
                    info.append(f"  {k}: type={type(v).__name__}, len={len(v) if hasattr(v, '__len__') else 'N/A'}")  
            except Exception as e:  
                info.append(f"  {k}: ERROR - {e}")  
    elif isinstance(sample, (list, tuple)):  
        for i, v in enumerate(sample):  
            try:  
                info.append(f"  [{i}] {tensor_stats(v) if torch.is_tensor(v) else type(v).__name__}")  
            except Exception:  
                pass  
    else:  
        info.append(f"  type={type(sample).__name__}")  
    return "\n".join(info)  
  
def test_lmdb_integrity(lmdb_path, max_samples=100):  
    """测试 LMDB 数据库完整性"""  
    print(f"\n=== 🔍 Testing LMDB integrity: {lmdb_path} ===")  
    try:  
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)  
        with env.begin() as txn:  
            meta = txn.get(b'__metadata__')  
            if meta is None:  
                print("❌ Missing metadata in LMDB")  
                return False  
              
            meta_dict = pickle.loads(meta)  
            total_samples = meta_dict.get('num_samples', 0)  
            print(f"✅ Metadata found: {total_samples} samples")  
              
            import random  
            test_indices = random.sample(range(total_samples), min(max_samples, total_samples))  
            corrupted = []  
              
            for idx in test_indices:  
                try:  
                    key = f'{idx}'.encode()  
                    data_bytes = txn.get(key)  
                    if data_bytes is None:  
                        corrupted.append((idx, "Missing key"))  
                        continue  
                    pkl_data = pickle.loads(data_bytes)  
                except Exception as e:  
                    corrupted.append((idx, str(e)))  
              
            if corrupted:  
                print(f"❌ Found {len(corrupted)} corrupted samples:")  
                for idx, err in corrupted[:10]:  
                    print(f"  Sample {idx}: {err}")  
                return False  
            else:  
                print(f"✅ All {len(test_indices)} tested samples are valid")  
                return True  
                  
    except Exception as e:  
        print(f"❌ LMDB error: {e}")  
        traceback.print_exc()  
        return False  
  
def main():  
    print("=" * 60)  
    print("🧩 AffinCraft DataLoader 崩溃诊断脚本")  
    print("=" * 60)  
    shm_status()  
    print(f"PID: {os.getpid()}\n")  
  
    lmdb_path = "/ssd/home/scw6f3q/train_lmdb"  
      
    # 1️⃣ 测试 LMDB 完整性  
    if not test_lmdb_integrity(lmdb_path):  
        print("\n⚠️  LMDB 数据库存在问题,请先修复")  
        return  
  
    # 2️⃣ 构造 Dataset  
    print(f"\n=== 📦 Loading dataset from {lmdb_path} ===")  
    try:  
        dataset = LMDBAffinCraftDataset(lmdb_path=lmdb_path)  
        print(f"✅ Loaded dataset with {len(dataset)} samples")  
    except Exception as e:  
        print(f"❌ [ERROR dataset init] {e}")  
        traceback.print_exc()  
        return  
  
    # 3️⃣ 测试单样本加载  
    print("\n=== 🔬 Testing individual sample loading ===")  
    test_indices = [0, len(dataset)//2, len(dataset)-1]  
    for idx in test_indices:  
        try:  
            sample = dataset[idx]  
            print(f"✅ Sample {idx} loaded successfully")  
            print(sample_summary(sample))  
        except Exception as e:  
            print(f"❌ Sample {idx} failed: {e}")  
            traceback.print_exc()  
  
    # 4️⃣ DataLoader 测试 - 使用简单的 collate  
    print("\n=== 🚀 Testing DataLoader ===")  
      
    def simple_collate(batch):  
        """简单的 collate 函数,避免复杂依赖"""  
        return batch  
      
    for num_workers in [0, 2, 4]:  
        print(f"\n--- Testing with num_workers={num_workers} ---")  
        loader = DataLoader(  
            dataset,  
            batch_size=4,  
            num_workers=num_workers,  
            collate_fn=simple_collate,  
            persistent_workers=False,  
            pin_memory=False,  
            shuffle=False,  
            timeout=30 if num_workers > 0 else 0,  
        )  
  
        try:  
            for i, batch in enumerate(loader):  
                if i % 50 == 0:  
                    shm_status()  
                    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9  
                    print(f"[STEP {i}] RAM={rss:.2f} GB")  
                  
                # 检查 batch 内容  
                for sample in batch:  
                    if isinstance(sample, dict):  
                        for k, v in sample.items():  
                            if torch.is_tensor(v):  
                                stats = tensor_stats(v)  
                                if "NaN=True" in stats or "Inf=True" in stats:  
                                    print(f"⚠️  Batch {i} {k}: {stats}")  
                  
                if i >= 200:  
                    break  
                      
            print(f"✅ Completed {i+1} batches with num_workers={num_workers}")  
              
        except Exception as e:  
            print(f"\n{'='*60}")  
            print(f"💥 CRASH DETECTED at batch {i} with num_workers={num_workers}")  
            print(f"{'='*60}")  
            print(f"Exception: {str(e)}")  
            traceback.print_exc()  
  
            print("\n--- 🔍 Single sample investigation ---")  
            start_idx = i * 4  
            for j in range(start_idx, min(start_idx + 4, len(dataset))):  
                try:  
                    s = dataset[j]  
                    print(f"✅ [Sample {j}] OK")  
                    print(sample_summary(s))  
                except Exception as ee:  
                    print(f"❌ [Sample {j}] ERROR: {ee}")  
                    traceback.print_exc()  
            print("="*60)  
            break  
  
    print("\n=== ✅ Diagnostic completed ===")  
  
if __name__ == "__main__":  
    main()