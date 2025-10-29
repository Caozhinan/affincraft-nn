from graphormer.data.affincraft_dataset import LMDBAffinCraftDataset  
dataset = LMDBAffinCraftDataset('/ssd/home/scw6f3q/lmdb/valid.lmdb')  
for i in range(10):  
    try:  
        sample = dataset[i]  
        print(f"样本 {i}: OK")  
    except Exception as e:  
        print(f"样本 {i}: 失败 - {e}")