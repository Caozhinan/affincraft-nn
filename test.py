# 在登录节点的命令行中输入 python
import pickle

file_path = '/data/run01/scw6f3q/zncao/data_pkl/train.pkl'

try:
    with open(file_path, 'rb') as f:
        # 尝试只加载一次
        data = pickle.load(f)
        print("成功加载一个对象！")
        print(f"对象的类型是: {type(data)}")
        # 如果是列表或字典，可以看看它有多大
        if hasattr(data, '__len__'):
            print(f"对象包含的元素数量: {len(data)}")

except Exception as e:
    print(f"加载失败: {e}")