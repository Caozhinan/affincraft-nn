#!/usr/bin/env python3  
import pickle  
import numpy as np  
# import torch  
from pathlib import Path  
  
def check_nan_in_large_pkl_streaming(pkl_file_path, output_file="nan_samples.txt"):  
    """  
    逐个加载 PKL 文件中的对象,检测 NaN 值  
    适用于超大文件(如 4TB),不会一次性加载到内存  
    """  
    print(f"开始扫描 PKL 文件: {pkl_file_path}")  
    print(f"文件大小: {Path(pkl_file_path).stat().st_size / (1024**4):.2f} TB")  
      
    nan_samples = []  
    total_count = 0  
      
    with open(pkl_file_path, 'rb') as f:  
        with open(output_file, 'w') as out_f:  
            out_f.write(f"PKL 文件: {pkl_file_path}\n")  
            out_f.write("=" * 80 + "\n\n")  
              
            while True:  
                try:  
                    # 逐个加载对象  
                    pkl_data = pickle.load(f)  
                    total_count += 1  
                      
                    # 每处理 100 个样本打印进度  
                    if total_count % 100 == 0:  
                        print(f"已处理 {total_count} 个样本...")  
                      
                    # 检查是否为字典  
                    if not isinstance(pkl_data, dict):  
                        print(f"警告: 样本 {total_count} 不是字典类型,跳过")  
                        continue  
                      
                    pdbid = pkl_data.get('pdbid', f'sample_{total_count}')  
                    has_nan = False  
                    nan_fields = []  
                      
                    # 只检查 numpy 数组字段  
                    for key, value in pkl_data.items():  
                        # 关键修改:只对 numpy 数组和浮点数标量检查 NaN  
                        if isinstance(value, np.ndarray):  
                            # 只对浮点类型数组检查 NaN  
                            if np.issubdtype(value.dtype, np.floating):  
                                if np.isnan(value).any():  
                                    nan_count = np.isnan(value).sum()  
                                    total_elements = value.size  
                                    nan_percentage = (nan_count / total_elements) * 100  
                                      
                                    nan_fields.append({  
                                        'field': key,  
                                        'shape': value.shape,  
                                        'dtype': value.dtype,  
                                        'nan_count': nan_count,  
                                        'total_count': total_elements,  
                                        'nan_percentage': nan_percentage  
                                    })  
                                    has_nan = True  
                        elif isinstance(value, (float, np.floating)):  
                            # 检查标量浮点数  
                            if np.isnan(value):  
                                nan_fields.append({  
                                    'field': key,  
                                    'shape': 'scalar',  
                                    'dtype': type(value).__name__,  
                                    'nan_count': 1,  
                                    'total_count': 1,  
                                    'nan_percentage': 100.0  
                                })  
                                has_nan = True  
                      
                    # 如果发现 NaN,立即打印并记录  
                    if has_nan:  
                        print(f"\n发现 NaN 样本 #{total_count} (pdbid: {pdbid})")  
                        out_f.write(f"样本索引: {total_count}\n")  
                        out_f.write(f"PDB ID: {pdbid}\n")  
                        out_f.write(f"包含 NaN 的字段:\n")  
                          
                        for field_info in nan_fields:  
                            print(f"  字段: {field_info['field']}")  
                            print(f"    形状: {field_info['shape']}, 类型: {field_info['dtype']}")  
                            print(f"    NaN 数量: {field_info['nan_count']}/{field_info['total_count']} ({field_info['nan_percentage']:.2f}%)")  
                              
                            out_f.write(f"  - {field_info['field']}\n")  
                            out_f.write(f"      形状: {field_info['shape']}, 类型: {field_info['dtype']}\n")  
                            out_f.write(f"      NaN: {field_info['nan_count']}/{field_info['total_count']} ({field_info['nan_percentage']:.2f}%)\n")  
                          
                        out_f.write("\n" + "-" * 80 + "\n\n")  
                        out_f.flush()  # 立即写入磁盘  
                          
                        nan_samples.append({  
                            'index': total_count,  
                            'pdbid': pdbid,  
                            'nan_fields': nan_fields  
                        })  
                  
                except EOFError:  
                    # 文件读取完毕  
                    break  
                except Exception as e:  
                    print(f"错误: 处理样本 {total_count} 时出错: {str(e)}")  
                    continue  
      
    print(f"\n" + "=" * 80)  
    print(f"扫描完成!")  
    print(f"总样本数: {total_count}")  
    print(f"包含 NaN 的样本数: {len(nan_samples)} ({len(nan_samples)/total_count*100:.2f}%)")  
    print(f"详细报告已保存到: {output_file}")  
      
    return nan_samples  
  
if __name__ == "__main__":  
    # 修改为您的 PKL 文件路径  
    PKL_FILE = "/data/run01/scw6f3q/zncao/data_pkl/train.pkl"  
      
    nan_samples = check_nan_in_large_pkl_streaming(PKL_FILE)