import re
import pandas as pd
import os
import openpyxl

def parse_log_file(input_path):
    """
    解析日志文件，提取每个攻击比例下的所有巨片大小。
    """
    # 定义正则表达式
    # 匹配 "初始攻击比例: 0.3750"
    proportion_re = re.compile(r"初始攻击比例: (\d+\.\d+)")
    # 匹配 "实验 1/150: ... 大小为 408"
    size_re = re.compile(r"实验 \d+/\d+: .*大小为 (\d+)")

    # 使用字典存储数据，键为攻击比例，值为巨片大小列表
    data = {}
    current_proportion = None

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 检查是否是新的攻击比例行
                proportion_match = proportion_re.search(line)
                if proportion_match:
                    current_proportion = proportion_match.group(1)
                    if current_proportion not in data:
                        data[current_proportion] = []
                    continue

                # 如果已确定当前攻击比例，则查找巨片大小
                if current_proportion:
                    size_match = size_re.search(line)
                    if size_match:
                        size = int(size_match.group(1))
                        data[current_proportion].append(size)
        
        # 验证数据完整性
        for prop, sizes in data.items():
            if len(sizes) != 150:
                print(f"警告: 攻击比例 {prop} 的数据不完整，只找到了 {len(sizes)}/150 条记录。")

        return data

    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {input_path}")
        return None

def save_data_to_excel(data, output_path):
    """
    将提取的数据保存到Excel文件。
    """
    if not data:
        print("没有数据可保存。")
        return

    # 创建一个pandas DataFrame
    # DataFrame的列是攻击比例，行是每次实验的结果
    df = pd.DataFrame(data)

    # 为了更清晰，给行添加索引标签
    num_experiments = len(next(iter(data.values())))
    df.index = [f"实验 {i+1}" for i in range(num_experiments)]

    # 将DataFrame保存到Excel文件
    try:
        df.to_excel(output_path, sheet_name='Giant_Component_Sizes')
        print(f"数据已成功保存到: {output_path}")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")


if __name__ == "__main__":
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建输入和输出文件的完整路径
    input_file = os.path.join(script_dir, 'out', 'ER_N4000_K4.txt')
    output_file = os.path.join(script_dir, 'out', 'ER_N4000_K4_giant_sizes.xlsx')

    # 1. 解析日志文件
    print(f"正在从 {input_file} 提取数据...")
    extracted_data = parse_log_file(input_file)

    # 2. 保存到Excel
    if extracted_data:
        save_data_to_excel(extracted_data, output_file)
