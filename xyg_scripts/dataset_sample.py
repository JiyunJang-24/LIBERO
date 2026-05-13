#!/usr/bin/env python3
"""
datasets_vis中查看数据集

dir: datasets_vis/video_lan
subdir: libero_10, libero_90, libero_spatial, libero_object, libero_goal, 分别表示五个数据集
每个数据集中，存在 subdirs, 表示数据集中的若干任务，每个任务中，存在若干视频，*.gif和唯一的task_description.txt

现在对于每个数据集，所有的任务可视化。对于每个任务，选择一个视频 demo_0.gif，保存到 LIBERO/xyg_scripts/datasets_sample
"""

import os
import shutil
from pathlib import Path
import glob
import argparse
from tqdm import tqdm
import pandas as pd

# 定义数据集路径和目标路径
SOURCE_DIR = "datasets_vis/video_lan"
DATASETS = ["libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal"]
TARGET_DIR = "LIBERO/xyg_scripts/datasets_sample"


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(directory, exist_ok=True)


def sample_dataset(dataset_name, source_base, target_base, overwrite=False):
    """
    从指定数据集中采样示例视频
    
    参数:
        dataset_name: 数据集名称
        source_base: 源目录基路径
        target_base: 目标目录基路径
        overwrite: 是否覆盖已存在的文件
    
    返回:
        采样任务数量
    """
    # 构造数据集路径
    dataset_path = os.path.join(source_base, dataset_name)
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"数据集 {dataset_name} 路径不存在: {dataset_path}")
        return 0
    
    # 创建目标数据集目录
    target_dataset_dir = os.path.join(target_base, dataset_name)
    ensure_dir(target_dataset_dir)
    
    # 获取任务列表 (子目录)
    tasks = [d for d in os.listdir(dataset_path) 
             if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not tasks:
        print(f"数据集 {dataset_name} 中没有找到任务")
        return 0
    
    # 存储任务信息的列表，用于生成汇总文件
    task_info = []
    
    # 采样每个任务的demo_0.gif
    sampled_count = 0
    for task in tqdm(tasks, desc=f"采样 {dataset_name}"):
        task_path = os.path.join(dataset_path, task)
        
        # 查找 demo_0.gif 文件
        demo_file = os.path.join(task_path, "demo_0.gif")
        
        # 如果没有 demo_0.gif，尝试查找任何 gif 文件
        if not os.path.exists(demo_file):
            gif_files = glob.glob(os.path.join(task_path, "*.gif"))
            if gif_files:
                demo_file = gif_files[0]
            else:
                print(f"  警告: 在任务 {task} 中没有找到GIF文件")
                continue
        
        # 获取任务描述（如果存在）
        description_file = os.path.join(task_path, "task_description.txt")
        description = "No description available"
        if os.path.exists(description_file):
            with open(description_file, 'r', encoding='utf-8') as f:
                description = f.read().strip()
        
        # 目标文件路径
        target_gif_file = os.path.join(target_dataset_dir, f"{task}.gif")
        
        # 如果文件已存在且不覆盖，则跳过
        if os.path.exists(target_gif_file) and not overwrite:
            print(f"  文件已存在 (跳过): {target_gif_file}")
            # 仍然加入到任务信息中
            task_info.append({
                "task": task,
                "description": description,
                "gif": os.path.basename(target_gif_file)
            })
            sampled_count += 1
            continue
        
        # 复制 GIF 文件
        try:
            shutil.copy2(demo_file, target_gif_file)
            print(f"  采样: {task} -> {os.path.basename(target_gif_file)}")
            
            # 记录任务信息
            task_info.append({
                "task": task,
                "description": description,
                "gif": os.path.basename(target_gif_file)
            })
            
            # 将 description 也写入到 target_gif_file = os.path.join(target_dataset_dir, f"{task}.txt")
            with open(os.path.join(target_dataset_dir, f"{task}.txt"), 'w', encoding='utf-8') as f:
                f.write(description)
            
            sampled_count += 1
        except Exception as e:
            print(f"  错误: 复制 {demo_file} 时出错: {str(e)}")
    
    # 创建任务信息CSV文件
    if task_info:
        csv_path = os.path.join(target_dataset_dir, f"{dataset_name}_tasks.csv")
        df = pd.DataFrame(task_info)
        df.to_csv(csv_path, index=False)
        print(f"任务信息已保存到: {csv_path}")
    
    return sampled_count

def create_html_gallery(target_base):
    """为所有采样数据创建HTML图库"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LIBERO Dataset Samples</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
            h1, h2, h3 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            .dataset-section { margin-bottom: 40px; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); grid-gap: 20px; margin-top: 20px; }
            .sample-item { border: 1px solid #ddd; border-radius: 5px; overflow: hidden; background-color: white; transition: transform 0.2s; }
            .sample-item:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .sample-item img { width: 100%; height: 200px; object-fit: cover; display: block; }
            .sample-info { padding: 15px; }
            .sample-info h3 { margin-top: 0; margin-bottom: 10px; font-size: 18px; }
            .sample-info p { margin: 0; font-size: 14px; color: #666; max-height: 80px; overflow: auto; }
            .nav { position: sticky; top: 0; background-color: #2c3e50; padding: 10px 0; margin-bottom: 20px; z-index: 1000; }
            .nav ul { display: flex; list-style: none; padding: 0; margin: 0; justify-content: center; }
            .nav ul li { margin: 0 15px; }
            .nav ul li a { color: white; text-decoration: none; font-weight: bold; }
            .nav ul li a:hover { text-decoration: underline; }
            .footer { margin-top: 40px; text-align: center; font-size: 14px; color: #777; }
        </style>
    </head>
    <body>
        <div class="nav">
            <ul>
    """
    
    # 添加导航链接
    for dataset in DATASETS:
        html_content += f'                <li><a href="#{dataset}">{dataset}</a></li>\n'
    
    html_content += """
            </ul>
        </div>
        <div class="container">
            <h1>LIBERO Dataset Samples</h1>
            <p>This gallery shows sample GIFs from each task in the LIBERO datasets.</p>
    """
    
    # 为每个数据集创建部分
    for dataset in DATASETS:
        dataset_dir = os.path.join(target_base, dataset)
        if not os.path.exists(dataset_dir):
            continue
        
        # 读取任务信息CSV（如果存在）
        tasks_info = []
        csv_path = os.path.join(dataset_dir, f"{dataset}_tasks.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                tasks_info = df.to_dict('records')
            except:
                # 如果CSV读取失败，尝试从文件名获取信息
                gif_files = glob.glob(os.path.join(dataset_dir, "*.gif"))
                for gif in gif_files:
                    tasks_info.append({
                        "task": os.path.splitext(os.path.basename(gif))[0],
                        "description": "No description available",
                        "gif": os.path.basename(gif)
                    })
        else:
            # 如果CSV不存在，尝试从文件名获取信息
            gif_files = glob.glob(os.path.join(dataset_dir, "*.gif"))
            for gif in gif_files:
                tasks_info.append({
                    "task": os.path.splitext(os.path.basename(gif))[0],
                    "description": "No description available",
                    "gif": os.path.basename(gif)
                })
        
        # 添加数据集部分
        html_content += f"""
            <div class="dataset-section" id="{dataset}">
                <h2>{dataset}</h2>
                <p>Number of tasks: {len(tasks_info)}</p>
                <div class="samples-grid">
        """
        
        # 添加每个任务的样本
        for task_info in tasks_info:
            task = task_info.get("task", "Unknown")
            description = task_info.get("description", "No description available")
            gif = task_info.get("gif", "")
            
            if not gif:
                continue
                
            gif_path = os.path.join(dataset, gif)
            
            html_content += f"""
                    <div class="sample-item">
                        <img src="{gif_path}" alt="{task}">
                        <div class="sample-info">
                            <h3>{task}</h3>
                            <p>{description[:150]}{'...' if len(description) > 150 else ''}</p>
                        </div>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
    
    # 添加页脚
    html_content += """
            <div class="footer">
                <p>Generated by LIBERO Dataset Sample Script</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    html_path = os.path.join(target_base, "dataset_samples_gallery.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML图库已生成: {html_path}")
    return html_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LIBERO数据集样本采集工具')
    parser.add_argument('--source', default=SOURCE_DIR, help='源数据集目录')
    parser.add_argument('--target', default=TARGET_DIR, help='采样目标目录')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的文件')
    parser.add_argument('--gallery', action='store_true', help='生成HTML图库')
    args = parser.parse_args()
    
    # 确保目标目录存在
    ensure_dir(args.target)
    
    # 记录采样统计
    dataset_stats = {}
    
    # 对每个数据集进行采样
    total_sampled = 0
    for dataset in DATASETS:
        print(f"\n开始采样数据集: {dataset}")
        sampled_count = sample_dataset(dataset, args.source, args.target, args.overwrite)
        dataset_stats[dataset] = sampled_count
        total_sampled += sampled_count
        print(f"完成 {dataset} 采样: {sampled_count} 个任务")
    
    # 打印总结
    print("\n" + "="*50)
    print("采样完成! 总结:")
    print("="*50)
    for dataset, count in dataset_stats.items():
        print(f"{dataset}: {count} 个任务")
    print(f"总计: {total_sampled} 个任务样本")
    
    # 生成HTML图库
    if args.gallery:
        html_path = create_html_gallery(args.target)
        print(f"\n已生成HTML图库: {html_path}")


if __name__ == "__main__":
    main()