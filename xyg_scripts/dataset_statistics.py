"""_summary_
    现在我有五个数据集，分别是 libero_spatial, libero_goal, libero_object, libero_90, libero_10
    他们的配置文件 PDDL 都在 LIBERO/libero/libero/bddl_files下
    以libero_spatial为例，他的PDDL文件位于 LIBERO/libero/libero/bddl_files/libero_spatial 中
        在该文件夹中，有若干 bddl 文件，每个 bddl 文件，描述一个 task，其中bddl文件格式如下：
        (define (problem LIBERO_Tabletop_Manipulation)
        (:domain LIBERO_Tabletop_Manipulation)
        (:objects
        ...
        )
        (:init
        ...
        )
        (:goal
        ...
        )
        (:fixture
        ...
        )
    我想统计每个数据集中，它的bddl的信息，比如problem，objects，goal，fixture等等
"""

#!/usr/bin/env python3
"""
LIBERO BDDL 数据集统计脚本

该脚本分析 LIBERO 的 BDDL 文件，统计每个数据集的详细信息，包括：
- 任务数量
- 对象类型和数量
- 目标类型和数量
- 固定装置信息
- 谓词使用频率
- 其他元数据

"""

import os
import re
import json
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

import libero.libero.envs.bddl_utils as BDDLUtils


# 定义 BDDL 文件位置
BASE_PATH = "LIBERO/libero/libero/bddl_files"
# DATASETS = ["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"]
DATASETS = ["libero_spatial"]


def parse_bddl_file(file_path):
    """解析单个 BDDL 文件并提取关键信息"""
    # ['problem_name', 'fixtures', 'regions', 'objects', 'scene_properties', 
    # 'initial_state', 'goal_state', 'language_instruction', 'obj_of_interest']
    bddl_file = BDDLUtils.robosuite_parse_problem(file_path)
    problem_name = bddl_file['problem_name']
    fixtures = bddl_file['fixtures']    # dict {fixture_type: [fixture_name1, fixture_name2, ...]}
    regions = bddl_file['regions']      # dict {region_type: {'target', 'ranges', 'extra', 'yaw_rotation', 'rgba'}}
    objects = bddl_file['objects']      # dict {object_type: [object_name1, object_name2, ...]}
    scene_properties = bddl_file['scene_properties']
    initial_state = bddl_file['initial_state']      # list[['predicate', 'object1', 'object2', ...]]
    goal_state = bddl_file['goal_state']            # list[['predicate', 'object1', 'object2', ...]]
    language_instruction = bddl_file['language_instruction']    # list[str]
    obj_of_interest = bddl_file['obj_of_interest']        # list[str]
    
    return bddl_file


def analyze_dataset(dataset_name):
    """分析指定数据集中的所有 BDDL 文件"""
    dataset_path = os.path.join(BASE_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return {
            "dataset_name": dataset_name,
            "error": "数据集路径不存在"
        }
    
    # 获取所有 BDDL 文件
    bddl_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.bddl'):
                bddl_files.append(os.path.join(root, file))
    
    if not bddl_files:
        print(f"数据集 {dataset_name} 中没有找到 BDDL 文件")
        return {
            "dataset_name": dataset_name,
            "error": "没有找到 BDDL 文件"
        }
    
    # 解析所有文件
    print(f"正在分析数据集 {dataset_name} 中的 {len(bddl_files)} 个 BDDL 文件...")
    file_data_list = []
    for file_path in bddl_files:
        data = parse_bddl_file(file_path)
        file_data_list.append(data)
    
    """
    现在我们的file_data 是一个列表，列表中的每个元素是一个字典，字典的键值对如下：
        file_data: List of [bddl_file]
        bddl_file = BDDLUtils.robosuite_parse_problem(file_path)
        其中bddl_file 是一个字典，字典的键值对如下：
        problem_name = bddl_file['problem_name']
        fixtures = bddl_file['fixtures']    # dict {fixture_type: [fixture_name1, fixture_name2, ...]}
        regions = bddl_file['regions']      # dict {region_type: {'target', 'ranges', 'extra', 'yaw_rotation', 'rgba'}}
        objects = bddl_file['objects']      # dict {object_type: [object_name1, object_name2, ...]}
        scene_properties = bddl_file['scene_properties']
        initial_state = bddl_file['initial_state']      # list[['predicate', 'object1', 'region1', ...]], 有时候可能不是个region，直接 ['on', 'akita_black_bowl_1', 'cookies_1']
        goal_state = bddl_file['goal_state']            # list[['predicate', 'object1', 'object2', ...]]
        language_instruction = bddl_file['language_instruction']    # list[str]
        obj_of_interest = bddl_file['obj_of_interest']        # list[str]
    """
    for data in file_data_list:
        # print('problem_name: ', data['problem_name'])
        # print('language_instruction: ', ' '.join(data['language_instruction']))
        print('goal_state: ', data['goal_state'])
    
    
    return
    # 统计数据集中，objects 有哪些, fixtures 有哪些, regions 有哪些
    tmp_objects = []
    for data in file_data_list:
        # print('problem_name: ', data['problem_name'])
        # print('language_instruction: ', ' '.join(data['language_instruction']))
        # print('goal_state: ', data['goal_state'])
        fixtures = []
        for _, v in data['fixtures'].items():
            fixtures.extend(v)
        objects = []
        for _, v in data['objects'].items():
            objects.extend(v)
        # print('fixtures: ', sorted(fixtures))
        print('objects: ', sorted(objects))
        tmp_objects.extend(objects)
    
    for data in file_data_list:
        # print('problem_name: ', data['problem_name'])
        print('language_instruction: ', ' '.join(data['language_instruction']))
        # print('goal_state: ', data['goal_state'])
    
    # 统计tmp_objects 中每个object 出现的次数
    object_count = Counter(tmp_objects)
    print('object_count: ', object_count)
        
        
    target_objects = ['main_table', 'wooden_cabinet_1', 'flat_stove_1'] + \
        ['akita_black_bowl_1', 'akita_black_bowl_2', 'cookies_1', 'glazed_rim_porcelain_ramekin_1', 'plate_1']
    
    target_objects = []
    for target_object in target_objects:
        for data in file_data_list:
            # print('problem_name: ', data['problem_name'])
            # print('fixtures: ', [v for _, v in data['fixtures'].items()])
            # print('objects: ', [v for _, v in data['objects'].items()])
            # print('language_instruction: ', ' '.join(data['language_instruction']))
            # print('goal_state: ', data['goal_state'])
            objects = data['objects']
            fixtures = data['fixtures']
            
            new_objects = []
            for _, v in objects.items():
                new_objects.extend(v)
            objects = new_objects
            
            new_fixtures = []
            for _, v in fixtures.items():
                new_fixtures.extend(v)
            fixtures = new_fixtures
            
            regions = data['regions']
            initial_state = data['initial_state']
            # 通过 initial_state 查看object 是怎么和 regions 关联的

            # target_object = 'akita_black_bowl_1'
            for fixture in fixtures:
                for region in initial_state:
                    if fixture == region[1] and fixture == target_object:
                        try:
                            target_region = regions[region[2]]['target']
                            range_region = regions[region[2]]['ranges']
                            print(f"{fixture} {region[0]} {region[2]} {target_region} {range_region}")
                        except:
                            # import ipdb; ipdb.set_trace()
                            print(f"{fixture} {region[0]} {region[2]}")
            
            for object in objects:
                for region in initial_state:
                    if object == region[1] and object == target_object:
                        # region[2]
                        try:
                            target_region = regions[region[2]]['target']
                            range_region = regions[region[2]]['ranges']
                            print(f"{object} {region[0]} {region[2]} {target_region} {range_region}")
                        except:
                            # import ipdb; ipdb.set_trace()
                            print(f"{object} {region[0]} {region[2]}")
    

def main():
    # 分析所有数据集
    all_datasets_stats = []
    for dataset_name in DATASETS:   # libero_spatial, libero_goal, libero_object, libero_90, libero_10
        stats = analyze_dataset(dataset_name)
        all_datasets_stats.append(stats)
        


if __name__ == "__main__":
    main()
