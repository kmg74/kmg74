#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import math
from collections import defaultdict
import cProfile
import pstats
import os
import argparse


def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错：{e}")
        sys.exit(1)


def preprocess_text(text):
    """预处理文本：去除标点、转小写、压缩空格"""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_ngrams(text, n=2):
    """获取文本的n-gram列表"""
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i + n])
    return ngrams


def calculate_cosine_similarity(text1, text2, n=2):
    """计算两个文本的余弦相似度"""
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)

    freq1 = defaultdict(int)
    for gram in ngrams1:
        freq1[gram] += 1

    freq2 = defaultdict(int)
    for gram in ngrams2:
        freq2[gram] += 1

    all_grams = set(freq1.keys()) | set(freq2.keys())

    dot_product = 0
    for gram in all_grams:
        dot_product += freq1[gram] * freq2[gram]

    magnitude1 = math.sqrt(sum(freq ** 2 for freq in freq1.values()))
    magnitude2 = math.sqrt(sum(freq ** 2 for freq in freq2.values()))

    if magnitude1 * magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def run_check(original_path, plagiarized_path, output_path):
    """核心查重逻辑（抽离出来以便性能分析）"""
    try:
        original_text = read_file(original_path)
        plagiarized_text = read_file(plagiarized_path)

        similarity = calculate_cosine_similarity(original_text, plagiarized_text)
        result = round(similarity, 2)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(result))
        print(f"查重完成，重复率: {result}")

    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)


def generate_performance_report(profile_file):
    """生成snakeviz交互式报告 + gprof2dot可视化图表"""
    try:
        # 1. 启动snakeviz交互式分析（后台运行）
        print("正在启动snakeviz交互式性能分析...")
        os.system(f"snakeviz {profile_file} &")

        # 2. 用gprof2dot生成DOT格式，再转PNG（需Graphviz）
        dot_file = profile_file.replace('.prof', '.dot')
        png_file = profile_file.replace('.prof', '.png')

        print("正在生成gprof2dot可视化图表...")
        os.system(f"python -m gprof2dot -f pstats {profile_file} > {dot_file}")
        os.system(f"dot -Tpng {dot_file} -o {png_file}")

        print(f"性能图表生成成功：{png_file}")
        print(f"DOT格式文件路径：{dot_file}")

    except Exception as e:
        print(f"生成性能报告失败: {e}")


def main():
    """主函数：支持普通运行和性能分析模式"""
    parser = argparse.ArgumentParser(description='论文查重程序（带性能分析）')
    parser.add_argument('original_path', help='原文文件路径')
    parser.add_argument('plagiarized_path', help='抄袭版文件路径')
    parser.add_argument('output_path', help='输出结果文件路径')
    parser.add_argument('--profile', action='store_true', help='启用性能分析')
    parser.add_argument('--profile-output', default='profile_output.prof',
                        help='性能分析结果保存路径（默认：profile_output.prof）')

    args = parser.parse_args()

    if args.profile:
        # 启用cProfile性能分析
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            run_check(args.original_path, args.plagiarized_path, args.output_path)
        finally:
            profiler.disable()
            profiler.dump_stats(args.profile_output)
            print(f"性能数据已保存至: {args.profile_output}")
            generate_performance_report(args.profile_output)
    else:
        # 普通运行（不分析性能）
        run_check(args.original_path, args.plagiarized_path, args.output_path)


if __name__ == "__main__":
    main()
