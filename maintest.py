#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
论文查重程序 - 性能优化版
功能：计算两篇论文的相似度，并将结果输出到指定文件
算法：基于字符级n-gram和余弦相似度
作者：AI助手
版本：3.0 (性能优化版)
"""

import sys
import re
import math
import os
import argparse
import cProfile
import pstats
from collections import Counter
from functools import lru_cache

class PaperCheckException(Exception):
    """自定义异常类，用于论文查重程序中的特定异常"""
    pass
def validate_file_path(file_path, is_output=False):
    """验证文件路径的有效性"""
    if not file_path:
        raise PaperCheckException("文件路径不能为空")

    if not isinstance(file_path, str):
        raise PaperCheckException("文件路径必须是字符串")

    # 检查路径长度
    if len(file_path) > 260:  # Windows路径长度限制
        raise PaperCheckException("文件路径过长")

    # 对于输出文件，检查目录是否存在且可写
    if is_output:
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            raise PaperCheckException(f"输出目录不存在: {dir_path}")
        if dir_path and not os.access(dir_path, os.W_OK):
            raise PaperCheckException(f"输出目录不可写: {dir_path}")
    # 对于输入文件，检查文件是否存在且可读
    else:
        if not os.path.exists(file_path):
            raise PaperCheckException(f"文件不存在: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PaperCheckException(f"文件不可读: {file_path}")


def read_file(file_path):
    """读取文件内容，包含详细的异常处理"""
    try:
        validate_file_path(file_path, is_output=False)

        # 检查文件大小（限制为10MB）
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise PaperCheckException(f"文件过大 ({file_size}字节)，超过10MB限制")

        # 检查文件是否为空
        if file_size == 0:
            raise PaperCheckException("文件为空")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # 检查读取的内容是否为空
            if not content.strip():
                raise PaperCheckException("文件内容为空或只包含空白字符")

            return content

    except PaperCheckException:
        raise
    except UnicodeDecodeError:
        raise PaperCheckException("文件编码不是UTF-8，无法读取")
    except Exception as e:
        raise PaperCheckException(f"读取文件时发生未知错误: {str(e)}")


@lru_cache(maxsize=128)
def preprocess_text(text):
    """预处理文本：去除标点符号和多余空格，转换为小写"""
    if not isinstance(text, str):
        raise PaperCheckException("预处理文本必须是字符串")

    try:
        # 使用正则表达式移除非单词字符（包括标点符号）
        text = re.sub(r'[^\w\s]', '', text)
        # 将所有字符转换为小写，统一文本格式
        text = text.lower()
        # 将多个连续空白字符替换为单个空格，并去除首尾空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        raise PaperCheckException(f"文本预处理失败: {str(e)}")


@lru_cache(maxsize=128)
def get_ngram_frequency(text, n=2):
    """
    直接生成n-gram频率计数器，而不构建n-gram列表

    参数:
        text (str): 预处理后的文本
        n (int): n-gram的大小，默认为2（二元组）

    返回:
        Counter: n-gram频率计数器
    """
    if not isinstance(text, str):
        raise PaperCheckException("文本必须是字符串")

    if not isinstance(n, int) or n < 1:
        raise PaperCheckException("n必须是正整数")

    if len(text) < n:
        return Counter()  # 返回空计数器而不是抛出异常

    try:
        freq = Counter()
        # 直接构建频率计数器，避免中间列表
        for i in range(len(text) - n + 1):
            freq[text[i:i + n]] += 1
        return freq
    except Exception as e:
        raise PaperCheckException(f"提取n-gram频率失败: {str(e)}")


def calculate_cosine_similarity(text1, text2, n=2):
    """
    计算两个文本的余弦相似度（优化版）

    优化点：
    1. 使用缓存预处理和n-gram频率计算
    2. 只计算共同n-gram的点积，减少计算量
    3. 使用更高效的Counter数据结构
    """
    try:
        # 预处理文本（使用缓存）
        text1 = preprocess_text(text1)
        text2 = preprocess_text(text2)

        # 检查预处理后的文本是否为空
        if not text1 or not text2:
            return 0.0

        # 获取n-gram频率计数器（使用缓存）
        freq1 = get_ngram_frequency(text1, n)
        freq2 = get_ngram_frequency(text2, n)

        # 如果其中一个频率计数器为空，则相似度为0
        if not freq1 or not freq2:
            return 0.0

        # 计算点积：只遍历共同出现的n-gram
        common_grams = set(freq1.keys()) & set(freq2.keys())
        dot_product = sum(freq1[gram] * freq2[gram] for gram in common_grams)

        # 计算向量模长
        magnitude1 = math.sqrt(sum(freq ** 2 for freq in freq1.values()))
        magnitude2 = math.sqrt(sum(freq ** 2 for freq in freq2.values()))

        # 计算余弦相似度
        if magnitude1 * magnitude2 == 0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)

        # 确保相似度在合理范围内
        return max(0.0, min(1.0, similarity))

    except Exception as e:
        raise PaperCheckException(f"计算余弦相似度失败: {str(e)}")


def write_result(result, output_path):
    """将结果写入输出文件"""
    try:
        validate_file_path(output_path, is_output=True)

        # 确保结果是浮点数且在合理范围内
        if not isinstance(result, (int, float)):
            raise PaperCheckException("结果必须是数字")

        result = float(result)
        if result < 0 or result > 1:
            raise PaperCheckException("结果必须在0到1之间")

        # 格式化结果，保留两位小数
        formatted_result = round(result, 2)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(formatted_result))

    except PaperCheckException:
        raise
    except Exception as e:
        raise PaperCheckException(f"写入结果文件失败: {str(e)}")


def main():
    """主函数：程序入口点"""
    parser = argparse.ArgumentParser(description='论文查重程序')
    parser.add_argument('original_path', help='原文文件路径')
    parser.add_argument('plagiarized_path', help='抄袭版文件路径')
    parser.add_argument('output_path', help='输出结果文件路径')
    parser.add_argument('--profile', action='store_true', help='启用性能分析')
    parser.add_argument('--profile-output', default='profile_output.prof',
                        help='性能分析输出文件路径')

    args = parser.parse_args()

    # 如果启用了性能分析，使用 cProfile 运行
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            run_check(args.original_path, args.plagiarized_path, args.output_path)
        finally:
            profiler.disable()
            # 保存性能分析结果
            profiler.dump_stats(args.profile_output)
            print(f"性能分析结果已保存到: {args.profile_output}")

            # 生成性能分析报告
            generate_performance_report(args.profile_output)
    else:
        run_check(args.original_path, args.plagiarized_path, args.output_path)


def run_check(original_path, plagiarized_path, output_path):
    """运行论文查重的主要逻辑"""
    try:
        # 读取文件内容
        print(f"正在读取原文文件: {original_path}")
        original_text = read_file(original_path)

        print(f"正在读取抄袭版文件: {plagiarized_path}")
        plagiarized_text = read_file(plagiarized_path)

        # 计算相似度
        print("正在计算文本相似度...")
        similarity = calculate_cosine_similarity(original_text, plagiarized_text)

        # 写入输出文件
        print(f"正在将结果写入输出文件: {output_path}")
        write_result(similarity, output_path)

        print(f"查重完成，重复率: {similarity:.2f}")

    except PaperCheckException as e:
        print(f"错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"发生未知错误: {e}")
        sys.exit(1)


def generate_performance_report(profile_file):
    """生成性能分析报告"""
    try:
        # 使用 snakeviz 打开交互式报告
        print("正在生成交互式性能分析报告...")
        os.system(f"snakeviz {profile_file} &")

        # 使用 gprof2dot 生成可视化图表
        print("正在生成性能分析图表...")
        dot_file = profile_file.replace('.prof', '.dot')
        png_file = profile_file.replace('.prof', '.png')

        # 生成 DOT 文件
        os.system(f"python -m gprof2dot -f pstats {profile_file} > {dot_file}")

        # 生成 PNG 图像 (需要安装 Graphviz)
        os.system(f"dot -Tpng {dot_file} -o {png_file}")

        print(f"性能分析图表已生成: {png_file}")
        print(f"DOT 文件: {dot_file}")

    except Exception as e:
        print(f"生成性能分析报告时出错: {e}")


if __name__ == "__main__":
    main()
