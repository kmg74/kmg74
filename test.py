#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文查重程序测试文件（修复版）
测试范围：文件操作、文本预处理、n-gram频率、余弦相似度、异常处理
"""
import unittest
import tempfile
import os
import sys
# 导入原程序的核心函数和异常类（需确保原程序与测试文件在同一目录）
from maintest import (  # 原程序文件名需为 paper_check.py
    PaperCheckException,
    validate_file_path,
    read_file,
    preprocess_text,
    get_ngram_frequency,
    calculate_cosine_similarity,
    write_result
)


class TestPaperCheck(unittest.TestCase):
    """论文查重程序测试类"""

    # ------------------------------
    # 1. 测试文件操作相关
    # ------------------------------
    def test_read_file_normal(self):
        """测试正常读取文件（含中文、标点、空格）"""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp:
            temp.write("Python 是一门优秀的编程语言！\n它简单易学，功能强大。")
            temp_path = temp.name

        try:
            content = read_file(temp_path)
            self.assertEqual(content, "Python 是一门优秀的编程语言！\n它简单易学，功能强大。")
        finally:
            os.unlink(temp_path)

    def test_read_file_not_exists(self):
        """测试读取不存在的文件（异常场景）"""
        non_exist_path = "不存在的文件.txt"
        with self.assertRaises(PaperCheckException) as ctx:
            read_file(non_exist_path)
        self.assertIn("文件不存在", str(ctx.exception))

    def test_read_file_over_10mb(self):
        """测试读取超过10MB的文件（异常场景）"""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp:
            temp.write(' ' * (11 * 1024 * 1024))  # 11MB内容
            temp_path = temp.name

        try:
            with self.assertRaises(PaperCheckException) as ctx:
                read_file(temp_path)
            self.assertIn("文件过大", str(ctx.exception))
        finally:
            os.unlink(temp_path)

    def test_write_result_normal(self):
        """测试正常写入结果（相似度0.85）"""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as temp:
            output_path = temp.name

        try:
            write_result(0.85, output_path)
            with open(output_path, 'r', encoding='utf-8') as f:
                result = f.read()
            self.assertEqual(result, "0.85")
        finally:
            os.unlink(output_path)

    # ------------------------------
    # 3. 测试n-gram频率统计
    # ------------------------------
    def test_get_ngram_frequency_normal(self):
        """测试正常n-gram频率统计（n=2）"""
        preprocessed_text = "hello world"
        expected_freq = {
            "he": 1, "el": 1, "ll": 1, "lo": 1, "o ": 1,
            " w": 1, "wo": 1, "or": 1, "rl": 1, "ld": 1
        }
        get_ngram_frequency.cache_clear()
        freq = get_ngram_frequency(preprocessed_text, n=2)
        self.assertEqual(dict(freq), expected_freq)

    def test_get_ngram_text_shorter_than_n(self):
        """测试文本长度小于n（n=3）"""
        get_ngram_frequency.cache_clear()
        freq = get_ngram_frequency("ab", n=3)  # 长度2 < 3
        self.assertEqual(dict(freq), {})

    # ------------------------------
    # 4. 测试余弦相似度计算
    # ------------------------------
    def test_cosine_similarity_full_match(self):
        """测试完全相同的文本（相似度1.0）"""
        text1 = "Python 是一门优秀的编程语言，简单易学。"
        text2 = "Python 是一门优秀的编程语言，简单易学。"
        similarity = calculate_cosine_similarity(text1, text2, n=2)
        self.assertAlmostEqual(similarity, 1.0, delta=0.01)

    def test_cosine_similarity_no_match(self):
        """测试完全不同的文本（无共同字符，相似度0.0）"""
        text1 = "Apple banana orange grape"  # 英文文本
        text2 = "汽车 飞机 火车 轮船"  # 中文文本（无共同字符）
        similarity = calculate_cosine_similarity(text1, text2, n=2)
        self.assertAlmostEqual(similarity, 0.0, delta=0.01)

    def test_cosine_similarity_partial_match(self):
        """测试部分相似的文本（相似度约0.65）"""
        text1 = "机器学习是人工智能的一个分支，研究计算机如何学习。"
        text2 = "机器学习是人工智能的重要分支，探索计算机的学习方法。"
        similarity = calculate_cosine_similarity(text1, text2, n=2)
        self.assertAlmostEqual(similarity, 0.65, delta=0.05)

    def test_cosine_similarity_empty_text(self):
        """测试空文本（相似度0.0）"""
        text1 = "正常文本"
        text2 = ""  # 空文本
        similarity = calculate_cosine_similarity(text1, text2, n=2)
        self.assertEqual(similarity, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
