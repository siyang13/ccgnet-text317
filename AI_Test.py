#!/usr/bin/env python3
"""
千问共晶分析修复版 - 修复网络和API问题
"""

import requests
import json
import time
import sys
import os
from datetime import datetime
import socket


def check_network_connection():
    """检查网络连接"""
    print("检查网络连接...")

    # 测试多个域名
    test_urls = [
        ("DashScope API", "dashscope.aliyuncs.com", 443),
        ("Google", "8.8.8.8", 53),  # Google DNS
        ("百度", "www.baidu.com", 80)
    ]

    for name, host, port in test_urls:
        try:
            socket.create_connection((host, port), timeout=5)
            print(f"  ✓ {name} 连接正常")
            return True
        except Exception as e:
            print(f"  ✗ {name} 连接失败: {e}")

    return False


def test_api_key(api_key):
    """测试API密钥有效性"""
    print("测试API密钥...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 简单测试请求
    payload = {
        "model": "qwen-turbo",  # 使用更轻量的模型测试
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "请回复'test ok'"
                }
            ]
        }
    }

    try:
        response = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers=headers,
            json=payload,
            timeout=30
        )

        print(f"  测试响应状态码: {response.status_code}")

        if response.status_code == 200:
            print("  ✓ API密钥有效")
            return True
        elif response.status_code == 401:
            print("  ✗ API密钥无效")
            return False
        else:
            print(f"  ⚠ API响应异常: {response.status_code}")
            print(f"  响应内容: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"  ✗ API测试失败: {type(e).__name__}: {e}")
        return False


def analyze_molecules(api_key, input_file, output_file, max_lines=3):
    """
    分析分子对

    Args:
        api_key: API密钥
        input_file: 输入文件
        output_file: 输出文件
        max_lines: 最大分析行数
    """

    # 读取输入文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 限制行数
    lines = lines[:max_lines]

    print(f"\n开始分析 {len(lines)} 对分子...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    results.append("=" * 80)
    results.append("千问共晶分析结果")
    results.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results.append(f"输入文件: {input_file}")
    results.append("=" * 80)

    # API配置
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for i, line in enumerate(lines):
        parts = line.split('\t')
        if len(parts) < 2:
            continue

        mol1 = parts[0].strip()
        mol2 = parts[1].strip()

        print(f"\n[{i + 1}/{len(lines)}] 分析: {mol1} + {mol2}")

        # 构建更简单的提示词
        prompt = f"""请分析这两个分子是否能形成共晶：
        分子1: {mol1}
        分子2: {mol2}

        请给出明确的"结论: 是/否"，并简要说明理由。"""

        payload = {
            "model": "qwen-max",
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一名专业的化学家，专门研究共晶系统。请给出准确的分析。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "max_tokens": 500,
                "temperature": 0.3
            }
        }

        try:
            print("  发送API请求...")
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=payload,
                timeout=60  # 增加超时时间
            )

            print(f"  响应状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # 提取分析结果
                analysis = "未获取到分析结果"
                if 'output' in data and 'choices' in data['output']:
                    choices = data['output']['choices']
                    if choices and len(choices) > 0:
                        analysis = choices[0]['message']['content']

                # 判断结论
                conclusion = "Unknown"
                analysis_lower = analysis.lower()

                if "结论: 是" in analysis_lower or "结论：是" in analysis_lower:
                    conclusion = "Yes"
                elif "结论: 否" in analysis_lower or "结论：否" in analysis_lower:
                    conclusion = "No"
                elif "可以形成" in analysis:
                    conclusion = "Yes"
                elif "不能形成" in analysis or "无法形成" in analysis:
                    conclusion = "No"

                # 记录结果
                results.append(f"\n分子对 {i + 1}:")
                results.append(f"  {mol1}")
                results.append(f"  {mol2}")
                results.append(f"结论: {conclusion}")
                results.append(f"分析:\n{analysis}")
                results.append("-" * 60)

                print(f"  结论: {conclusion}")
                print(f"  分析摘要: {analysis[:80]}...")

            elif response.status_code == 429:
                print("  达到速率限制，等待10秒...")
                time.sleep(10)
                continue
            else:
                print(f"  API错误: {response.status_code}")
                print(f"  错误信息: {response.text[:200]}")
                results.append(f"\n分子对 {i + 1}: {mol1} + {mol2}")
                results.append(f"错误: API返回 {response.status_code}")

        except requests.exceptions.Timeout:
            print("  请求超时")
            results.append(f"\n分子对 {i + 1}: {mol1} + {mol2}")
            results.append("错误: 请求超时")
        except requests.exceptions.ConnectionError:
            print("  连接错误")
            results.append(f"\n分子对 {i + 1}: {mol1} + {mol2}")
            results.append("错误: 连接错误")
            time.sleep(5)
        except Exception as e:
            print(f"  请求异常: {e}")
            results.append(f"\n分子对 {i + 1}: {mol1} + {mol2}")
            results.append(f"错误: {type(e).__name__}: {str(e)}")

        # 请求间隔
        time.sleep(3)

    # 保存结果
    try:
        # 使用绝对路径确保能保存
        if output_file.startswith("/"):
            # Linux路径，转换为Windows路径
            output_file = os.path.join(os.getcwd(), output_file.lstrip("/").replace("/", "\\"))

        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))

        print(f"\n✓ 分析完成!")
        print(f"结果保存到: {output_file}")
        print(f"文件大小: {os.path.getsize(output_file)} 字节")

    except Exception as e:
        print(f"\n✗ 保存结果出错: {e}")

        # 尝试保存到当前目录
        backup_file = "AI_analysis_backup.txt"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(results))
            print(f"结果已备份到: {backup_file}")
        except:
            print("无法保存任何结果")


def main():
    """主函数"""
    print("=" * 60)
    print("千问共晶分析工具")
    print("=" * 60)

    # API密钥
    API_KEY = "sk-324133baaa3645adb409411db3ff0c4a"

    # 检查网络连接
    if not check_network_connection():
        print("\n⚠ 网络连接可能有问题，但继续尝试...")

    # 测试API密钥
    if not test_api_key(API_KEY):
        print("\n✗ API密钥测试失败，请检查:")
        print("  1. API密钥是否正确")
        print("  2. 是否有足够的余额")
        print("  3. 是否开通了相关服务")
        print("\n可以访问 https://dashscope.console.aliyun.com/ 检查")
        return

    # 输入文件路径
    input_file = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/CC_Table/CC_Table_converted.tab"

    # 如果文件不存在，尝试其他路径
    if not os.path.exists(input_file):
        print(f"\n⚠ 找不到文件: {input_file}")

        # 尝试其他可能的位置
        possible_paths = [
            "CC_Table_converted.tab",
            "data/CC_Table/CC_Table_converted.tab",
            "../data/CC_Table/CC_Table_converted.tab"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                input_file = path
                print(f"找到文件: {input_file}")
                break
        else:
            print("请手动输入文件路径: ", end="")
            input_file = input().strip()
            if not os.path.exists(input_file):
                print("文件不存在，程序退出")
                return

    # 输出文件
    output_file = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/AI/AI_analysis_test.txt"

    print(f"\n配置信息:")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  API密钥: {API_KEY[:8]}...")
    print(f"  最大分析数: 3 (测试模式)")

    # 确认开始
    confirm = input("\n是否开始分析? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消分析")
        return

    # 开始分析
    analyze_molecules(API_KEY, input_file, output_file, max_lines=3)

    print("\n" + "=" * 60)
    print("分析流程结束")


if __name__ == "__main__":
    main()