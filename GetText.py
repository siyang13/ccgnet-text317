import requests
import json
import time
import logging
import sys
import os
from datetime import datetime
import hashlib
import csv
from typing import Dict, List, Tuple
import re

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resualt/eutectic_analysis_ECC.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QwenEutecticAnalyzer:
    def __init__(self, api_key: str, model: str = "qwen3-max"):  # 修改为qwen3-max
        """
        初始化千问分析器

        Args:
            api_key: 千问API密钥
            model: 使用的模型名称，默认qwen3-max
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 缓存设置
        self.cache_dir = "cache"
        self.cache_file = os.path.join(self.cache_dir, "eutectic_cache.json")
        self.cache = {}
        self.load_cache()

        # 统计信息
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "cached": 0,
            "rate_limited": 0,
            "bad_request": 0
        }

        # 输入文件路径，用于结果保存
        self.input_file = None

    def load_cache(self):
        """加载缓存"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"已加载缓存，包含 {len(self.cache)} 条记录")
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self.cache = {}

    def save_cache(self):
        """保存缓存"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info("缓存已保存")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def get_cache_key(self, mol1: str, mol2: str) -> str:
        """
        生成缓存键

        Args:
            mol1: 分子1名称
            mol2: 分子2名称

        Returns:
            str: 缓存键
        """
        # 排序确保(mol1, mol2)和(mol2, mol1)使用相同的缓存键
        sorted_pair = tuple(sorted([mol1, mol2]))
        key_str = f"{sorted_pair[0]}|{sorted_pair[1]}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def call_qwen_api(self, prompt: str, max_retries: int = 3) -> str:
        """
        调用千问API

        Args:
            prompt: 提示词
            max_retries: 最大重试次数

        Returns:
            str: API响应内容
        """
        # 检查prompt长度
        if len(prompt) > 3000:
            logger.warning(f"提示词过长 ({len(prompt)} 字符)，进行截断")
            prompt = prompt[:3000]

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的化学家，专门研究共晶体系和相图分析。请提供准确、专业的分析，使用英文回答。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "result_format": "message",
                "temperature": 0.2,
                "max_tokens": 1000,
                "top_p": 0.8,
                "repetition_penalty": 1.05
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=45
                )

                logger.debug(f"API响应状态码: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()

                    if 'output' in result:
                        output = result['output']

                        if 'text' in output:
                            return output['text']

                        elif 'choices' in output:
                            choices = output['choices']
                            if choices and len(choices) > 0:
                                choice = choices[0]
                                if 'message' in choice and 'content' in choice['message']:
                                    return choice['message']['content']
                                elif 'content' in choice:
                                    return choice['content']

                        else:
                            return str(output)

                    elif 'text' in result:
                        return result['text']

                    else:
                        logger.warning(f"API响应格式异常: {list(result.keys())}")
                        return "API response format error"

                elif response.status_code == 429:
                    self.stats["rate_limited"] += 1
                    wait_time = (2 ** attempt) * 10
                    logger.warning(f"速率限制，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 401:
                    logger.error("API密钥无效")
                    return "API key error"

                elif response.status_code == 400:
                    self.stats["bad_request"] += 1
                    logger.error(f"API 400错误 (Bad Request)")
                    try:
                        error_detail = response.json()
                        logger.error(f"错误详情: {error_detail}")
                    except:
                        logger.error(f"错误响应: {response.text[:200]}")
                    return "API Bad Request Error"

                elif response.status_code == 403:
                    logger.error("API访问被拒绝，可能是免费额度已用尽或权限不足")
                    return "API access denied - quota may be exhausted"

                else:
                    logger.error(f"API调用失败，状态码: {response.status_code}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 3
                        time.sleep(wait_time)
                        continue
                    else:
                        return f"API Error: {response.status_code}"

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时，第 {attempt + 1} 次重试...")
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 5)
                    continue
                else:
                    return "Request timeout"

            except requests.exceptions.RequestException as e:
                logger.error(f"请求异常: {e}")
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 3)
                    continue
                else:
                    return f"Request exception: {str(e)}"

        return "API调用失败，达到最大重试次数"

    def analyze_eutectic_pair(self, mol1: str, mol2: str, use_cache: bool = True) -> Dict:
        """
        分析一对分子是否能形成共晶

        Args:
            mol1: 分子1名称
            mol2: 分子2名称
            use_cache: 是否使用缓存

        Returns:
            Dict: 分析结果
        """
        cache_key = self.get_cache_key(mol1, mol2)

        # 检查缓存
        if use_cache and cache_key in self.cache:
            self.stats["cached"] += 1
            logger.debug(f"使用缓存: {mol1} + {mol2}")
            return self.cache[cache_key]

        # 构建提示词
        prompt = f"""Please analyze the following two molecules: {mol1} and {mol2}, 
        determine whether they can form a eutectic, and describe the key characteristics of such a eutectic in detail. 

        The output must include a clear "Yes/No" conclusion and a structured feature description in English only.

        Please provide your analysis in the following format:

        Conclusion: [Yes/No]

        Likelihood: [High/Medium/Low]

        Key Characteristics:
        - [Characteristic 1]
        - [Characteristic 2]
        - [Characteristic 3]

        Detailed Analysis:
        [Detailed explanation]

        Potential Applications:
        [Applications if applicable]"""

        logger.info(f"分析: {mol1} + {mol2}")

        # 调用API
        analysis_result = self.call_qwen_api(prompt)

        # 解析结果
        result = {
            "molecule1": mol1,
            "molecule2": mol2,
            "raw_analysis": analysis_result,
            "timestamp": datetime.now().isoformat(),
            "conclusion": self.extract_conclusion(analysis_result),
            "likelihood": self.extract_likelihood(analysis_result),
            "key_characteristics": self.extract_key_characteristics(analysis_result),
            "cache_key": cache_key
        }

        # 保存到缓存
        if use_cache:
            self.cache[cache_key] = result

        return result

    def extract_conclusion(self, analysis_text: str) -> str:
        """
        从分析文本中提取结论

        Args:
            analysis_text: 分析文本

        Returns:
            str: 结论 (Yes/No/Unknown)
        """
        if not analysis_text or analysis_text.startswith("API") or analysis_text.startswith("Request"):
            return "Error"

        analysis_text_lower = analysis_text.lower()

        # 提取结论
        if "conclusion: yes" in analysis_text_lower or "conclusion:yes" in analysis_text_lower:
            return "Yes"
        elif "conclusion: no" in analysis_text_lower or "conclusion:no" in analysis_text_lower:
            return "No"
        elif analysis_text_lower.startswith("yes,") or analysis_text_lower.startswith("yes "):
            return "Yes"
        elif analysis_text_lower.startswith("no,") or analysis_text_lower.startswith("no "):
            return "No"
        elif "they can form" in analysis_text_lower:
            return "Yes"
        elif "they cannot form" in analysis_text_lower or "they can not form" in analysis_text_lower:
            return "No"

        return "Unknown"

    def extract_likelihood(self, analysis_text: str) -> str:
        """
        从分析文本中提取可能性

        Args:
            analysis_text: 分析文本

        Returns:
            str: 可能性 (High/Medium/Low/Unknown)
        """
        if not analysis_text or analysis_text.startswith("API") or analysis_text.startswith("Request"):
            return "Error"

        analysis_text_lower = analysis_text.lower()

        # 查找可能性关键词
        if "likelihood: high" in analysis_text_lower or "high likelihood" in analysis_text_lower:
            return "High"
        elif "likelihood: medium" in analysis_text_lower or "medium likelihood" in analysis_text_lower:
            return "Medium"
        elif "likelihood: low" in analysis_text_lower or "low likelihood" in analysis_text_lower:
            return "Low"
        elif "high" in analysis_text_lower and "likelihood" in analysis_text_lower:
            return "High"
        elif "medium" in analysis_text_lower and "likelihood" in analysis_text_lower:
            return "Medium"
        elif "low" in analysis_text_lower and "likelihood" in analysis_text_lower:
            return "Low"

        return "Unknown"

    def extract_key_characteristics(self, analysis_text: str) -> str:
        """
        从分析文本中提取关键特征

        Args:
            analysis_text: 分析文本

        Returns:
            str: 关键特征摘要
        """
        if not analysis_text or analysis_text.startswith("API") or analysis_text.startswith("Request"):
            return ""

        # 尝试提取关键特征部分
        lines = analysis_text.split('\n')
        key_characteristics = []
        in_key_characteristics = False

        for line in lines:
            line_lower = line.lower()

            if "key characteristic" in line_lower:
                in_key_characteristics = True
                continue

            if "detailed analysis" in line_lower and in_key_characteristics:
                break

            if in_key_characteristics and line.strip() and line.strip().startswith("-"):
                key_characteristics.append(line.strip())

        if key_characteristics:
            return "; ".join(key_characteristics[:3])

        return ""

    def parse_input_file(self, input_file: str) -> List[Tuple[str, str, str, str]]:
        """
        解析输入文件

        Args:
            input_file: 输入文件路径

        Returns:
            List[Tuple]: 解析后的数据列表
        """
        data = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 4:
                        mol1 = parts[0].strip()
                        mol2 = parts[1].strip()
                        col3 = parts[2].strip()
                        col4 = parts[3].strip()

                        data.append((mol1, mol2, col3, col4))
                    else:
                        logger.warning(f"第 {line_num} 行格式错误: {line}")

        except Exception as e:
            logger.error(f"解析文件失败: {e}")
            raise

        logger.info(f"成功解析 {len(data)} 行数据")
        return data

    def analyze_file(self, input_file: str, output_file: str,
                     batch_size: int = 3, delay: float = 5.0,  # 减小批次，增加延迟以适应免费额度
                     start_from: int = 0, max_rows: int = None,
                     use_cache: bool = True):
        """
        分析整个文件，支持断点续传

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            batch_size: 每批处理的行数（免费额度建议较小）
            delay: 批次之间的延迟(秒)（免费额度建议较长）
            start_from: 从第几行开始处理(0-based)
            max_rows: 最大处理行数，None表示全部
            use_cache: 是否使用缓存
        """
        # 保存输入文件路径
        self.input_file = input_file

        # 解析输入文件
        data = self.parse_input_file(input_file)

        if not data:
            logger.error("没有数据需要处理")
            return []

        # 计算实际要处理的数据
        if max_rows is not None:
            end_at = min(start_from + max_rows, len(data))
        else:
            end_at = len(data)

        actual_data = data[start_from:end_at]

        logger.info(f"从第 {start_from} 行开始，处理到第 {end_at - 1} 行，共 {len(actual_data)} 行")

        # 准备输出文件
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建临时文件，用于保存中间结果
        temp_output_file = output_file + ".tmp"

        # 先加载已有的结果（如果存在）
        existing_results = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            index = int(row['index'])
                            existing_results[index] = row
                        except (KeyError, ValueError):
                            continue
                logger.info(f"已加载 {len(existing_results)} 条已有结果")
            except Exception as e:
                logger.warning(f"加载已有结果失败: {e}")

        # 如果存在临时文件，从中加载部分结果
        temp_results = {}
        if os.path.exists(temp_output_file):
            try:
                with open(temp_output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            index = int(row['index'])
                            temp_results[index] = row
                        except (KeyError, ValueError):
                            continue
                logger.info(f"从临时文件加载了 {len(temp_results)} 条中间结果")
            except Exception as e:
                logger.warning(f"加载临时文件失败: {e}")

        # 合并已有结果和临时结果，临时结果优先级更高
        all_results = {**existing_results, **temp_results}

        # 分批处理
        batch_count = (len(actual_data) + batch_size - 1) // batch_size

        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(actual_data))
            batch_data = actual_data[start_idx:end_idx]

            logger.info(
                f"处理批次 {batch_idx + 1}/{batch_count}, 行 {start_idx + start_from} 到 {end_idx - 1 + start_from}")

            batch_results = []
            for idx, (mol1, mol2, col3, col4) in enumerate(batch_data):
                global_idx = start_from + start_idx + idx

                # 检查是否已经处理过（有成功结果）
                if global_idx in all_results:
                    existing_result = all_results[global_idx]
                    if existing_result.get('conclusion') not in ['Error', 'Unknown', 'Pending']:
                        logger.info(f"行 {global_idx}: 已有成功结果，跳过")
                        continue

                self.stats["total_processed"] += 1

                try:
                    # 分析分子对
                    analysis_result = self.analyze_eutectic_pair(mol1, mol2, use_cache)

                    # 构建结果行
                    result_row = {
                        "original_line": f"{mol1}\t{mol2}\t{col3}\t{col4}",
                        "global_index": global_idx,
                        "molecule1": mol1,
                        "molecule2": mol2,
                        "col3": col3,
                        "col4": col4,
                        "conclusion": analysis_result.get("conclusion", "Unknown"),
                        "likelihood": analysis_result.get("likelihood", "Unknown"),
                        "key_characteristics": analysis_result.get("key_characteristics", ""),
                        "analysis": analysis_result.get("raw_analysis", ""),
                        "timestamp": analysis_result.get("timestamp", ""),
                        "cache_key": analysis_result.get("cache_key", "")
                    }

                    batch_results.append(result_row)
                    self.stats["successful"] += 1

                    logger.info(
                        f"行 {global_idx}: {mol1} + {mol2} -> 结论: {result_row['conclusion']} (可能性: {result_row['likelihood']})")

                    # 单个请求间的延迟
                    time.sleep(2.0)

                except Exception as e:
                    logger.error(f"处理行 {global_idx} 时出错: {e}")
                    self.stats["failed"] += 1

                    error_row = {
                        "original_line": f"{mol1}\t{mol2}\t{col3}\t{col4}",
                        "global_index": global_idx,
                        "molecule1": mol1,
                        "molecule2": mol2,
                        "col3": col3,
                        "col4": col4,
                        "conclusion": "Error",
                        "likelihood": "Error",
                        "key_characteristics": "",
                        "analysis": f"Processing error: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                        "cache_key": ""
                    }
                    batch_results.append(error_row)

                    time.sleep(5.0)

            # 更新结果
            for result in batch_results:
                idx = result.get('global_index', 0)
                all_results[idx] = result

            # 每批处理后保存到临时文件
            self._save_temp_results(all_results, temp_output_file, start_from, len(data))

            # 批次间的延迟
            if batch_idx < batch_count - 1:
                logger.info(f"批次间延迟 {delay} 秒...")
                time.sleep(delay)

        # 最终保存结果到正式文件
        self._save_final_results(all_results, output_file, start_from, len(data))

        # 删除临时文件
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
                logger.info("临时文件已删除")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")

        # 保存缓存
        self.save_cache()

        # 打印统计信息
        self.print_statistics()

        return list(all_results.values())

    def _save_temp_results(self, results: Dict[int, Dict], temp_file: str, start_from: int, total_rows: int):
        """
        保存临时结果

        Args:
            results: 结果字典
            temp_file: 临时文件路径
            start_from: 开始行号
            total_rows: 总行数
        """
        try:
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "index", "molecule1", "molecule2", "col3", "col4",
                    "conclusion", "likelihood", "key_characteristics",
                    "analysis_timestamp", "cache_key", "analysis"
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                # 只保存从start_from开始的结果
                for idx in sorted(results.keys()):
                    if idx >= start_from:
                        result = results[idx]
                        analysis_text = str(result.get("analysis", ""))

                        row = {
                            "index": idx,
                            "molecule1": result.get("molecule1", ""),
                            "molecule2": result.get("molecule2", ""),
                            "col3": result.get("col3", ""),
                            "col4": result.get("col4", ""),
                            "conclusion": result.get("conclusion", "Unknown"),
                            "likelihood": result.get("likelihood", "Unknown"),
                            "key_characteristics": result.get("key_characteristics", ""),
                            "analysis_timestamp": result.get("timestamp", ""),
                            "cache_key": result.get("cache_key", ""),
                            "analysis": analysis_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        }
                        writer.writerow(row)

            logger.info(f"临时结果已保存到: {temp_file}")

        except Exception as e:
            logger.error(f"保存临时结果失败: {e}")

    def _save_final_results(self, results: Dict[int, Dict], output_file: str, start_from: int, total_rows: int):
        """
        保存最终结果

        Args:
            results: 结果字典
            output_file: 输出文件路径
            start_from: 开始行号
            total_rows: 总行数
        """
        try:
            data = []
            if self.input_file:
                try:
                    with open(self.input_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split('\t')
                                if len(parts) >= 4:
                                    data.append(parts[:4])
                except Exception as e:
                    logger.warning(f"重新读取输入文件失败: {e}")

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "index", "molecule1", "molecule2", "col3", "col4",
                    "conclusion", "likelihood", "key_characteristics",
                    "analysis_timestamp", "cache_key", "analysis"
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for idx in range(total_rows):
                    if idx in results:
                        # 使用已有结果
                        result = results[idx]
                    elif data and idx < len(data):
                        # 从输入文件获取原始数据
                        mol1, mol2, col3, col4 = data[idx]
                        result = {
                            "molecule1": mol1,
                            "molecule2": mol2,
                            "col3": col3,
                            "col4": col4,
                            "conclusion": "Pending" if idx >= start_from else "Not Processed",
                            "likelihood": "Pending" if idx >= start_from else "Not Processed",
                            "key_characteristics": "",
                            "analysis": "Not processed yet" if idx >= start_from else "Out of processing range",
                            "timestamp": "",
                            "cache_key": ""
                        }
                    else:
                        result = {
                            "molecule1": "",
                            "molecule2": "",
                            "col3": "",
                            "col4": "",
                            "conclusion": "Error",
                            "likelihood": "Error",
                            "key_characteristics": "",
                            "analysis": "Data not found",
                            "timestamp": "",
                            "cache_key": ""
                        }

                    analysis_text = str(result.get("analysis", ""))

                    row = {
                        "index": idx,
                        "molecule1": result.get("molecule1", ""),
                        "molecule2": result.get("molecule2", ""),
                        "col3": result.get("col3", ""),
                        "col4": result.get("col4", ""),
                        "conclusion": result.get("conclusion", "Unknown"),
                        "likelihood": result.get("likelihood", "Unknown"),
                        "key_characteristics": result.get("key_characteristics", ""),
                        "analysis_timestamp": result.get("timestamp", ""),
                        "cache_key": result.get("cache_key", ""),
                        "analysis": analysis_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    }
                    writer.writerow(row)

            logger.info(f"最终结果已保存到: {output_file} (共 {total_rows} 行)")

        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")

    def print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("分析统计信息:")
        logger.info(f"总共处理: {self.stats['total_processed']}")
        logger.info(f"成功: {self.stats['successful']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info(f"使用缓存: {self.stats['cached']}")
        logger.info(f"速率限制次数: {self.stats['rate_limited']}")
        logger.info(f"400错误次数: {self.stats['bad_request']}")

        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"成功率: {success_rate:.2f}%")

        logger.info("=" * 60)


def main():
    print("=" * 60)
    print("千问Qwen3-max API共晶体系批量分析工具")
    print("(使用免费额度，适合长时间运行)")
    print("=" * 60)

    # 配置参数 - 使用Qwen3-max的API密钥
    API_KEY = "sk-7f00496c7c9b432b9ab2731d776ad4f8"  # 替换为你的Qwen3-max API密钥

    # 从环境变量获取API密钥
    if os.getenv("QWEN_API_KEY"):
        API_KEY = os.getenv("QWEN_API_KEY")

    # 检查API密钥
    if not API_KEY or API_KEY == "your_qwen_api_key_here":
        print("错误: 请设置您的千问API密钥")
        print("方法1: 编辑脚本中的API_KEY变量")
        print("方法2: 设置环境变量 QWEN_API_KEY")
        print("示例:")
        print("  Windows (CMD): set QWEN_API_KEY=your_api_key")
        print("  Windows (PowerShell): $env:QWEN_API_KEY='your_api_key'")
        print("  Linux/Mac: export QWEN_API_KEY='your_api_key'")
        sys.exit(1)

    # 文件路径配置
    INPUT_FILE = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/CC_Table/ECC_Table_converted.tab"
    OUTPUT_FILE = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/AI/ECC_result.csv"

    # 从命令行参数获取文件路径
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]

    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件不存在: {INPUT_FILE}")
        print("请先运行PubChem转换脚本生成 CC_Table_converted - repair.tab 文件")
        print("或者指定正确的文件路径")
        sys.exit(1)

    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"API密钥: {API_KEY[:8]}...")
    print(f"使用模型: qwen3-max (免费额度)")
    print("-" * 60)

    # 显示文件信息
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"输入文件总行数: {len(lines)}")

            # 显示指定范围的行
            print("文件630-640行示例:")
            for i in range(630, min(640, len(lines))):
                print(f"  第{i + 1}行: {lines[i].strip()}")
    except Exception as e:
        print(f"读取文件信息时出错: {e}")

    print("-" * 60)

    # 用户配置 - 从指定行开始
    print("配置选项:")
    print("1. 从第631行开始重新处理 (推荐)")
    print("2. 从其他行开始")
    print("3. 从头开始重新处理所有行")

    choice = input("请选择 (1/2/3, 默认1): ").strip()

    if choice == "2":
        try:
            START_FROM = int(input("请输入开始行号 (0-based, 例如631): "))
        except ValueError:
            print("输入格式错误，使用默认值631")
            START_FROM = 631
    elif choice == "3":
        START_FROM = 0
        print("从头开始处理所有行")
    else:
        START_FROM = 631
        print(f"从第 {START_FROM} 行开始重新处理")

    # 检查是否已有输出文件
    if os.path.exists(OUTPUT_FILE):
        print(f"\n注意: 输出文件 {OUTPUT_FILE} 已存在")
        print("脚本将加载已有结果，并只处理未完成或失败的行")
        print("如果要从头开始，请删除或重命名现有输出文件")

    # 配置参数
    BATCH_SIZE = 3
    DELAY = 10.0
    MAX_ROWS = None

    print("-" * 60)
    print("配置确认:")
    print(f"- 起始行: {START_FROM}")
    print(f"- 最大行数: {'无限制' if MAX_ROWS is None else MAX_ROWS}")
    print(f"- 批次大小: {BATCH_SIZE} (免费额度建议小批次)")
    print(f"- 批次延迟: {DELAY}秒 (免费额度需要较长延迟)")
    print(f"- 请求间隔: 2秒 (单个请求间延迟)")

    confirm = input("\n确认开始分析? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消操作")
        sys.exit(0)

    # 创建分析器 - 使用qwen3-max模型
    analyzer = QwenEutecticAnalyzer(api_key=API_KEY, model="qwen3-max")

    # 执行分析
    try:
        print("\n开始分析...")
        start_time = datetime.now()

        results = analyzer.analyze_file(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            batch_size=BATCH_SIZE,
            delay=DELAY,
            start_from=START_FROM,
            max_rows=MAX_ROWS,
            use_cache=True
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("分析完成!")
        print(f"总耗时: {duration}")
        print(f"结果已保存到: {OUTPUT_FILE}")
        print(f"临时文件已删除")
        print(f"缓存文件: {analyzer.cache_file}")
        print("=" * 60)

        # 显示摘要
        if results:
            # 只统计新处理的结果
            new_results = [r for r in results if isinstance(r, dict) and r.get('conclusion')]

            if new_results:
                conclusions = [r.get("conclusion", "") for r in new_results]
                yes_count = conclusions.count("Yes")
                no_count = conclusions.count("No")
                unknown_count = conclusions.count("Unknown")
                error_count = conclusions.count("Error")
                pending_count = conclusions.count("Pending")

                print(f"\n本次处理结果摘要 (从第 {START_FROM} 行开始):")
                print(f"新处理行数: {len(new_results)}")
                print(f"能形成共晶 (Yes): {yes_count}")
                print(f"不能形成共晶 (No): {no_count}")
                print(f"不确定 (Unknown): {unknown_count}")
                print(f"错误 (Error): {error_count}")
                if pending_count > 0:
                    print(f"待处理 (Pending): {pending_count}")

                # 显示最近处理的结果
                print("\n最近处理的5个结果:")
                for i, result in enumerate(new_results[-5:]):
                    print(f"{i + 1}. {result.get('molecule1', '')} + {result.get('molecule2', '')}")
                    print(
                        f"   结论: {result.get('conclusion', 'Unknown')} (可能性: {result.get('likelihood', 'Unknown')})")
                    analysis_preview = str(result.get('analysis', ''))[:100] + "..." if len(
                        str(result.get('analysis', ''))) > 100 else str(result.get('analysis', ''))
                    print(f"   分析预览: {analysis_preview}")
                    print()

        # 生成摘要报告
        try:
            summary_file = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/resualt/analysis_summary_ECC.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("千问Qwen3-max API共晶分析摘要报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"使用模型: qwen3-max (免费额度)\n")
                f.write(f"输入文件: {INPUT_FILE}\n")
                f.write(f"输出文件: {OUTPUT_FILE}\n")
                f.write(f"开始行: {START_FROM}\n")
                f.write(f"总处理行数: {len(results) if results else 0}\n")
                f.write(
                    f"成功率: {analyzer.stats['successful']}/{analyzer.stats['total_processed']} ({analyzer.stats['successful'] / analyzer.stats['total_processed'] * 100:.1f}%)\n")
                f.write("=" * 50 + "\n")
            print(f"摘要报告已保存到: {summary_file}")
        except Exception as e:
            print(f"生成摘要报告时出错: {e}")

    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()