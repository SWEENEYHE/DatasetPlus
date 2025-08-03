"""Core module for DatasetPlus functionality."""

import os
import inspect
import hashlib
import json
import logging
import re
import random
import glob
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from openai import OpenAI
import pandas as pd
import json5
from json_repair import repair_json

class DatasetPlusMeta(type):
    def __getattr__(cls, name):
        dataset_attr = getattr(Dataset, name, None)
        if dataset_attr is not None:
            # 处理类方法
            if inspect.ismethod(dataset_attr) and dataset_attr.__self__ is Dataset:
                original_func = dataset_attr.__func__

                # 直接返回静态方法，内部显式传递Dataset类
                def static_wrapper(*args, **kwargs):
                    result = original_func(Dataset, *args, **kwargs)
                    return DatasetPlus(result) if isinstance(result, (Dataset, DatasetDict)) else result

                return staticmethod(static_wrapper)
            # 处理静态方法或普通函数
            elif inspect.isfunction(dataset_attr):
                def func_wrapper(*args, **kwargs):
                    result = dataset_attr(*args, **kwargs)
                    return DatasetPlus(result) if isinstance(result, (Dataset, DatasetDict)) else result

                return staticmethod(func_wrapper)
            else:
                return dataset_attr
        raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")


class DatasetPlus(metaclass=DatasetPlusMeta):
    @staticmethod
    def load_dataset(file_name, output_file="DatasetPlus_temp/temp_map.jsonl"):
        """加载dataset"""
        ds = None
        filename, postfix = os.path.splitext(file_name)
        if postfix == ".jsonl" or postfix == ".json":
            ds = load_dataset("json", data_files=file_name)
        elif postfix == ".xlsx":
            df = pd.read_excel(file_name)
            ds = Dataset.from_pandas(df)
        elif postfix == ".csv":
            ds = load_dataset("csv", data_files=file_name)
        return DatasetPlus(ds, output_file=output_file)

    @staticmethod
    def load_dataset_plus(input_path, output_file="DatasetPlus_temp/temp_map.jsonl"):
        """
        加载数据集。
        如果 input_path 是文件，则根据后缀加载。
        如果 input_path 是目录，则加载目录下所有支持类型的文件并合并。
        """
        loaded_datasets_list = []  # 用于存储从各个文件加载的 Dataset 对象

        # 内部辅助函数，用于处理单个文件加载的逻辑
        def _load_and_append_single_file(file_path, target_list):
            _, postfix = os.path.splitext(file_path)
            postfix = postfix.lower()  # 统一转为小写以兼容不同大小写后缀

            dataset_dict_item = None  # Hugging Face load_dataset 通常返回 DatasetDict
            try:
                if postfix in [".jsonl", ".json"]:
                    logging.info(f"Loading JSON file: {file_path}")
                    dataset_dict_item = load_dataset("json", data_files=file_path)
                elif postfix == ".xlsx":
                    logging.info(f"Loading Excel file: {file_path}")
                    dataset_dict_item = load_dataset("excel", data_files=file_path)
                elif postfix == ".csv":
                    logging.info(f"Loading CSV file: {file_path}")
                    dataset_dict_item = load_dataset("csv", data_files=file_path)
                else:
                    logging.warning(f"Skipping unsupported file type: {file_path}")
                    return

                # 从 DatasetDict 中提取 Dataset
                if dataset_dict_item:
                    if 'train' in dataset_dict_item:
                        target_list.append(dataset_dict_item['train'])
                    elif dataset_dict_item:  # 如果没有 'train' 但 DatasetDict 不为空
                        first_split_name = list(dataset_dict_item.keys())[0]
                        target_list.append(dataset_dict_item[first_split_name])
                        logging.info(f"Used first available split '{first_split_name}' for {file_path}")
                    else:
                        logging.warning(f"load_dataset returned an empty DatasetDict for {file_path}")
                else:
                    logging.warning(f"Failed to load dataset from {file_path}, result was None.")

            except Exception as e:
                logging.error(f"Error loading file {file_path}: {e}")

        # 检查 input_path 是目录还是文件
        if os.path.isdir(input_path):
            logging.info(f"Input path is a directory: {input_path}. Scanning for supported files...")
            for item_name in os.listdir(input_path):
                full_item_path = os.path.join(input_path, item_name)
                if os.path.isfile(full_item_path):
                    _load_and_append_single_file(full_item_path, loaded_datasets_list)
                else:
                    logging.debug(f"Skipping non-file item in directory: {full_item_path}")

        elif os.path.isfile(input_path):
            logging.info(f"Input path is a single file: {input_path}")
            _load_and_append_single_file(input_path, loaded_datasets_list)

        else:
            # 如果不是本地文件或目录，可以尝试作为Hugging Face Hub的dataset ID加载
            # 注意：这种加载方式通常不需要指定文件类型，load_dataset 会自动处理
            logging.info(
                f"Input path '{input_path}' is not a local file or directory. Attempting to load as a Hugging Face Hub dataset ID or remote URL.")
            try:
                # 直接用 input_path 作为标识符尝试加载
                # Hugging Face load_dataset 返回的可能是 Dataset 或 DatasetDict
                direct_load_result = load_dataset(input_path)  # 不指定 data_files 或 type

                # DatasetPlus的构造函数会处理Dataset或DatasetDict
                # 这里我们直接把结果传给DatasetPlus，因为它已经有处理逻辑
                # 但为了保持一致性，如果返回DatasetDict，我们先尝试提取
                if isinstance(direct_load_result, Dataset):
                    loaded_datasets_list.append(direct_load_result)
                elif isinstance(direct_load_result, DatasetDict):
                    # DatasetPlus 的构造函数会处理这个，但为了明确，可以先提取
                    if 'train' in direct_load_result:
                        loaded_datasets_list.append(direct_load_result['train'])
                    elif direct_load_result:
                        first_split_name = list(direct_load_result.keys())[0]
                        loaded_datasets_list.append(direct_load_result[first_split_name])
                        logging.info(f"Used first available split '{first_split_name}' for Hub dataset {input_path}")
                    else:
                        logging.warning(f"Direct load of '{input_path}' resulted in an empty DatasetDict.")
                else:
                    logging.warning(
                        f"Direct load of '{input_path}' returned an unexpected type: {type(direct_load_result)}")

            except Exception as e:
                logging.error(f"Failed to load '{input_path}' as a Hub dataset ID or remote URL: {e}")
                # 如果加载失败，loaded_datasets_list 将为空或保持之前的内容

        # 合并所有加载的数据集
        final_dataset = None
        if not loaded_datasets_list:
            logging.warning("No datasets were successfully loaded.")
            # final_dataset 保持 None，DatasetPlus 构造函数会处理
        elif len(loaded_datasets_list) == 1:
            final_dataset = loaded_datasets_list[0]
            logging.info("One dataset loaded.")
        else:
            try:
                logging.info(f"Concatenating {len(loaded_datasets_list)} loaded datasets.")
                final_dataset = concatenate_datasets(loaded_datasets_list)
            except Exception as e:
                logging.error(f"Error concatenating datasets: {e}")
                # final_dataset 保持 None

        return DatasetPlus(final_dataset, output_file=output_file)

    def __init__(self, ds=None, output_file="DatasetPlus_temp/DatasetPlus_map.jsonl"):
        if ds is not None:
            assert isinstance(ds, (Dataset, DatasetDict)), f"Expected Dataset/DatasetDict, got {type(ds)}"
            self.ds = ds
        else:
            self.ds = None  # 或者设置一个默认的空Dataset
        self.output_file = output_file

    def map(self, fn, num_proc=1, max_inner_num=1000, cache=True):
        # self.max_inner_num = max_inner_num # 或者作为参数传递，或者已经是成员变量
        if self.ds is None:
            logging.warning("Dataset is None. map() operation will not be performed. Please load a dataset first.")
            return None
        original_dir, file_name_full = os.path.split(self.output_file)
        file_name_base, post_fix = os.path.splitext(file_name_full)

        logging.info(f"Original directory: {original_dir}, Original file base: {file_name_base}")
        if cache:
            # 1. 计算函数内容的哈希值 (使用hashlib)
            code_bytes = fn.__code__.co_code
            m_code = hashlib.sha256()  # 或者 hashlib.md5()
            m_code.update(code_bytes)
            code_hash_hex = m_code.hexdigest()

            # 2. 计算数据集内容的哈希值 (使用hashlib)
            # 注意：self.ds.to_list() 对于非常大的数据集可能会消耗大量内存
            # 并且 str(x) 对于复杂对象（如字典）的字符串表示可能不是唯一的（例如键的顺序）
            # 使用 json.dumps(x, sort_keys=True) 会更稳定
            m_ds = hashlib.sha256()  # 或者 hashlib.md5()
            try:
                # 尝试逐项处理，更节省内存，也更推荐
                # 如果数据集项是字典，json.dumps能确保顺序一致性
                # 确保你的数据集项可以被json序列化
                for item in self.ds:  # 假设self.ds是可迭代的
                    item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    m_ds.update(item_str.encode('utf-8'))  # 哈希函数需要bytes
                ds_hash_hex = m_ds.hexdigest()
            except Exception as e:
                logging.warning(
                    f"Could not hash dataset item by item ({e}), falling back to to_list(). This might be memory intensive.")
                # 回退到原有逻辑，但使用更稳定的序列化和hashlib
                ds_content_parts = []
                for x in self.ds.to_list():
                    # 对于字典等，确保字符串表示是稳定的
                    if isinstance(x, dict):
                        ds_content_parts.append(json.dumps(x, sort_keys=True, ensure_ascii=False))
                    else:
                        ds_content_parts.append(str(x))
                full_ds_string = "".join(ds_content_parts)
                m_ds.update(full_ds_string.encode('utf-8'))  # 哈希函数需要bytes
                ds_hash_hex = m_ds.hexdigest()

            # 3. 组合哈希值 (使用字符串拼接，而不是数字相加)
            # 数字相加可能导致冲突，且丢失了各自哈希的信息
            # fn_hash_str = f"{code_hash_hex}_{ds_hash_hex}"
            # 或者只取一部分避免文件名过长，但会增加冲突概率
            combined_hash_input = f"{code_hash_hex}-{ds_hash_hex}".encode('utf-8')
            m_combined = hashlib.sha256()
            m_combined.update(combined_hash_input)
            fn_hash_final = m_combined.hexdigest()[:16]  # 取前16位作为示例，可以调整长度

            # 构建缓存目录路径
            # 原始逻辑: dir += f"_{fn_hash}" 会把哈希附加到原始目录名上
            # 例如 /path/to/data -> /path/to/data_hashvalue
            # 一个更常见的模式是创建一个子目录：/path/to/cache_hashvalue

            # 保持你的原始逻辑，将哈希附加到目录名上
            # cached_dir = f"{original_dir}_{fn_hash_final}"
            # 或者创建一个以哈希命名的子目录 (通常更清晰)
            cached_dir = os.path.join(original_dir, f"cache_{fn_hash_final}")

            # 更新 output_file 路径前缀，不包含扩展名
            # self.output_file 变量将被用于构造每个分块的文件名
            # 原始代码是 self.output_file = f"{dir}/{file_name}" (file_name不带扩展名)
            current_output_file_prefix = os.path.join(cached_dir, file_name_base)

            if not os.path.exists(cached_dir):
                os.makedirs(cached_dir)
                logging.info(f"Created cache directory: {cached_dir}")

        num_samples = len(self.ds)
        # 最大内存存储量，必须小于等于ds，否则无法运行后续循环逻辑
        max_inner_num = min(num_samples, max_inner_num)

        if num_proc > max_inner_num:  # 你原来的self.max_inner_num可能是笔误
            # 应该是 num_proc 和 max_inner_num 的关系，而不是 self.max_inner_num
            # 这里的逻辑可能需要根据你的意图调整，例如比较 num_proc 和 CPU 核心数
            num_proc = min(num_proc, max(1, max_inner_num // 10))  # 确保num_proc至少为1
            logging.warning(
                f"num_proc was adjusted. Original num_proc might be too large relative to max_inner_num. New num_proc: {num_proc}")

        updated_ds_list = []  # 重命名以避免与 Hugging Face 的 updated_ds 混淆

        for start in range(0, num_samples, max_inner_num):
            # if start >= num_samples: # 这个检查其实不需要，range的特性会处理
            #     break
            end = min(start + max_inner_num, num_samples)
            # 构造分块文件的完整路径
            if cache:
                file_path = f"{current_output_file_prefix}#{start}_{end}{post_fix}"

            if cache and os.path.exists(file_path):
                logging.info(f"Loading cached chunk: {file_path}")
                # 确保你的load_dataset能正确处理单个json文件
                # 并且返回的是 Dataset 对象
                # 如果 load_dataset 返回 DatasetDict，需要取特定split，如['train']
                loaded_chunk = load_dataset("json", data_files=file_path, split="train")  # 假设总是 'train'
                updated_ds_list.append(loaded_chunk)
            else:
                try:
                    logging.info(f"Processing batch from {start} to {end}")
                    batch = self.ds.select(range(start, end))
                    
                    # 创建包装函数来处理fn返回数组的情况，并自动过滤None
                    def wrapper_fn(example):
                        result = fn(example)
                        # 如果fn返回None，标记为需要删除
                        if result is None:
                            return {"_delete_state_": True, "_expand_state_":False, "_expand_list_":[]}
                        # 如果fn返回的是列表，过滤掉None元素
                        elif isinstance(result, list):
                            filtered_list = [x for x in result if x is not None]
                            return {"_delete_state_": False, "_expand_state_":True, "_expand_list_":filtered_list}
                        else:
                            result.update({"_delete_state_": False, "_expand_state_":False, "_expand_list_":[]})
                            return result
                    
                    # 应用map转换
                    # 注意：Hugging Face datasets .map() 默认就是多进程的，num_proc 控制其内部进程数
                    processed_batch = batch.map(wrapper_fn, num_proc=num_proc)  # Hugging Face map
                    
                    # 过滤掉标记为删除的行，并处理列表展开
                    final_data = []
                    for item in processed_batch:
                        # 跳过标记为删除的行
                        if item["_delete_state_"]:
                            continue
                        # 处理需要展开的列表
                        elif isinstance(item["_expand_list_"], list) and item["_expand_state_"]:
                            for expanded_item in item["_expand_list_"]:
                                # 清理状态字段
                                clean_item = {k: v for k, v in expanded_item.items() 
                                            if k not in ["_delete_state_", "_expand_state_", "_expand_list_"]}
                                final_data.append(clean_item)
                        else:
                            # 清理状态字段
                            clean_item = {k: v for k, v in item.items() 
                                        if k not in ["_delete_state_", "_expand_state_", "_expand_list_"]}
                            final_data.append(clean_item)
                    
                    # 创建新的Dataset
                    if final_data:
                        processed_batch = Dataset.from_list(final_data)
                        # 保存到JSON Lines文件
                        if cache:
                            processed_batch.to_json(file_path, lines=True, force_ascii=False)  # mode='a'在这里不需要，因为我们是写新文件
                            logging.info(f"Processed batch saved to {file_path}")
                        updated_ds_list.append(processed_batch)
                    # 如果final_data为空，说明这个批次的所有数据都被过滤掉了，不需要添加到updated_ds_list
                       
                except Exception as e:
                    print("Error processing batch from {start} to {end}: {e}")
                    logging.error(f"Error processing batch from {start} to {end}: {e}")
                    # 可以选择是否继续处理其他批次，或者抛出异常
                    continue
                    # raise

        if not updated_ds_list:
            logging.warning("No data was processed or loaded. Returning original dataset.")
            # return DatasetPlus(self.ds) # 或者根据你的类设计返回
            # 如果 updated_ds_list 为空，concatenate_datasets 可能会出错
            if len(self.ds) == 0:  # 如果原始数据集也为空
                return DatasetPlus(self.ds)  # 或者返回一个空的 DatasetPlus 对象
            else:  # 如果原始数据集不为空，但处理后为空列表，这可能是一个错误或特殊情况
                logging.error(
                    "updated_ds_list is empty after processing non-empty dataset. This might indicate an issue.")
                # 根据情况决定是返回原始数据集还是抛出错误
                # For now, let's assume if it's empty, something went wrong or no chunks were meant to be processed/loaded
                # If original self.ds was non-empty, this state is problematic.
                # If self.ds was empty, updated_ds_list will be empty, and concatenating empty list is fine.
                if len(self.ds) > 0:
                    raise ValueError(
                        "updated_ds_list is empty but original dataset was not. Processing likely failed for all chunks.")

        # 只有当 updated_ds_list 非空时才进行拼接
        if updated_ds_list:
            self.ds = concatenate_datasets(updated_ds_list)
        # else: self.ds 保持不变（可能是空数据集，或者处理失败）

        return DatasetPlus(self.ds)  # 假设DatasetPlus是你定义的类

    def __len__(self):
        """返回数据集的长度"""
        if self.ds is None:
            return 0
        if isinstance(self.ds, DatasetDict):
            # 如果是DatasetDict，返回第一个split的长度
            first_split = list(self.ds.keys())[0]
            return len(self.ds[first_split])
        return len(self.ds)

    def __getattr__(self, name):
        attr = getattr(self.ds, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return DatasetPlus(result) if isinstance(result, (Dataset, DatasetDict)) else result

            return wrapper
        return attr

    def __getitem__(self, key):
        #print(key)
        #print(type(key))
        """实现中括号切片和词典取值操作"""
        if isinstance(key, str):
            #print("str")
            if isinstance(self.ds[key], list):
                #print("list")
                return self.ds[key]
            elif isinstance(self.ds[key], dict):
                #print("dict")
                return DatasetPlus(Dataset.from_dict(self.ds[key]))
            elif isinstance(self.ds[key], Dataset):
                #print("ds")
                return DatasetPlus(self.ds[key])

        elif isinstance(key, slice):
            #print("slice")
            s = key.start if key.start is not None else 0
            e = key.stop if key.stop is not None else len(self.ds)
            step = key.step if key.step is not None else 1
            return DatasetPlus(self.ds.select(range(s, e, step)))
        elif isinstance(key, int):
            return self.ds[key]
        #print("here")
        raise NotImplementedError("Only support str and tuple")

    def __str__(self):
        """返回原始Dataset的字符串表示"""
        return str(self.ds).replace("Dataset", "DatasetPlus")

    def __repr__(self):
        """返回原始Dataset的详细表示"""

        def check(text):
            # 定义正则表达式模式
            pattern = re.compile(
                r"Dataset\(\{\s*"
                r"features:\s*\[(?:'[^\']*'(?:,\s*'[^']*')*)?\],\s*"
                r"num_rows:\s*\d+\s*"
                r"\}\)"
            )
            # 检查文本是否匹配模式
            return bool(pattern.match(text))

        original_repr = repr(self.ds)
        if check(original_repr):
            return original_repr.replace("Dataset", "DatasetPlus")
        else:
            return f"DatasetPlus({original_repr})"

    @staticmethod
    def from_pandas(df):
        return DatasetPlus(Dataset.from_pandas(df))

    @staticmethod
    def iter(iterate_num, fn, num_proc=1, max_inner_num=1000, cache=False):
        """无中生有，迭代、构造数据"""
        dsp = DatasetPlus.from_pandas(pd.DataFrame({"id": list(range(0, iterate_num))}))
        return dsp.map(fn, num_proc=num_proc, max_inner_num=max_inner_num, cache=cache)


class DatasetPlusExcels(DatasetPlus):
    """继承DatasetPlus，专门用于处理多sheet的Excel文件"""
    
    def __init__(self, file_path, output_file="DatasetPlus_temp/DatasetPlusExcels_map.jsonl"):
        """初始化DatasetPlusExcels
        
        Args:
            file_path: Excel文件路径
            output_file: 输出文件路径
        """
        self.file_path = file_path
        self.excel_data = None
        self.sheet_names = []
        super().__init__(output_file=output_file)
        
    @staticmethod
    def load_dataset(file_path, target_sheets=None, output_file="DatasetPlus_temp/DatasetPlusExcels_map.jsonl"):
        """加载Excel文件的指定sheets
        
        Args:
            file_path: Excel文件路径
            target_sheets: 目标sheet列表，None表示加载所有sheet
            output_file: 输出文件路径
            
        Returns:
            DatasetPlusExcels: 实例对象
        """
        instance = DatasetPlusExcels(file_path, output_file)
        
        # 读取所有sheet信息
        all_excel_data = pd.read_excel(file_path, sheet_name=None)
        all_sheet_names = list(all_excel_data.keys())
        
        # 确定要加载的sheets
        if target_sheets is None:
            target_sheets = all_sheet_names
        else:
            # 验证target_sheets是否存在
            invalid_sheets = [sheet for sheet in target_sheets if sheet not in all_sheet_names]
            if invalid_sheets:
                raise ValueError(f"以下sheets不存在于Excel文件中: {invalid_sheets}，可用sheets: {all_sheet_names}")
        
        # 只加载目标sheets
        instance.excel_data = {sheet: all_excel_data[sheet] for sheet in target_sheets}
        instance.sheet_names = target_sheets
        
        # 打印加载的sheets顺序，给用户提示
        print(f"已加载Excel文件: {file_path}")
        print(f"目标sheets顺序: {target_sheets}")
        print(f"注意：map方法中的fn参数将按此顺序对应处理")
        logging.info(f"成功加载Excel文件: {file_path}，目标sheets: {target_sheets}")
        
        return instance
    
    def map(self, fn=None, num_proc=1, max_inner_num=200, cache=True):
        """处理已加载的Excel sheets
        
        Args:
            fn: 处理函数列表，可以是单个函数或函数列表，按照load_dataset时的target_sheets顺序对应
            num_proc: 进程数
            max_inner_num: 最大内存存储量
            cache: 是否使用缓存
            
        Returns:
            dict: 包含处理后的各个sheet数据的字典
        """
        if self.excel_data is None:
            raise ValueError("请先使用load_dataset方法加载Excel文件")
        
        # 使用已加载的sheets
        target_sheets = self.sheet_names
        
        if fn is None:
            fn = [lambda x: x]  # 默认不做任何处理
        
        # 如果只传入一个函数
        if callable(fn):
            fn = [fn]
        
        # 确保函数数量与sheet数量匹配
        if callable(fn) or (len(fn) == 1 and len(target_sheets) > 1):
            fn = fn * len(target_sheets)
        elif len(fn) != len(target_sheets):
            print(f"当前sheets顺序: {target_sheets}")
            print(f"函数数量: {len(fn)}, sheet数量: {len(target_sheets)}")
            raise ValueError(f"函数数量({len(fn)})与sheet数量({len(target_sheets)})不匹配。请确保fn按照sheets顺序对应: {target_sheets}")
        
        # 打印处理顺序提示
        print(f"开始处理sheets，顺序为: {target_sheets}")
        print(f"函数对应关系:")
        for i, sheet_name in enumerate(target_sheets):
            func_name = getattr(fn[i], '__name__', f'function_{i}')
            print(f"  {i+1}. {sheet_name} -> {func_name}")
        
        processed_sheets = {}
        
        # 获取Excel文件名（不含路径和扩展名）
        excel_name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        for i, sheet_name in enumerate(target_sheets):
            # 获取当前sheet的数据
            sheet_df = self.excel_data[sheet_name]
            
            # 为当前sheet创建专用的输出文件路径，包含excel名和sheet名
            original_dir, file_name_full = os.path.split(self.output_file)
            sheet_output_file = os.path.join(original_dir, f"temp_{excel_name}_{sheet_name}")
            
            # 转换为DatasetPlus对象，使用专用的输出文件路径
            sheet_dataset = DatasetPlus(Dataset.from_pandas(sheet_df), output_file=sheet_output_file)
            
            # 应用对应的处理函数
            current_fn = fn[i]
            if current_fn is not None:
                try:
                    print(f"正在处理sheet: {sheet_name} (第{i+1}个)")
                    processed_dataset = sheet_dataset.map(current_fn, num_proc=num_proc, max_inner_num=max_inner_num, cache=cache)
                    processed_sheets[sheet_name] = processed_dataset
                    logging.info(f"成功处理sheet: {sheet_name}")
                except Exception as e:
                    logging.error(f"处理sheet '{sheet_name}' 时出错: {e}")
                    processed_sheets[sheet_name] = sheet_dataset  # 保存原始数据
            else:
                processed_sheets[sheet_name] = sheet_dataset
        
        return processed_sheets
    
    def to_excel(self, processed_sheets, output_path=None):
        """保存处理后的数据到Excel文件，保持原sheet格式
        
        Args:
            processed_sheets: 处理后的sheet数据字典
            output_path: 输出Excel文件路径，None则使用原文件名加后缀
            
        Returns:
            str: 输出文件路径
        """
        if output_path is None:
            base_name, ext = os.path.splitext(self.file_path)
            output_path = f"{base_name}_processed{ext}"
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, dataset_plus in processed_sheets.items():
                # 将DatasetPlus转换回pandas DataFrame
                if hasattr(dataset_plus, 'ds'):
                    df = dataset_plus.ds.to_pandas()
                else:
                    df = dataset_plus.to_pandas()
                
                # 写入对应的sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logging.info(f"已保存sheet '{sheet_name}' 到 {output_path}")
        
        logging.info(f"所有处理后的数据已保存到: {output_path}")
        return output_path
    
    def get_sheet_names(self):
        """获取所有sheet名称"""
        return self.sheet_names
    
    def get_sheet_data(self, sheet_name):
        """获取指定sheet的原始数据"""
        if self.excel_data and sheet_name in self.excel_data:
            return self.excel_data[sheet_name]
        else:
            raise ValueError(f"Sheet '{sheet_name}' 不存在或Excel文件未加载")


class MyLLMTool:
    def __init__(self, model_name="", base_url="",
                 api_key=""):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        # 列出所有可用模型
        #print(self.client.models.list())
        self.model_name = model_name

    def getResult(self, query, sys_prompt=None, temperature=0.7, top_p=1, max_tokens=2048,  model_name=None):
        if model_name is None:
            model_name = self.model_name
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ] if sys_prompt else [{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False
        )
        #print(completion)
        result = completion.choices[0].message.content.strip()
        #print(completion)
        return result



class DataTool:
    @staticmethod
    def parse_json_safe(text_str: str):
        """
        从包含文本的字符串中提取并解析一个或多个JSON对象/数组。

        Args:
            text_str (str): 可能包含一个或多个嵌入式JSON的输入字符串。

        Returns:
            list: 一个包含解析成功的Python对象（字典或列表）的列表。
                  如果找不到有效的JSON或解析失败，则返回空列表。
        """
        results = []
        i = 0
        n = len(text_str)

        while i < n:
            char = text_str[i]

            start_char = None
            end_char = None

            # 检查是否是JSON对象或数组的开始
            if char == '{':
                start_char = '{'
                end_char = '}'
            elif char == '[':
                start_char = '['
                end_char = ']'

            if start_char:
                balance = 1  # 用于跟踪嵌套的括号/方括号
                start_index = i

                # 向前扫描以找到匹配的结束字符
                j = i + 1
                found_end_char = False
                while j < n:
                    current_scan_char = text_str[j]

                    # 如果在字符串内部遇到引号，需要特殊处理，但对于简单的括号匹配，这可以省略
                    # 实际的JSON解析器会处理字符串内的括号

                    if current_scan_char == start_char:  # 处理嵌套，例如 {"key": {"nested_key": "value"}}
                        balance += 1
                    elif current_scan_char == end_char:
                        balance -= 1

                    if balance == 0:  # 找到了匹配的结束字符
                        potential_json_str = text_str[start_index: j + 1]
                        try:
                            parsed_json = json.loads(potential_json_str)
                            results.append(parsed_json)
                            i = j  # 将主扫描指针移到已解析JSON的末尾
                            found_end_char = True
                        except json.JSONDecodeError:
                            # 这不是一个有效的JSON字符串，或者只是一个更大结构的一部分。
                            # 我们将忽略它，外部循环的 i 会递增，从下一个字符开始扫描。
                            pass
                        break  # 无论成功与否，都跳出内部扫描循环 (j loop)
                    j += 1

                # 如果内部循环结束但没有找到匹配的结束符 (found_end_char is False)
                # 外部循环的 i 会自动递增，跳过当前的 start_char

            i += 1  # 移动主扫描指针到下一个字符
        if len(results) == 1:
            return results[0]
        else:
            return results

    @classmethod
    def get_prompt(cls, file_path):
        # 读取txt文件，并将结果拼接成文本
        with open(file_path, 'r') as f:
            lines = f.readlines()
            prompt = ''.join(lines)
        return prompt

    @classmethod
    def check(cls, row):
        """
        检查数据结构是否符合要求
        :param row:
        :return:
        """
        if "messages" not in row:
            return False
        arr = row['messages']
        for a in arr:
            if 'content' not in a or 'role' not in a or a['role'] not in ['system', 'user', 'assistant']:
                return False
        return True

    @classmethod
    def check_with_system(cls, row):
        flag = cls.check(row)
        if flag is False:
            return False
        if row['messages'][0]['role'] == "system":
            return True
        return False

    @classmethod
    def parse_messages(cls, str_row):
        '''
        模型输出字符串类型样本
        :param str_row:
        :return:
        '''
        match = re.search(r"\[{['\"]role['\"].*?},?]", str_row, re.DOTALL)
        if match:
            o = json5.loads(match.group(0).replace("\'", "\"").replace(" ", "").replace("\n", ""))
            return o
        return None

    @classmethod
    def parse_json(cls, str, json_tag=False):
        if json_tag:
            match = re.search(r"```json\s*(.*?)\s*```", str, re.DOTALL)
            if match:
                str = match.group(1).strip()
        try:
            row = json5.loads(str)
            return row
        except Exception as e:
            print(e)
            try:
                good_json = repair_json(str, ensure_ascii=False)
                row = json5.loads(good_json)
                return row
            except Exception as e:
                print(f"JSON解析错误: {e}")
                # 打印错误发生位置附近的内容，帮助调试
                error_context_start = max(0, e.pos - 20)
                error_context_end = min(len(str), e.pos + 20)
                print(
                    f"错误位置 '{e.msg}' 附近: ...{str[error_context_start:e.pos]}<--ERROR-->{str[e.pos:error_context_end]}...")
                print(e)
                return None

    @classmethod
    def sample_from_file(cls, file_path, num=-1):
        """
        读取txt文件，每行一个数据，随机取其中一个
        :param file_path:
        :param num:
        :return:
        """
        rows = []
        with open(file_path, 'r') as f:
            for line in f:
                rows.append(line.strip())
        if num == -1:
            return rows
        return random.sample(rows, num)

    @classmethod
    def sample_from(cls, path, num=-1, granularity="auto", exclude=["prompt.*"]):
        """从文件夹中读取每个数据，粒度可以控制,默认为auto，即如果是文件则按行分割，如果是目录则每个文件单独分割"""
        files = []
        if os.path.isdir(path):
            files = glob.glob(path)
            # 从files 中删除  exclude中的所有内容,注意exclude中可以包含正则匹配
            for exclude_file in exclude:
                files = [file for file in files if not re.match(exclude_file, file)]
        else:
            files = [path]

        file_contents = []
        for file in files:
            if granularity == "file":
                file_contents.append(cls.get_prompt(file))
            elif granularity == "line":
                file_contents.extend(cls.sample_from_file(file))
        if num == -1:
            return file_contents
        return random.sample(file_contents, num)

    @classmethod
    def jsonl2json(cls, source_path, des_path):
        with open(source_path, 'r', encoding='utf-8') as infile:
            data = [json.loads(line) for line in infile if line.strip()]

        with open(des_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)


