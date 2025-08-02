# DatasetPlus

一个增强的Hugging Face datasets包装器，专为基于大模型数据的处理而设计，提供智能缓存、数据扩充和筛选功能。
使用大模型扩充数据的不稳定性因素太多：网络不佳、大模型输出格式不对导致bug、额度突然不够了，都可能导致失败

## 🚀 核心特性

### 1. 🔄 完全兼容 - 无缝替换datasets
- **100%兼容**: 支持所有datasets的方法和属性
- **相互转换，即插即用**: 只需`dsp = DatasetPlus(ds)` 转回来：`ds = dsp.ds`
- **静态方法支持**: 所有Dataset的静态方法、类方法都可直接调用

### 2. 🧠 智能缓存系统 - 大模型调用零丢失
- **自动函数级缓存**: 根据函数内容自动生成缓存，，即使网络不稳定、额度不足等导致中断，之前的结果也不会丢失
- **Jupyter友好**: 即使忘记赋值变量，也能从缓存中恢复结果
- **断点续传**: 支持中断后继续处理，已处理的数据自动读取缓存

### 3. 📈 数据扩充 - 一行变多行
- **数组自动展开**: map函数返回数组时自动展开为多行数据
- **LLM结果解析**: 配合MyLLMTool轻松实现数据扩充

### 4. 🔍 智能筛选 - 返回None自动删除
- **条件筛选**: map函数返回None时自动删除该行
- **LLM智能筛选**: 使用大模型进行复杂条件筛选

### 5. 🎨 无中生有 - 直接从大模型生成数据
- **iter方法**: 支持从零开始生成数据集
- **灵活生成**: 可生成任意格式和内容的数据
- **批量生成**: 支持大规模数据生成，自动缓存和并行处理

## 📦 安装

### 从PyPI安装

```bash
pip install datasetplus
```

### 从源码安装

```bash
git clone https://github.com/yourusername/datasetplus.git
cd datasetplus
pip install -e .
```

### 依赖安装

```bash
# 基础依赖
pip install datasets pandas numpy

# Excel支持
pip install openpyxl

# LLM支持
pip install openai
```

## 🎯 快速开始

### 基础用法
```python
from datasetplus import DatasetPlus, MyLLMTool

# 加载数据集
dsp = DatasetPlus.load_dataset("data.jsonl")

# 完全兼容datasets - 即插即用
ds = dsp.ds  # 现在ds拥有所有datasets功能 + DatasetPlus增强功能
```

### 🔄 特性1: 完全兼容datasets

```python
# 所有datasets的方法都可以直接使用
ds = dsp.ds  # 获取原生dataset对象
dsp_shuffled = dsp.shuffle(seed=42)
dsp_split = dsp.train_test_split(test_size=0.2)
dsp_filtered = dsp.filter(lambda x: len(x['text']) > 100)

# pandas也可以无缝衔接
dsp_df = dsp.to_pandas()
dsp = DatasetPlus.from_pandas(dsp_df)

# 静态方法也完全支持
dsp_from_dict = DatasetPlus.from_dict({"text": ["hello", "world"]})
dsp_from_hf = DatasetPlus.from_pretrained("squad")

# 与原生datasets无缝切换
from datasets import Dataset
ds = Dataset.from_dict({"text": ["test"]})
dsp = DatasetPlus(ds)  # 直接包装现有dataset

# jupyter友好显示
dsp = DatasetPlus.from_dict({"text": ["a", "b", "c"]})
dsp
----------------
DatasetPlus({
    features: ['text'],
    num_rows: 3
})
~~~~~~~~~~~~~~~~~

## 人性化的切片、显示逻辑1
dsp[0] # 等价于 ds.select(range(0)) ,当然dsp同样支持dsp.select(range(0))
----------------
{'text': 'a'}

## 人性化的切片、显示逻辑2
dsp[1:2] #等价于 ds.select(range(1,2)) 
----------------
DatasetPlus({
    features: ['text'],
    num_rows: 1

## 人性化的切片、显示逻辑3
dsp[1:] 
----------------
DatasetPlus({
    features: ['text'],
    num_rows: 2
})
```

### 🧠 特性2: 智能缓存 - 大模型调用零丢失

```python
# 定义包含大模型调用的处理函数
def enhance_with_llm(example):
    # 初始化LLM工具（多进程的时候需要示例化在内部）
    llm = MyLLMTool(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
    # 这里调用大模型进行数据增强
    prompt = f"请为以下文本生成摘要: {example['text']}"
    summary = llm.getResult(prompt)
    example['summary'] = summary
    return example

# 第一次运行 - 会调用大模型
dsp_enhanced = dsp.map(enhance_with_llm, num_proc=4, cache=True)

# 在Jupyter中忘记赋值？没关系！
# 即使你运行了: dsp.map(enhance_with_llm, cache=True)  # 忘记赋值
# 也可以通过以下方式恢复结果:
dsp_enhanced = dsp.map(enhance_with_llm, cache=True)  # 自动从缓存读取，不会重新调用LLM

# 中断后继续处理也没问题，已处理的数据会自动跳过
```

### 📈 特性3: 数据扩充 - 一行变多行

```python
def expand_data_with_llm(example):
    # 使用LLM生成多个相关问题
    prompt = f"基于以下文本生成3个相关问题，用JSON数组格式返回: {example['text']}"
    questions_json = llm.getResult(prompt)
    
    try:
        questions = json.loads(questions_json)
        # 返回数组，DatasetPlus会自动展开为多行
        return [{
            'original_text': example['text'],
            'question': q,
            'source': 'llm_generated'
        } for q in questions]
    except:
        return example  # 解析失败时返回原数据 或者直接删除: return None

# 原始数据: 100行
# 处理后: 可能变成300行 (每行生成3个问题)
dsp_expanded = dsp.map(expand_data_with_llm, cache=True)
print(f"原始数据: {len(dsp)} 行")
print(f"扩充后: {len(dsp_expanded)} 行")
```

### 🔍 特性4: 智能筛选 - 返回None自动删除

```python
def filter_with_llm(example):
    # 使用LLM进行质量评估
    prompt = f"""评估以下文本的质量，返回JSON格式: {{"quality": "high/mid/low"}}
    文本: {example['text']}"""
    
    result = llm.getResult(prompt)
    try:
        quality_data = json.loads(result)
        quality = quality_data.get('quality', 'low')
        
        # 只保留高质量数据，其他返回None会被自动删除
        if quality == 'high':
            example['quality_score'] = quality
            return example
        else:
            return None  # 自动删除低质量数据
    except:
        return None  # 解析失败也删除

# 原始数据: 1000行
# 筛选后: 可能只剩200行高质量数据
dsp_filtered = dsp.map(filter_with_llm, cache=True)
print(f"筛选前: {len(dsp)} 行")
print(f"筛选后: {len(dsp_filtered)} 行")
```

### 🎨 特性5: 无中生有 - 直接从大模型生成数据

```python
# 使用iter方法从大模型直接生成数据
def generate_dialogues(example):
    llm = MyLLMTool(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
    
    # 提示词：生成10段客服对话
    prompt = """请生成10段不同的客服与用户对话，每段包含问题和解答。
    要求：
    1. 每段对话用户提出不同的具体问题
    2. 客服给出专业回答
    3. 返回JSON数组格式: [{"user": "用户问题1", "assistant": "客服回答1", "category": "问题分类1"}, ...]
    4. 涵盖不同类型的问题（技术支持、售后服务、产品咨询等）
    """
    
    try:
        result = llm.getResult(prompt)
        dialogues_data = json.loads(result)
        
        # 返回数组，DatasetPlus会自动展开为多行
        return [{
            'batch_id': example['id'],
            'dialogue_id': i,
            'user': dialogue.get('user', ''),
            'assistant': dialogue.get('assistant', ''),
            'category': dialogue.get('category', ''),
            'source': 'generated'
        } for i, dialogue in enumerate(dialogues_data)]
    except Exception as e:
        print(f"生成失败: {e}")
        return None  # 生成失败则跳过

# 生成10批对话数据，每批包含10段对话
dsp_generated = DatasetPlus.iter(
    iterate_num=10,           # 生成10批数据
    fn=generate_dialogues,    # 生成函数
    num_proc=2,              # 2个进程并行 
    cache=False              # iter的cache默认是False
)

print(f"生成了 {len(dsp_generated)} 条对话数据")  # 应该是100条（10批 × 10段）
print(dsp_generated[0])  # 查看第一条生成的数据
```

## 📁 支持的数据格式

- **JSON/JSONL**: 标准JSON和JSON Lines格式
- **CSV**: 逗号分隔值文件
- **Excel**: .xlsx和.xls文件
- **Hugging Face Datasets**: Hub上的任何数据集
- **Dataframe, datasets**: 支持pandas DataFrame和Hugging Face datasets
- **目录批量加载**: 自动合并目录下的多个文件

## 🔧 高级功能详解

### 智能缓存机制原理

```python
# 缓存基于函数内容的哈希值，确保函数改变时重新计算
def process_v1(example):
    return {"result": example["text"].upper()}  # 版本1

def process_v2(example):
    return {"result": example["text"].lower()}  # 版本2

# 不同函数会生成不同缓存，互不干扰
ds1 = ds.map(process_v1, cache=True)  # 缓存A
ds2 = ds.map(process_v2, cache=True)  # 缓存B
```

### 批量处理与多进程

```python
# 大数据集的高效处理
dsp = DatasetPlus.load_dataset("large_dataset.jsonl")
dsp_processed = dsp.map(
    enhance_with_llm,
    num_proc=8,           # 8个进程并行
    max_inner_num=1000,   # 每批处理1000条
    cache=True            # 启用缓存
)
```

### 目录批量加载

```python
# 自动加载并合并目录下的所有支持文件
dsp = DatasetPlus.load_dataset_plus("./data_folder/")
# 支持混合格式: data_folder/
#   ├── file1.jsonl
#   ├── file2.csv
#   └── file3.xlsx
```

### Excel专业处理

```python
from datasetplus import DatasetPlusExcels

# Excel文件的专业处理
excel_dsp = DatasetPlusExcels("spreadsheet.xlsx")

# 支持多sheet处理
sheet_names = excel_dsp.get_sheet_names()
for sheet in sheet_names:
    sheet_data = excel_dsp.get_sheet_data(sheet)
    dsp_processed = excel_dsp.map(lambda x: {'cleaned': x['column'].strip()})
```

## 📚 API参考

### DatasetPlus

增强的数据集处理类，完全兼容Hugging Face datasets。

#### 核心方法

- `map(fn, num_proc=1, max_inner_num=1000, cache=True)`: 增强的映射函数
  - **fn**: 处理函数，支持返回数组(自动展开)和None(自动删除)
  - **cache**: 智能缓存，基于函数内容自动生成缓存键
  - **num_proc**: 多进程并行处理
  - **max_inner_num**: 批处理大小

#### 静态方法

- `load_dataset(file_name, output_file)`: 加载单个数据集文件
- `load_dataset_plus(input_path, output_file)`: 从文件、目录或Hub加载
- `from_pandas(df)`: 从pandas DataFrame创建
- `from_dict(data)`: 从字典创建
- `from_pretrained(path)`: 从Hugging Face Hub加载
- `iter(iterate_num, fn, num_proc=1, max_inner_num=1000, cache=True)`: 无中生有，迭代生成数据
  - **iterate_num**: 生成数据的数量
  - **fn**: 生成函数，接收包含id的example，返回生成的数据
  - **num_proc**: 多进程并行处理
  - **cache**: 启用缓存，避免重复生成

#### 兼容性

```python
# 所有datasets方法都可直接使用
ds.shuffle()          # 打乱数据
ds.filter()           # 过滤数据
ds.select()           # 选择数据
ds.train_test_split() # 分割数据
ds.save_to_disk()     # 保存到磁盘
# ... 以及所有其他datasets方法
```

### MyLLMTool

大模型调用工具，支持OpenAI兼容的API。

#### 初始化

```python
llm = MyLLMTool(
    model_name="gpt-3.5-turbo",      # 模型名称
    base_url="https://api.openai.com/v1",  # API基础URL
    api_key="your-api-key"           # API密钥
)
```

#### 方法

- `getResult(query, sys_prompt=None, temperature=0.7, top_p=1, max_tokens=2048, model_name=None)`: 获取LLM响应
  - **query**: 用户查询
  - **sys_prompt**: 系统提示词
  - **temperature**: 温度参数
  - **max_tokens**: 最大token数

### DatasetPlusExcels

Excel文件专业处理类。

#### 方法

- `__init__(file_path, output_file)`: 初始化Excel处理器
- `get_sheet_names()`: 获取所有sheet名称
- `get_sheet_data(sheet_name)`: 获取指定sheet的数据

## 🎯 实际使用场景

### 场景1: 大规模数据标注

```python
# 使用LLM对大量文本进行情感分析标注
def sentiment_labeling(example):
    prompt = f"分析以下文本的情感倾向，返回positive/negative/neutral: {example['text']}"
    sentiment = llm.getResult(prompt)
    example['sentiment'] = sentiment.strip()
    return example

# 处理10万条数据，支持断点续传
dsp_labeled = dsp.map(sentiment_labeling, cache=True, num_proc=4)
```

### 场景2: 数据质量筛选

```python
# 使用LLM筛选高质量的训练数据
def quality_filter(example):
    prompt = f"评估文本质量(1-5分): {example['text']}"
    score = llm.getResult(prompt)
    try:
        if int(score) >= 4:
            return example
        else:
            return None  # 低质量数据自动删除
    except:
        return None

dsp_filtered = dsp.map(quality_filter, cache=True)
```

### 场景3: 数据增强扩充

```python
# 为每条数据生成多个变体
def data_augmentation(example):
    prompt = f"为以下文本生成3个同义改写: {example['text']}"
    variants = llm.getResult(prompt).split('\n')
    
    # 返回数组，自动展开为多行
    return [{
        'text': variant.strip(),
        'label': example['label'],
        'source': 'augmented'
    } for variant in variants if variant.strip()]

dsp_augmented = dsp.map(data_augmentation, cache=True)
```

### 场景4: 从零生成训练数据

```python
# 使用LLM从零生成训练数据
def generate_qa_pairs(example):
    llm = MyLLMTool(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
    
    # 生成问答对的提示词
    prompt = """生成一个关于Python编程的问答对。
    要求：
    1. 问题要具体且有实用价值
    2. 答案要准确且详细
    3. 返回JSON格式: {"question": "问题", "answer": "答案", "difficulty": "easy/medium/hard"}
    """
    
    try:
        result = llm.getResult(prompt)
        qa_data = json.loads(result)
        return {
            'id': example['id'],
            'question': qa_data.get('question', ''),
            'answer': qa_data.get('answer', ''),
            'difficulty': qa_data.get('difficulty', 'medium'),
            'domain': 'python_programming',
            'generated_at': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"生成失败: {e}")
        return None

# 生成1000个Python编程问答对
dsp_qa_dataset = DatasetPlus.iter(
    iterate_num=1000,
    fn=generate_qa_pairs,
    num_proc=4,
    cache=True
)

print(f"成功生成 {len(dsp_qa_dataset)} 个问答对")
```

## 💡 最佳实践

### 1. 缓存策略
- 始终开启缓存: `cache=True`
- 大模型调用友好，即使中间网络不稳定、额度不足等导致中断，之前的结果不丢失
- 函数修改后会自动重新计算

### 2. 性能优化
- 合理设置 `num_proc` (根据大模型能接受的最大并发，进行设置)
- 调整 `max_inner_num` （最大内存数据存储量，每max_inner_num会写入磁盘持久化）
- 大数据集使用分批处理

### 3. 错误处理
```python
def robust_processing(example):
    try:
        # LLM调用
        result = llm.getResult(prompt)
        return process_result(result)
    except Exception as e:
        print(f"处理失败: {e}")
        return None  # 失败的数据自动删除
```

## 📋 系统要求

- **Python**: >= 3.7
- **datasets**: >= 2.0.0
- **pandas**: >= 1.3.0
- **numpy**: >= 1.21.0
- **openpyxl**: >= 3.0.0 (Excel支持)
- **openai**: >= 1.0.0 (LLM支持)

## 🤝 贡献

欢迎提交Pull Request！请确保:
- 代码符合项目规范
- 添加适当的测试
- 更新相关文档

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

## 📝 更新日志

### v0.2.0 (最新)
- ✨ 新增智能缓存系统
- ✨ 支持数组自动展开
- ✨ 支持None自动筛选
- ✨ 完全兼容datasets API
- ✨ 新增MyLLMTool大模型工具

### v0.1.0
- 🎉 初始发布
- 📁 基础数据集加载功能
- 📊 Excel文件支持
- ⚡ 缓存和批处理
- 📂 目录加载支持