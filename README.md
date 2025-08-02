# DatasetPlus

[中文版本](README-CN.md)

An enhanced Hugging Face datasets wrapper designed for large model data processing, providing intelligent caching, data augmentation, and filtering capabilities.


## 🚀 Core Features

### 1. 🔄 Full Compatibility - Seamless datasets Replacement
- **100% Compatible**: Supports all datasets methods and attributes
- **Mutual Conversion, Plug and Play**: Simply `dsp = DatasetPlus(ds)` to convert back: `ds = dsp.ds`
- **Static Method Support**: All Dataset static methods and class methods can be called directly

### 2. 🧠 Intelligent Caching System - Zero Loss for Large Model Calls
- **Automatic Function-level Caching**: Automatically generates cache based on function content, even if interrupted due to network instability, insufficient quota, etc., previous results won't be lost
- **Jupyter Friendly**: Even if you forget to assign variables, results can be recovered from cache
- **Resume from Breakpoint**: Supports continuing processing after interruption, automatically reads cache for processed data

### 3. 📈 Data Augmentation - One Row Becomes Multiple Rows
- **Array Auto-expansion**: Automatically expands arrays returned by map function into multiple rows of data
- **LLM Result Parsing**: Easy data augmentation with MyLLMTool

### 4. 🔍 Intelligent Filtering - Auto-delete When Returning None
- **Conditional Filtering**: Automatically deletes rows when map function returns None
- **LLM Intelligent Filtering**: Use large models for complex conditional filtering

### 5. 🎨 Generate from Scratch - Direct Data Generation from Large Models
- **iter Method**: Supports generating datasets from scratch
- **Flexible Generation**: Can generate data in any format and content
- **Batch Generation**: Supports large-scale data generation with automatic caching and parallel processing

## 📦 Installation

### Install from PyPI

```bash
pip install datasetplus
```

### Install from Source

```bash
git clone https://github.com/yourusername/datasetplus.git
cd datasetplus
pip install -e .
```

### Dependencies Installation

```bash
# Basic dependencies
pip install datasets pandas numpy

# Excel support
pip install openpyxl

# LLM support
pip install openai
```

## 🎯 Quick Start

### Basic Usage
```python
from datasetplus import DatasetPlus, MyLLMTool

# Load dataset
dsp = DatasetPlus.load_dataset("data.jsonl")

# Fully compatible with datasets - plug and play
ds = dsp.ds  # Now ds has all datasets functionality + DatasetPlus enhancements
```

### 🔄 Feature 1: Full datasets Compatibility

```python
# All datasets methods can be used directly
ds = dsp.ds  # Get native dataset object
dsp_shuffled = dsp.shuffle(seed=42)
dsp_split = dsp.train_test_split(test_size=0.2)
dsp_filtered = dsp.filter(lambda x: len(x['text']) > 100)

# pandas can also seamlessly connect
dsp_df = dsp.to_pandas()
dsp = DatasetPlus.from_pandas(dsp_df)

# Static methods are fully supported
dsp_from_dict = DatasetPlus.from_dict({"text": ["hello", "world"]})
dsp_from_hf = DatasetPlus.from_pretrained("squad")

# Seamless switching with native datasets
from datasets import Dataset
ds = Dataset.from_dict({"text": ["test"]})
dsp = DatasetPlus(ds)  # Directly wrap existing dataset

# Jupyter-friendly display
dsp = DatasetPlus.from_dict({"text": ["a", "b", "c"]})
dsp
----------------
DatasetPlus({
    features: ['text'],
    num_rows: 3
})
~~~~~~~~~~~~~~~~~

## Humanized slicing and display logic 1
dsp[0] # Equivalent to ds.select(range(0)), of course dsp also supports dsp.select(range(0))
----------------
{'text': 'a'}

## Humanized slicing and display logic 2
dsp[1:2] # Equivalent to ds.select(range(1,2))
----------------
DatasetPlus({
    features: ['text'],
    num_rows: 1

## Humanized slicing and display logic 3
dsp[1:]
----------------
DatasetPlus({
    features: ['text'],
    num_rows: 2
})
```

### 🧠 Feature 2: Intelligent Caching - Zero Loss for Large Model Calls

```python
# Define processing function containing large model calls
def enhance_with_llm(example):
    # Initialize LLM tool (needs to be instantiated internally for multiprocessing)
    llm = MyLLMTool(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
    # Call large model for data enhancement
    prompt = f"Please generate a summary for the following text: {example['text']}"
    summary = llm.getResult(prompt)
    example['summary'] = summary
    return example

# First run - will call large model
dsp_enhanced = dsp.map(enhance_with_llm, num_proc=4, cache=True)

# Forgot to assign in Jupyter? No problem!
# Even if you ran: dsp.map(enhance_with_llm, cache=True)  # Forgot assignment
# You can still recover results with:
dsp_enhanced = dsp.map(enhance_with_llm, cache=True)  # Auto-read from cache, won't call LLM again

# Continuing after interruption is also fine, processed data will be automatically skipped
```

### 📈 Feature 3: Data Augmentation - One Row Becomes Multiple Rows

```python
def expand_data_with_llm(example):
    # Use LLM to generate multiple related questions
    prompt = f"Generate 3 related questions based on the following text, return in JSON array format: {example['text']}"
    questions_json = llm.getResult(prompt)
    
    try:
        questions = json.loads(questions_json)
        # Return array, DatasetPlus will automatically expand into multiple rows
        return [{
            'original_text': example['text'],
            'question': q,
            'source': 'llm_generated'
        } for q in questions]
    except:
        return example  # Return original data on parsing failure or delete directly: return None

# Original data: 100 rows
# After processing: might become 300 rows (3 questions generated per row)
dsp_expanded = dsp.map(expand_data_with_llm, cache=True)
print(f"Original data: {len(dsp)} rows")
print(f"After expansion: {len(dsp_expanded)} rows")
```

### 🔍 Feature 4: Intelligent Filtering - Auto-delete When Returning None

```python
def filter_with_llm(example):
    # Use LLM for quality assessment
    prompt = f"""Evaluate the quality of the following text, return JSON format: {{"quality": "high/mid/low"}}
    Text: {example['text']}"""
    
    result = llm.getResult(prompt)
    try:
        quality_data = json.loads(result)
        quality = quality_data.get('quality', 'low')
        
        # Only keep high-quality data, others returning None will be auto-deleted
        if quality == 'high':
            example['quality_score'] = quality
            return example
        else:
            return None  # Auto-delete low-quality data
    except:
        return None  # Also delete on parsing failure

# Original data: 1000 rows
# After filtering: might only have 200 rows of high-quality data
dsp_filtered = dsp.map(filter_with_llm, cache=True)
print(f"Before filtering: {len(dsp)} rows")
print(f"After filtering: {len(dsp_filtered)} rows")
```

### 🎨 Feature 5: Generate from Scratch - Direct Data Generation from Large Models

```python
# Use iter method to generate data directly from large models
def generate_dialogues(example):
    llm = MyLLMTool(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
    
    # Prompt: Generate 10 customer service dialogues
    prompt = """Please generate 10 different customer service dialogues, each containing questions and answers.
    Requirements:
    1. Each dialogue has users asking different specific questions
    2. Customer service provides professional answers
    3. Return JSON array format: [{"user": "User question 1", "assistant": "Service answer 1", "category": "Question category 1"}, ...]
    4. Cover different types of questions (technical support, after-sales service, product consultation, etc.)
    """
    
    try:
        result = llm.getResult(prompt)
        dialogues_data = json.loads(result)
        
        # Return array, DatasetPlus will automatically expand into multiple rows
        return [{
            'batch_id': example['id'],
            'dialogue_id': i,
            'user': dialogue.get('user', ''),
            'assistant': dialogue.get('assistant', ''),
            'category': dialogue.get('category', ''),
            'source': 'generated'
        } for i, dialogue in enumerate(dialogues_data)]
    except Exception as e:
        print(f"Generation failed: {e}")
        return None  # Skip on generation failure

# Generate 10 batches of dialogue data, each batch containing 10 dialogues
dsp_generated = DatasetPlus.iter(
    iterate_num=10,           # Generate 10 batches of data
    fn=generate_dialogues,    # Generation function
    num_proc=2,              # 2 processes in parallel
    cache=False              # iter cache defaults to False
)

print(f"Generated {len(dsp_generated)} dialogue data")  # Should be 100 (10 batches × 10 dialogues)
print(dsp_generated[0])  # View first generated data
```

## 📁 Supported Data Formats

- **JSON/JSONL**: Standard JSON and JSON Lines format
- **CSV**: Comma-separated values files
- **Excel**: .xlsx and .xls files
- **Hugging Face Datasets**: Any dataset from the Hub
- **Dataframe, datasets**: Support pandas DataFrame and Hugging Face datasets
- **Directory Batch Loading**: Automatically merge multiple files in directory

## 🔧 Advanced Features

### Intelligent Caching Mechanism

```python
# Cache based on function content hash, ensures recalculation when function changes
def process_v1(example):
    return {"result": example["text"].upper()}  # Version 1

def process_v2(example):
    return {"result": example["text"].lower()}  # Version 2

# Different functions generate different caches, no interference
ds1 = ds.map(process_v1, cache=True)  # Cache A
ds2 = ds.map(process_v2, cache=True)  # Cache B
```

### Batch Processing and Multiprocessing

```python
# Efficient processing of large datasets
dsp = DatasetPlus.load_dataset("large_dataset.jsonl")
dsp_processed = dsp.map(
    enhance_with_llm,
    num_proc=8,           # 8 processes in parallel
    max_inner_num=1000,   # Process 1000 items per batch
    cache=True            # Enable caching
)
```

### Directory Batch Loading

```python
# Automatically load and merge all supported files in directory
dsp = DatasetPlus.load_dataset_plus("./data_folder/")
# Supports mixed formats: data_folder/
#   ├── file1.jsonl
#   ├── file2.csv
#   └── file3.xlsx
```

### Professional Excel Processing

```python
from datasetplus import DatasetPlusExcels

# Professional Excel file processing
excel_dsp = DatasetPlusExcels("spreadsheet.xlsx")

# Support multi-sheet processing
sheet_names = excel_dsp.get_sheet_names()
for sheet in sheet_names:
    sheet_data = excel_dsp.get_sheet_data(sheet)
    dsp_processed = excel_dsp.map(lambda x: {'cleaned': x['column'].strip()})
```

## 📚 API Reference

### DatasetPlus

Enhanced dataset processing class, fully compatible with Hugging Face datasets.

#### Core Methods

- `map(fn, num_proc=1, max_inner_num=1000, cache=True)`: Enhanced mapping function
  - **fn**: Processing function, supports returning arrays (auto-expand) and None (auto-delete)
  - **cache**: Intelligent caching, automatically generates cache keys based on function content
  - **num_proc**: Multiprocess parallel processing
  - **max_inner_num**: Batch processing size

#### Static Methods

- `load_dataset(file_name, output_file)`: Load single dataset file
- `load_dataset_plus(input_path, output_file)`: Load from file, directory, or Hub
- `from_pandas(df)`: Create from pandas DataFrame
- `from_dict(data)`: Create from dictionary
- `from_pretrained(path)`: Load from Hugging Face Hub
- `iter(iterate_num, fn, num_proc=1, max_inner_num=1000, cache=True)`: Generate from scratch, iteratively generate data
  - **iterate_num**: Number of data to generate
  - **fn**: Generation function, receives example with id, returns generated data
  - **num_proc**: Multiprocess parallel processing
  - **cache**: Enable caching, avoid duplicate generation

#### Compatibility

```python
# All datasets methods can be used directly
ds.shuffle()          # Shuffle data
ds.filter()           # Filter data
ds.select()           # Select data
ds.train_test_split() # Split data
ds.save_to_disk()     # Save to disk
# ... and all other datasets methods
```

### MyLLMTool

Large model calling tool, supports OpenAI-compatible APIs.

#### Initialization

```python
llm = MyLLMTool(
    model_name="gpt-3.5-turbo",      # Model name
    base_url="https://api.openai.com/v1",  # API base URL
    api_key="your-api-key"           # API key
)
```

#### Methods

- `getResult(query, sys_prompt=None, temperature=0.7, top_p=1, max_tokens=2048, model_name=None)`: Get LLM response
  - **query**: User query
  - **sys_prompt**: System prompt
  - **temperature**: Temperature parameter
  - **max_tokens**: Maximum tokens

### DatasetPlusExcels

Professional Excel file processing class.

#### Methods

- `__init__(file_path, output_file)`: Initialize Excel processor
- `get_sheet_names()`: Get all sheet names
- `get_sheet_data(sheet_name)`: Get data from specified sheet

## 🎯 Real-world Use Cases

### Case 1: Large-scale Data Annotation

```python
# Use LLM for sentiment analysis annotation on large amounts of text
def sentiment_labeling(example):
    prompt = f"Analyze the sentiment of the following text, return positive/negative/neutral: {example['text']}"
    sentiment = llm.getResult(prompt)
    example['sentiment'] = sentiment.strip()
    return example

# Process 100,000 data points, supports resume from breakpoint
dsp_labeled = dsp.map(sentiment_labeling, cache=True, num_proc=4)
```

### Case 2: Data Quality Filtering

```python
# Use LLM to filter high-quality training data
def quality_filter(example):
    prompt = f"Rate text quality (1-5 points): {example['text']}"
    score = llm.getResult(prompt)
    try:
        if int(score) >= 4:
            return example
        else:
            return None  # Auto-delete low-quality data
    except:
        return None

dsp_filtered = dsp.map(quality_filter, cache=True)
```

### Case 3: Data Augmentation

```python
# Generate multiple variants for each data point
def data_augmentation(example):
    prompt = f"Generate 3 synonymous rewrites for the following text: {example['text']}"
    variants = llm.getResult(prompt).split('\n')
    
    # Return array, automatically expand into multiple rows
    return [{
        'text': variant.strip(),
        'label': example['label'],
        'source': 'augmented'
    } for variant in variants if variant.strip()]

dsp_augmented = dsp.map(data_augmentation, cache=True)
```

### Case 4: Generate Training Data from Scratch

```python
# Use LLM to generate training data from scratch
def generate_qa_pairs(example):
    llm = MyLLMTool(
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
    
    # Prompt for generating Q&A pairs
    prompt = """Generate a Q&A pair about Python programming.
    Requirements:
    1. Question should be specific and practical
    2. Answer should be accurate and detailed
    3. Return JSON format: {"question": "question", "answer": "answer", "difficulty": "easy/medium/hard"}
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
        print(f"Generation failed: {e}")
        return None

# Generate 1000 Python programming Q&A pairs
dsp_qa_dataset = DatasetPlus.iter(
    iterate_num=1000,
    fn=generate_qa_pairs,
    num_proc=4,
    cache=True
)

print(f"Successfully generated {len(dsp_qa_dataset)} Q&A pairs")
```

## 💡 Best Practices

### 1. Caching Strategy
- Always enable caching: `cache=True`
- Large model call friendly, even if interrupted due to network instability, insufficient quota, etc., previous results won't be lost
- Automatically recalculates when function is modified

### 2. Performance Optimization
- Set `num_proc` reasonably (based on maximum concurrency the large model can accept)
- Adjust `max_inner_num` (maximum memory data storage, writes to disk for persistence every max_inner_num)
- Use batch processing for large datasets

### 3. Error Handling
```python
def robust_processing(example):
    try:
        # LLM call
        result = llm.getResult(prompt)
        return process_result(result)
    except Exception as e:
        print(f"Processing failed: {e}")
        return None  # Failed data auto-deleted
```

## 📋 System Requirements

- **Python**: >= 3.7
- **datasets**: >= 2.0.0
- **pandas**: >= 1.3.0
- **numpy**: >= 1.21.0
- **openpyxl**: >= 3.0.0 (Excel support)
- **openai**: >= 1.0.0 (LLM support)

## 🤝 Contributing

Pull Requests are welcome! Please ensure:
- Code follows project standards
- Add appropriate tests
- Update relevant documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Changelog

### v0.2.0 (Latest)
- ✨ Added intelligent caching system
- ✨ Support array auto-expansion
- ✨ Support None auto-filtering
- ✨ Full datasets API compatibility
- ✨ Added MyLLMTool large model tool

### v0.1.0
- 🎉 Initial release
- 📁 Basic dataset loading functionality
- 📊 Excel file support
- ⚡ Caching and batch processing
- 📂 Directory loading support