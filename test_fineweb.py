import pyarrow.parquet as pq
import tiktoken
import numpy as np

print("Testing parquet read...")
try:
    table = pq.read_table("/work/scratch/data/datasets/fineweb/_downloads/sample/100BT/000_00000.parquet", columns=["text"])
    texts = table.column("text").to_pylist()
    print(f"Read {len(texts)} texts.")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    all_encoded = tokenizer.encode_ordinary_batch(texts[:100])
    print("Tokenized 100 texts successfully.")
except Exception as e:
    print(f"Error: {e}")
