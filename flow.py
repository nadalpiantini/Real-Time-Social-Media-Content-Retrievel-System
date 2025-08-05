from bytewax import operators as op
from typing import List
from bytewax.dataflow import Dataflow
from qdrant_client import QdrantClient
from utils.embedding import EmbeddingModelSingleton
from models.data_source import UnifiedDataSource
from models.post import ChunkedPost, CleanedPost, EmbeddedChunkedPost, RawPost
from utils.qdrant import QdrantVectorOutput
import streamlit as st


def log_step(step_name: str, data):
    """Helper function to log pipeline steps"""
    if hasattr(data, 'post_id'):
        message = f"🔄 {step_name}: Processing post {data.post_id}"
    else:
        message = f"🔄 {step_name}: Processing {type(data).__name__}"
    
    print(message)
    return data


def build(in_memory: bool = False, data_source_path: List[str]=None):
    print("🚀 Building Bytewax flow...")
    
    embedding_model = EmbeddingModelSingleton()
    flow = Dataflow("flow")
    
    print(f"📊 Data source paths: {data_source_path}")
    
    # Use UnifiedDataSource with auto-detection of Supabase or JSON files
    stream = op.input(
        "input", flow, UnifiedDataSource(json_files=data_source_path)
    )
    
    # Add logging at each step
    stream = op.map("log_input", stream, lambda x: log_step("Input", x))
    stream = op.map("raw_post", stream, RawPost.from_source)
    stream = op.map("log_raw", stream, lambda x: log_step("Raw Post", x))
    stream = op.map("cleaned_post", stream, CleanedPost.from_raw_post)
    stream = op.map("log_cleaned", stream, lambda x: log_step("Cleaned Post", x))
    stream = op.flat_map(
        "chunked_post",
        stream,
        lambda cleaned_post: ChunkedPost.from_cleaned_post(
            cleaned_post, embedding_model=embedding_model
        ),
    )
    stream = op.map("log_chunked", stream, lambda x: log_step("Chunked Post", x))
    stream = op.map(
        "embedded_chunked_post",
        stream,
        lambda chunked_post: EmbeddedChunkedPost.from_chunked_post(
            chunked_post, embedding_model=embedding_model
        ),
    )
    stream = op.map("log_embedded", stream, lambda x: log_step("Embedded Post", x))
    
    # Enhanced inspect with more details and completion tracking
    def detailed_inspect(step_id, data):
        message = f"🔍 Final: Post {data.post_id} → {len(data.text)} chars → {len(data.text_embedding)} dims"
        print(message)
        return data
    
    op.inspect("inspect", stream, detailed_inspect)
    
    # Add a completion marker to help detect when processing is done
    def mark_completion(data):
        print(f"🏁 Processing completed for post {data.post_id}")
        return data
    
    stream = op.map("completion_marker", stream, mark_completion)
    
    op.output(
        "output", stream, _build_output(model=embedding_model, in_memory=in_memory)
    )
    
    print("✅ Bytewax flow built successfully")
    
    return flow


def _build_output(model: EmbeddingModelSingleton, in_memory: bool = False):
    if in_memory:
        return QdrantVectorOutput(
            vector_size=model.embedding_size,
            client=QdrantClient(":memory:"),
        )
    else:
        return QdrantVectorOutput(
            vector_size=model.embedding_size,
        )
