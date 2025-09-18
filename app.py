"""
Streamlit app: Transcript Sentiment Scorer (Single-file)

Updates in this version:
- Token-aware chunking using the tokenizer so no input exceeds model max tokens (512)
- Tokenizer is cached per-process to avoid repeated downloads
- HF scoring explicitly truncates via tokenizer as extra safety
- All other features retained: ONNX export & quantize button, ONNX runtime inference, multiprocessing, checkpointing, single-file app
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------- Configuration -------------------------
_ONNX_DIR = "onnx_model"
_ONNX_QUANT_PATH = os.path.join(_ONNX_DIR, "model_quant.onnx")
_ONNX_RAW_PATH = os.path.join(_ONNX_DIR, "model.onnx")

# Per-process cache for worker initialization
_process_state = {
    'onnx_available': None,
    'onnx_session': None,
    'tokenizer': None,   # HF tokenizer cached per process
    'hf_pipeline': None
}

# ------------------------- Helper utilities -------------------------

def chunk_text_by_tokens(text, tokenizer_name="distilbert-base-uncased", max_model_tokens=512, safe_margin=32):
    """
    Token-aware chunker that uses tokenizer to ensure returned chunks do not exceed
    (max_model_tokens - safe_margin) token length. Caches tokenizer in _process_state for reuse.
    Returns a list of text chunks.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []

    # lazy-init tokenizer per process
    if _process_state.get('tokenizer') is None:
        try:
            from transformers import AutoTokenizer
            _process_state['tokenizer'] = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        except Exception:
            # fallback to simple split if tokenizer can't be loaded
            words = text.split()
            max_words = 150
            return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

    tokenizer = _process_state['tokenizer']
    max_chunk_tokens = max_model_tokens - safe_margin

    # tokenize entire text to ids (no special tokens)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_chunk_tokens:
        return [text]

    chunks = []
    start = 0
    total = len(token_ids)
    while start < total:
        end = min(start + max_chunk_tokens, total)
        # decode token slice back to text for scoring (clean spaces)
        chunk = tokenizer.decode(token_ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk)
        start = end
    return chunks

# ------------------------- ONNX export & quantize helper (runs when user requests) -------------------------

def export_and_quantize_onnx(model_name="distilbert-base-uncased-finetuned-sst-2-english", out_dir=_ONNX_DIR):
    """Export a Hugging Face text-classification model to ONNX and quantize it (INT8 dynamic).
    This function will attempt to install required packages if missing and then run export + quantization.
    It returns the path to the quantized onnx file on success.
    """
    import subprocess, sys, shutil, glob

    os.makedirs(out_dir, exist_ok=True)

    # Ensure necessary packages are installed; attempt light-weight installs
    try:
        import onnxruntime as ort
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from optimum.exporters.onnx import export
    except Exception:
        # Install required packages
        pkg_install_cmds = [
            [sys.executable, "-m", "pip", "install", "optimum[export]"],
            [sys.executable, "-m", "pip", "install", "onnx", "onnxruntime", "onnxruntime-tools"]
        ]
        for cmd in pkg_install_cmds:
            subprocess.check_call(cmd)
        # re-import
        import onnxruntime as ort
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from optimum.exporters.onnx import export

    # Export using optimum export helper
    try:
        export(model=model_name, output=out_dir, device="cpu", opset=13, optimize=True, task="text-classification")
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")

    # find the ONNX file in out_dir
    onnx_files = glob.glob(os.path.join(out_dir, "*.onnx"))
    if len(onnx_files) == 0:
        raise RuntimeError("No ONNX model file found after export")

    onnx_src = sorted(onnx_files, key=os.path.getsize, reverse=True)[0]
    onnx_dest = os.path.join(out_dir, "model.onnx")
    shutil.copyfile(onnx_src, onnx_dest)

    # Quantize dynamically to INT8
    quantized_path = os.path.join(out_dir, "model_quant.onnx")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_dest, quantized_path, weight_type=QuantType.QInt8)
    except Exception as e:
        # attempt a fallback quantize with different API
        try:
            from onnxruntime.quantization import quantize_dynamic as qd
            qd(onnx_dest, quantized_path)
        except Exception as e2:
            raise RuntimeError(f"ONNX quantization failed: {e} / {e2}")

    if not os.path.exists(quantized_path):
        raise RuntimeError("Quantized ONNX file not found after quantization")

    return quantized_path

# ------------------------- Scoring backends -------------------------

def _softmax(x, axis=1):
    import numpy as _np
    e_x = _np.exp(x - _np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _init_onnx():
    """Initialize ONNX session & tokenizer for current worker process if available."""
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except Exception:
        _process_state['onnx_available'] = False
        return False

    if not os.path.exists(_ONNX_QUANT_PATH):
        _process_state['onnx_available'] = False
        return False

    try:
        sess = ort.InferenceSession(_ONNX_QUANT_PATH, providers=["CPUExecutionProvider"]) 
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
        _process_state['onnx_session'] = sess
        _process_state['tokenizer'] = tokenizer
        _process_state['onnx_available'] = True
        return True
    except Exception:
        _process_state['onnx_available'] = False
        return False


def _init_hf_pipeline(batch_size=32):
    """Initialize HF pipeline for current worker process as fallback."""
    try:
        from transformers import pipeline, AutoTokenizer
    except Exception:
        _process_state['hf_pipeline'] = None
        return False
    try:
        _process_state['hf_pipeline'] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=batch_size)
        # also ensure tokenizer present for safe truncation
        if _process_state.get('tokenizer') is None:
            _process_state['tokenizer'] = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
        return True
    except Exception:
        _process_state['hf_pipeline'] = None
        return False


def score_texts_onnx(texts, max_length=512):
    import numpy as _np
    sess = _process_state.get('onnx_session')
    tokenizer = _process_state.get('tokenizer')
    if sess is None or tokenizer is None:
        raise RuntimeError("ONNX not initialized in worker")

    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='np')
    # create ort inputs - use tokenizer.model_input_names
    ort_inputs = {}
    for name in tokenizer.model_input_names:
        if name in enc:
            ort_inputs[name] = enc[name]
    if len(ort_inputs) == 0:
        ort_inputs = {k: enc[k] for k in enc}

    ort_outs = sess.run(None, ort_inputs)
    logits = ort_outs[0]
    probs = _softmax(logits, axis=1)
    labels = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)
    signed = [ (1 if int(lbl)==1 else -1) * float(score) for lbl, score in zip(labels, max_probs) ]
    return signed


def score_texts_hf(texts, batch_size=32, max_length=512):
    # ensure pipeline + tokenizer
    if _process_state.get('hf_pipeline') is None:
        _init_hf_pipeline(batch_size=batch_size)
    pipe = _process_state.get('hf_pipeline')
    tokenizer = _process_state.get('tokenizer')
    if pipe is None:
        raise RuntimeError("HF pipeline not available in worker")

    # Pre-truncate texts using tokenizer to guarantee token length limits
    safe_texts = []
    for t in texts:
        try:
            token_ids = tokenizer.encode(t, add_special_tokens=False)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                t_trunc = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                safe_texts.append(t_trunc)
            else:
                safe_texts.append(t)
        except Exception:
            safe_texts.append(t if isinstance(t, str) else "")

    results = pipe(safe_texts)
    signed = [ (1 if r['label'] == 'POSITIVE' else -1) * r['score'] for r in results ]
    return signed

# ------------------------- Worker function -------------------------

def process_dataframe_partition(df_partition, model_name, batch_size, chunk_words, partition_index=None, checkpoint_folder=None):
    # lazy init
    if _process_state.get('onnx_available') is None:
        _init_onnx()
    if not _process_state.get('onnx_available'):
        _init_hf_pipeline(batch_size=batch_size)

    use_onnx = _process_state.get('onnx_available') is True

    row_scores = []
    for idx, row in df_partition.iterrows():
        text = row.get('transcript', '')
        if not isinstance(text, str) or text.strip() == '':
            row_scores.append(0.0)
            continue

        # Token-aware chunking to ensure model max tokens not exceeded
        chunks = chunk_text_by_tokens(text, tokenizer_name="distilbert-base-uncased", max_model_tokens=512, safe_margin=32)
        if len(chunks) == 0:
            row_scores.append(0.0)
            continue

        chunk_scores = []
        step = max(1, batch_size * 8)
        for i in range(0, len(chunks), step):
            sub = chunks[i:i+step]
            try:
                if use_onnx:
                    sc = score_texts_onnx(sub)
                else:
                    sc = score_texts_hf(sub, batch_size=batch_size)
            except Exception:
                sc = []
                for stext in sub:
                    try:
                        if use_onnx:
                            sc.append(score_texts_onnx([stext])[0])
                        else:
                            sc.append(score_texts_hf([stext], batch_size=1)[0])
                    except Exception:
                        sc.append(0.0)
            chunk_scores.extend(sc)

        avg_score = float(np.mean(chunk_scores)) if len(chunk_scores) > 0 else 0.0
        row_scores.append(avg_score)

    result = df_partition.copy()
    result['sentiment_score'] = row_scores

    # checkpoint per partition
    if checkpoint_folder and partition_index is not None:
        try:
            os.makedirs(checkpoint_folder, exist_ok=True)
            chk_path = os.path.join(checkpoint_folder, f"checkpoint_part_{partition_index}.parquet")
            result.to_parquet(chk_path, index=False)
        except Exception:
            pass

    return result

# ------------------------- Streamlit UI -------------------------

st.set_page_config(page_title="Transcript Sentiment Scorer (Single-file)", layout="wide")
st.title("📞 Transcript Sentiment Scorer — Single-file (ONNX export baked in)")
st.markdown("This single-file app can export & quantize an ONNX DistilBERT model for faster CPU inference. If you prefer not to export ONNX, the app will fall back to using Hugging Face pipeline (slower).")

uploaded = st.file_uploader("Upload transcript file (CSV or Parquet)", type=["csv", "parquet"], accept_multiple_files=False)

col1, col2 = st.columns(2)
with col1:
    model_hint = st.text_input("HF model (used for fallback and for ONNX export)", value="distilbert-base-uncased-finetuned-sst-2-english")
    batch_size = st.number_input("Model batch size", min_value=1, max_value=256, value=32)
    chunk_words = st.number_input("Max words per chunk", min_value=50, max_value=1000, value=300)
with col2:
    num_workers = st.number_input("CPU workers (processes)", min_value=1, max_value=os.cpu_count() or 1, value=min(8, os.cpu_count() or 1))
    preview_rows = st.number_input("Preview rows after run", min_value=1, max_value=1000, value=5)

col3, col4 = st.columns(2)
with col3:
    st.write("ONNX model status:")
    if os.path.exists(_ONNX_QUANT_PATH):
        st.success(f"Quantized ONNX found: {_ONNX_QUANT_PATH}")
    elif os.path.exists(_ONNX_RAW_PATH):
        st.info(f"Raw ONNX found (not quantized): {_ONNX_RAW_PATH}")
    else:
        st.warning("No ONNX model found. You can export & quantize DistilBERT from the HF model by clicking the button below.")

with col4:
    if st.button("Export & Quantize ONNX model now"):
        with st.spinner("Exporting and quantizing ONNX model — this may take several minutes and will download model weights..."):
            try:
                qpath = export_and_quantize_onnx(model_name=model_hint, out_dir=_ONNX_DIR)
                st.success(f"ONNX quantized model created at: {qpath}")
            except Exception as e:
                st.error(f"Failed to export/quantize ONNX model: {e}")

start_btn = st.button("Start Processing")

if uploaded is not None and start_btn:
    temp_dir = tempfile.mkdtemp()
    local_path = os.path.join(temp_dir, uploaded.name)
    with open(local_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved uploaded file to: {local_path}")

    st.info("Preparing partitions... This may take a moment.")
    _, ext = os.path.splitext(local_path)
    ext = ext.lower()

    partitions = []
    if ext == ".parquet":
        try:
            df = pd.read_parquet(local_path)
        except Exception as e:
            st.error(f"Failed to read Parquet: {e}")
            st.stop()
        st.write(f"Loaded Parquet with {len(df)} rows")
        partitions = np.array_split(df, int(num_workers))
    elif ext == ".csv":
        chunksize = 20000
        reader = pd.read_csv(local_path, chunksize=chunksize)
        buffers = [[] for _ in range(num_workers)]
        idx = 0
        total_rows = 0
        for c in reader:
            buffers[idx % num_workers].append(c)
            idx += 1
            total_rows += len(c)
        st.write(f"CSV read in streaming mode; approx rows: {total_rows}")
        partitions = []
        for b in buffers:
            if len(b) > 0:
                partitions.append(pd.concat(b, ignore_index=True))
            else:
                partitions.append(pd.DataFrame(columns=['transcript']))
    else:
        st.error("Unsupported file type. Please upload CSV or Parquet.")
        st.stop()

    for p in partitions:
        if 'transcript' not in p.columns:
            st.error("Each partition must contain a column named 'transcript'. Please check your file.")
            st.stop()

    # Setup checkpoint folder
    output_folder = os.path.join(temp_dir, "checkpoints")
    os.makedirs(output_folder, exist_ok=True)

    st.info("Processing partitions with worker processes. ONNX will be used if a quantized model exists at 'onnx_model/model_quant.onnx'.")
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = [None] * len(partitions)
    start_time = time.time()

    # Check for existing checkpoints and schedule processing
    to_process = []
    for i, part in enumerate(partitions):
        chk_path = os.path.join(output_folder, f"checkpoint_part_{i}.parquet")
        if os.path.exists(chk_path):
            try:
                results[i] = pd.read_parquet(chk_path)
                st.info(f"Loaded checkpoint for partition {i}")
            except Exception:
                results[i] = None
                to_process.append((i, part))
        else:
            to_process.append((i, part))

    total_jobs = len(to_process)
    completed = 0

    if total_jobs > 0:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            future_map = {executor.submit(process_dataframe_partition, part, model_hint, int(batch_size), int(chunk_words), idx, output_folder): idx for idx, part in to_process}

            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    res = fut.result()
                    results[idx] = res
                except Exception as e:
                    st.error(f"Worker failed for partition {idx}: {e}")
                    results[idx] = pd.DataFrame()
                completed += 1
                progress_bar.progress((completed) / total_jobs)
                status_text.text(f"Processed {completed}/{total_jobs} new partitions")

    # Combine
    final_parts = [r if r is not None else pd.DataFrame() for r in results]
    final_df = pd.concat(final_parts, ignore_index=True)
    elapsed = time.time() - start_time
    st.success(f"Processing complete in {elapsed:.1f} seconds. Total rows: {len(final_df)}")

    st.subheader("Preview")
    st.dataframe(final_df.head(int(preview_rows)))

    output_path = os.path.join(temp_dir, "transcripts_with_sentiment.parquet")
    final_df.to_parquet(output_path, index=False)

    with open(output_path, "rb") as f:
        st.download_button(label="Download Result (Parquet)", data=f, file_name="transcripts_with_sentiment.parquet")

    st.info("Tip: For repeated runs, export & quantize ONNX once, then reuse the ONNX file to get much faster CPU inference.")
