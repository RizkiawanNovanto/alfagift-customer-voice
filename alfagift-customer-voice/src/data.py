"""Pemuatan data, model prediksi, dan fungsi analisis murni (tanpa UI)."""
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import NEG, ROOT, STOP
from src.issues import issue_for, normalize
from src.predictor import load_predictor


@st.cache_data(show_spinner="Memuat data review…")
def load_data(path_str):
    """Baca CSV, rapikan tipe kolom, tandai review negatif & kategorinya."""
    path = Path(path_str)
    df = pd.read_csv(path)
    if "at" in df.columns:
        df["at"] = pd.to_datetime(df["at"], errors="coerce")
    if "sentiment" not in df.columns and "y_true_final" in df.columns:
        df["sentiment"] = df["y_true_final"]
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    pred = next((c for c in ("indobert_pred", "sentiment") if c in df.columns), None)
    if "content" in df.columns:
        df["kategori"] = df["content"].map(issue_for)
    df["is_neg"] = df[pred].astype(str).str.lower().isin(NEG) if pred else False
    return df, path.name, pred


@st.cache_resource(show_spinner="Memuat model IndoBERT…")
def get_predictor():
    """Muat IndoBERT sekali per sesi server. Melempar error jika model belum tersedia."""
    return load_predictor(ROOT)


@st.cache_data(show_spinner="Menjalankan IndoBERT pada seluruh review. Proses pertama bisa memakan ±10 menit di server kecil…")
def live_labels(path_str):
    """Prediksi IndoBERT untuk setiap baris CSV (bukan membaca kolom hasil yang tersimpan)."""
    texts = pd.read_csv(path_str, usecols=["content"])["content"].fillna("").astype(str).tolist()
    return get_predictor().predict_batch(texts)


def apply_live_predictions(df, path_str):
    """Ganti kolom indobert_pred dengan hasil IndoBERT yang dihitung langsung dari teks review."""
    out = df.copy()
    if "indobert_pred" in out.columns:
        out["indobert_pred_csv"] = out["indobert_pred"]
    out["indobert_pred"] = live_labels(path_str)
    out["is_neg"] = out["indobert_pred"].astype(str).str.lower().isin(NEG)
    return out


@st.cache_data
def top_terms(texts, ngram, k):
    c = Counter()
    for t in texts:
        w = [x for x in normalize(t).split() if len(x) > 2 and x not in STOP and not x.isdigit()]
        c.update(w if ngram == 1 else [" ".join(p) for p in zip(w, w[1:])])
    return pd.DataFrame(c.most_common(k), columns=["kata", "jumlah"])


def issue_table(neg):
    """Jumlah & persentase review negatif per kategori (tanpa 'Lainnya')."""
    out = (neg[neg["kategori"] != "Lainnya"]["kategori"].value_counts()
           .rename_axis("kategori").reset_index(name="jumlah"))
    out["persentase"] = out["jumlah"] / max(len(neg), 1) * 100
    return out


def trends(neg, days=90):
    """Perubahan jumlah review negatif per kategori: 90 hari terakhir vs 90 hari sebelumnya."""
    if neg.empty or "at" not in neg.columns or neg["at"].dropna().empty:
        return {}
    end = neg["at"].max()
    rec = neg[neg["at"] > end - pd.Timedelta(days=days)]["kategori"].value_counts()
    prev = neg[(neg["at"] <= end - pd.Timedelta(days=days))
               & (neg["at"] > end - pd.Timedelta(days=2 * days))]["kategori"].value_counts()
    return {k: (rec.get(k, 0) - prev[k]) / prev[k] * 100 for k in prev.index if prev[k] >= 5}


def monthly(frame):
    work = frame.dropna(subset=["at"]).copy()
    work["month"] = work["at"].dt.to_period("M").astype(str)
    return work
