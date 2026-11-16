"""Prediksi sentimen memakai IndoBERT hasil fine-tuning (model WAJIB).

Model dimuat dari salah satu lokasi berikut, sesuai urutan:
1. Variabel lingkungan ALFAGIFT_MODEL_DIR (path folder lokal ATAU id model di Hugging Face Hub).
2. Folder lokal models/indobert (atau nama lain pada MODEL_DIR_NAMES) yang berisi config.json.
3. Hugging Face Hub: Rizkiawan/alfagift-indobert (subfolder indobert), diunduh otomatis.

Tidak ada fallback ke SVM atau aturan kata: prediksi selalu dari IndoBERT.
Urutan kelas harus 0 = Negatif, 1 = Netral, 2 = Positif (atau diisi pada id2label di config.json).
"""
import os
import re
from pathlib import Path

LABELS = ["Negatif", "Netral", "Positif"]
MODEL_DIR_NAMES = ["indobert", "indobert_best", "indobert-sentiment", "indobert_sentiment"]
MAX_LENGTH = 128
HF_REPO = "Rizkiawan/alfagift-indobert"

SLANG = {"gk": "tidak", "gx": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "yg": "yang", "dgn": "dengan",
         "bgt": "banget", "aja": "saja", "udah": "sudah", "blm": "belum", "krn": "karena"}


def clean_text(text):
    """Pembersihan ringan yang sama dengan tahap preprocessing saat model dilatih (tanpa stemming)."""
    t = str(text).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(SLANG.get(w, w) for w in t.split())


def _as_dict(order, probs):
    d = {k: 0.0 for k in LABELS}
    for k, v in zip(order, probs):
        d[k] = float(v)
    return d


_MAP = {"negative": "Negatif", "neg": "Negatif", "negatif": "Negatif", "neutral": "Netral", "netral": "Netral",
        "positive": "Positif", "pos": "Positif", "positif": "Positif"}


def _norm_label(name, i):
    k = str(name).strip().lower()
    if k in _MAP:
        return _MAP[k]
    return LABELS[i] if i < len(LABELS) else str(name)


class MissingModelError(FileNotFoundError):
    """IndoBERT tidak ditemukan. Pesannya ditampilkan langsung ke pengguna aplikasi."""


# ---------------------------------------------------------------- penyiapan folder model
def _is_valid_dir(p):
    """Folder model valid: ada config, bobot asli (bukan pointer LFS), dan tokenizer.json."""
    p = Path(p)
    w = p / "model.safetensors"
    return (
        p.is_dir()
        and (p / "config.json").exists()
        and (p / "tokenizer.json").exists()
        and w.exists()
        and w.stat().st_size > 10_000_000
    )


def _find_valid_subdir(root):
    root = Path(root)
    if _is_valid_dir(root):
        return root
    for cfg in root.rglob("config.json"):
        if _is_valid_dir(cfg.parent):
            return cfg.parent
    return None


def _download_from_hub(repo_id):
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id)
    found = _find_valid_subdir(root)
    if found is None:
        raise MissingModelError(
            f"Repo Hugging Face `{repo_id}` sudah diunduh, tetapi tidak berisi model lengkap "
            "(config.json, tokenizer.json, model.safetensors).")
    return found


def _prepare_dir(path):
    """Kembalikan Path folder model yang siap dipakai; unduh dari Hugging Face bila perlu."""
    p = Path(str(path))
    if p.exists():
        found = _find_valid_subdir(p)
        if found is not None:
            return found
        # folder lokal ada tetapi tidak lengkap (mis. pointer LFS): pakai Hugging Face
        return _download_from_hub(HF_REPO)
    # bukan folder lokal: anggap sebagai id repo Hugging Face
    try:
        return _download_from_hub(str(path))
    except MissingModelError:
        raise
    except Exception:
        return _download_from_hub(HF_REPO)


class BertPredictor:
    kind = "bert"
    note = "Prediksi memakai model IndoBERT hasil fine-tuning."

    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
        self.torch = torch
        model_dir = _prepare_dir(path)
        self.tok = PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
            cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
            unk_token="[UNK]", mask_token="[MASK]",
            model_max_length=512,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        id2 = getattr(self.model.config, "id2label", {}) or {}
        n = self.model.config.num_labels
        self.order = [_norm_label(id2.get(i, f"LABEL_{i}"), i) for i in range(n)]
        self.name = "IndoBERT (fine-tuned)"
        self.source = str(model_dir)

    def predict(self, text):
        enc = self.tok(clean_text(text), truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc.pop("token_type_ids", None) if "token_type_ids" not in self.model.forward.__code__.co_varnames else None
        with self.torch.no_grad():
            logits = self.model(**enc).logits[0]
        p = self.torch.softmax(logits, dim=-1).cpu().numpy()
        d = _as_dict(self.order, p)
        return {"label": max(d, key=d.get), "probs": d}

    def predict_batch(self, texts, batch_size=32):
        """Label untuk banyak teks sekaligus (dipakai untuk menghitung ulang seluruh data review)."""
        labels = []
        for i in range(0, len(texts), batch_size):
            chunk = [clean_text(t) for t in texts[i:i + batch_size]]
            enc = self.tok(chunk, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            labels.extend(self.order[j] for j in logits.argmax(-1).tolist())
        return labels


def find_model_dir(root):
    """Cari lokasi IndoBERT. Kembalikan string path atau id Hub; None jika tidak ada."""
    env = os.environ.get("ALFAGIFT_MODEL_DIR", "").strip()
    if not env:  # Streamlit Community Cloud: nilai bisa diisi lewat menu Secrets
        try:
            import streamlit as st
            env = str(st.secrets.get("ALFAGIFT_MODEL_DIR", "")).strip()
        except Exception:
            env = ""
    if env:
        return env
    for n in MODEL_DIR_NAMES:
        c = Path(root) / "models" / n
        has_weights = (c / "model.safetensors").exists() or (c / "pytorch_model.bin").exists()
        if (c / "config.json").exists() and has_weights:
            return str(c)
    return None


def load_predictor(root):
    """Muat IndoBERT. Bila tidak ada lokasi lokal/env, otomatis unduh dari Hugging Face."""
    location = find_model_dir(root)
    if location is None:
        location = HF_REPO
    try:
        return BertPredictor(location)
    except MissingModelError:
        raise
    except OSError as e:
        raise MissingModelError(
            f"Model IndoBERT gagal dimuat dari `{location}`: {e}. "
            "Pastikan koneksi ke Hugging Face tersedia atau atur `ALFAGIFT_MODEL_DIR`.") from e


# ---------------------------------------------------------------- penilaian bintang vs teks
def rating_label(star):
    return "Negatif" if star <= 2 else ("Netral" if star == 3 else "Positif")


def assess(sent, star):
    """Gabungkan sentimen teks dan bintang menjadi kesimpulan singkat."""
    rl = rating_label(star)
    agree = sent == rl
    base = {"rating_label": rl, "agree": agree}
    if sent == "Negatif" and rl == "Negatif":
        return {**base, "urgency": "Tinggi", "tone": "red", "icon": "🚨", "headline": "Keluhan jelas: teks dan bintang sama-sama negatif",
                "explanation": "Pengguna kecewa dan menunjukkannya lewat tulisan maupun bintang. Sebaiknya segera ditindaklanjuti."}
    if sent == "Negatif" and rl == "Positif":
        return {**base, "urgency": "Sedang", "tone": "yellow", "icon": "🕵️", "headline": "Keluhan tersembunyi: bintang tinggi, tetapi teks mengeluh",
                "explanation": "Bintang tinggi bisa menutupi masalah yang sebenarnya. Label sentimen pada data latih berasal dari bintang, sehingga kasus seperti ini mudah terlewat. Tetap telaah isi teksnya."}
    if sent == "Negatif":
        return {**base, "urgency": "Sedang", "tone": "yellow", "icon": "⚠️", "headline": "Teks mengeluh, bintang netral",
                "explanation": "Ada keluhan pada teks meski bintang tidak terlalu rendah. Layak ditindaklanjuti sebelum berubah menjadi bintang rendah."}
    if sent == "Positif" and rl == "Negatif":
        return {**base, "urgency": "Rendah", "tone": "yellow", "icon": "🤔", "headline": "Teks positif, tetapi bintang rendah",
                "explanation": "Kemungkinan pengguna keliru memberi bintang. Tidak perlu perbaikan layanan, tetapi bisa diverifikasi."}
    if sent == "Positif":
        return {**base, "urgency": "Rendah", "tone": "green", "icon": "💚", "headline": "Pengalaman positif",
                "explanation": "Teks dan bintang menunjukkan pengguna puas. Pertahankan hal yang membuat mereka puas."}
    if rl == "Negatif":
        return {**base, "urgency": "Sedang", "tone": "yellow", "icon": "❓", "headline": "Bintang rendah, tetapi teks tidak jelas mengeluh",
                "explanation": "Bintang rendah belum disertai alasan yang jelas pada teks. Minta pengguna menjelaskan kendalanya."}
    return {**base, "urgency": "Rendah", "tone": "yellow" if rl == "Netral" else "green", "icon": "😐", "headline": "Sentimen netral",
            "explanation": "Belum ada keluhan atau pujian yang menonjol pada ulasan ini."}
