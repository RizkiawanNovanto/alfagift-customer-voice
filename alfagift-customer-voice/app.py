"""Alfagift Customer Voice Analytics — entry point.

Alur: setup halaman -> sidebar (menu, filter) -> muat data -> render halaman terpilih.
Isi tiap halaman ada di src/views/, logika data di src/data.py.
"""
import pandas as pd
import streamlit as st

from src import components as ui
from src.config import DATA_FILES, ICON, MENU, RATINGS
from src.context import Ctx
from src.data import apply_live_predictions, get_predictor, load_data
from src.views import PAGES

st.set_page_config(page_title="Alfagift Customer Voice", page_icon=str(ICON) if ICON.exists() else "🎁",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

# ---------------------------------------------------------------- sidebar: kerangka (urutan tampilan)
ui.sidebar_brand()
nav_box = st.sidebar.container()
filter_box = st.sidebar.container()
info_box = st.sidebar.container()
adv_box = st.sidebar.container()

# ---------------------------------------------------------------- sumber data
available = [p for p in DATA_FILES if p.exists()]
chosen = available[0] if available else None
with adv_box.expander("⚙️ Pengaturan lanjutan"):
    if len(available) > 1:
        chosen = st.selectbox("Sumber data", available, format_func=lambda p: p.name,
                              help="Periode & jumlah review mengikuti file yang dipilih.")
    live = st.toggle("Hitung ulang dengan IndoBERT", value=False,
                     help="Menjalankan IndoBERT pada setiap review saat halaman dibuka. "
                          "Hasilnya sama dengan prediksi tersimpan, tetapi butuh ±10 menit di server kecil.")

df, source, prediction_col = load_data(str(chosen)) if chosen else (pd.DataFrame(), None, None)
try:
    get_predictor()  # IndoBERT wajib: berhenti dengan pesan jelas bila model belum ada
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
if chosen and prediction_col == "indobert_pred" and live:
    df = apply_live_predictions(df, str(chosen))
has_date = not df.empty and "at" in df.columns and df["at"].notna().any()
dmin, dmax = (df["at"].min().date(), df["at"].max().date()) if has_date else (None, None)
vcol = next((c for c in ("reviewCreatedVersion", "version") if c in df.columns), None)
can_eval = (not df.empty and prediction_col == "indobert_pred"
            and {"sentiment", "content"} <= set(df.columns))

# ---------------------------------------------------------------- sidebar: menu
menu = [m for m in MENU if m != "Performa Model" or can_eval]
if st.session_state.get("page") not in menu:
    st.session_state["page"] = menu[0]
page = nav_box.radio("Menu", menu, format_func=lambda x: f"{MENU[x]}  {x}",
                     label_visibility="collapsed", key="page")


# ---------------------------------------------------------------- sidebar: filter
def reset_filters():
    if has_date:
        st.session_state["f_date"] = (dmin, dmax)
    st.session_state["f_rating"] = RATINGS
    st.session_state["f_ver"] = []


fdf = df
if not df.empty:
    with filter_box.expander("🎛️ Filter data"):
        rng = st.date_input("Periode", (dmin, dmax), min_value=dmin, max_value=dmax, key="f_date") if has_date else None
        ratings = st.multiselect("Rating", RATINGS, default=RATINGS, key="f_rating") if "score" in df.columns else RATINGS
        versions = []
        if vcol:
            opts = df[vcol].dropna().astype(str).value_counts().index[:40].tolist()
            versions = st.multiselect("Versi aplikasi", opts, key="f_ver", placeholder="Semua versi")
        st.button("↺ Reset filter", on_click=reset_filters, width="stretch")
    if has_date and isinstance(rng, (tuple, list)) and len(rng) == 2 and (rng[0], rng[1]) != (dmin, dmax):
        fdf = fdf[(fdf["at"] >= pd.Timestamp(rng[0])) & (fdf["at"] < pd.Timestamp(rng[1]) + pd.Timedelta(days=1))]
    if "score" in df.columns and set(ratings) != set(RATINGS):
        fdf = fdf[fdf["score"].isin(ratings)]
    if versions:
        fdf = fdf[fdf[vcol].astype(str).isin(versions)]

with info_box:
    ui.sidebar_chip(f"📅 Periode data: <b>{dmin:%b %Y} – {dmax:%b %Y}</b>" if has_date else "📅 Periode data: <b>-</b>")
    ui.sidebar_chip("🤖 Model: <b>IndoBERT</b> • " + ("hitung ulang live" if live else "hasil tersimpan"))
    if source:
        st.caption(f"Sumber data: {source}")

# ---------------------------------------------------------------- konten
ctx = Ctx(df, fdf, source, prediction_col, chosen)
footer = (f"Alfagift Customer Voice • Dashboard Internal • {dmin.year}–{dmax.year}"
          if has_date else "Alfagift Customer Voice • Dashboard Internal")

if page == "Ringkasan":
    ui.hero("ALFAGIFT CUSTOMER VOICE", "Memahami Suara Pelanggan Alfagift",
            "Menerjemahkan ribuan ulasan menjadi gambaran pengalaman pengguna dan area layanan yang perlu diperhatikan.")

if not df.empty and page not in ("Tentang Alfagift", "Coba Prediksi"):
    ui.note(f"Menampilkan <b>{len(fdf):,}</b> dari <b>{len(df):,}</b> review"
            + (" • <b style='color:#E4002B'>filter aktif</b> (ubah di sidebar → Filter data)" if ctx.filtered else ""))
    if fdf.empty:
        ui.empty_state("Tidak ada review yang cocok dengan filter. Coba perluas periode/rating atau klik Reset filter.")
        ui.footer(footer)
        st.stop()

PAGES[page](ctx)
ui.footer(footer)
