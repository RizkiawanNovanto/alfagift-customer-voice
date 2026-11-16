"""Registry halaman: nama menu -> fungsi render(ctx)."""
from src.views import (about, insights, keywords, model, overview, predict,
                       problems, reviews, sentiment, trend, versions)

PAGES = {
    "Ringkasan": overview.render,
    "Coba Prediksi": predict.render,
    "Sentimen": sentiment.render,
    "Masalah Pengguna": problems.render,
    "Wawasan": insights.render,
    "Tren": trend.render,
    "Kata Kunci": keywords.render,
    "Versi Aplikasi": versions.render,
    "Review": reviews.render,
    "Performa Model": model.render,
    "Tentang Alfagift": about.render,
}
