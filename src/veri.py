from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
VERI_DOSYASI = KOK / "data" / "creditcard.csv"

V_SUTUNLARI = [f"V{i}" for i in range(1, 29)]
OZELLIKLER = V_SUTUNLARI + ["log_tutar", "saat_sin", "saat_cos"]


def veri_oku(yol=VERI_DOSYASI):
    yol = Path(yol)
    if not yol.exists():
        raise FileNotFoundError(
            f"{yol} bulunamadı. Kaggle'dan creditcard.csv dosyasını indirip data/ klasörüne koyun."
        )
    return pd.read_csv(yol)


def ozellik_cikar(df):
    # Time ilk işlemden beri geçen saniye. Veri iki gün sürüyor, başlangıcı gece 00:00 kabul edip
    # günün saatine çeviriyorum. Saat döngüsel olduğu için sin/cos ile kodluyorum (23 ile 0 yan yana olsun).
    saat = (df["Time"] % 86400) / 3600
    X = df[V_SUTUNLARI].copy()
    X["log_tutar"] = np.log1p(df["Amount"])
    X["saat_sin"] = np.sin(2 * np.pi * saat / 24)
    X["saat_cos"] = np.cos(2 * np.pi * saat / 24)
    return X[OZELLIKLER]


def zamana_gore_bol(df, egitim=0.6, dogrulama=0.2):
    # Rastgele bölmek yerine zamana göre bölüyorum: model geçmiş işlemlerle eğitiliyor,
    # sonraki işlemlerde test ediliyor. Gerçek kullanımda da böyle olacak.
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    a, b = int(n * egitim), int(n * (egitim + dogrulama))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]
