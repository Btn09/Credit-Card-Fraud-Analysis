"""Fraud tespit servisi.

Çalıştırma:  uvicorn api.app:app --reload
Dokümantasyon: http://127.0.0.1:8000/docs
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from src.modeller import agac_modeli
from src.veri import KOK, ozellik_cikar

MODEL_KLASORU = Path(os.environ.get("MODEL_KLASORU", KOK / "models"))

# Girdi: Kaggle verisindeki ham sütunlar (Time, V1..V28, Amount)
Islem = create_model(
    "Islem",
    Time=(float, Field(ge=0, description="İlk işlemden beri geçen saniye")),
    Amount=(float, Field(ge=0, description="İşlem tutarı (€)")),
    **{f"V{i}": (float, ...) for i in range(1, 29)},
)


class Neden(BaseModel):
    ozellik: str
    etki: float


class Sonuc(BaseModel):
    olasilik: float
    supheli: bool
    esik: float
    nedenler: list[Neden]


durum = {}


@asynccontextmanager
async def lifespan(app):
    durum["model"] = joblib.load(MODEL_KLASORU / "model.joblib")
    durum["bilgi"] = json.loads((MODEL_KLASORU / "model_bilgisi.json").read_text(encoding="utf-8"))
    durum["aciklayici"] = shap.TreeExplainer(agac_modeli(durum["model"]))
    yield
    durum.clear()


app = FastAPI(title="Kredi Kartı Fraud Tespiti", lifespan=lifespan)


def degerlendir(islemler):
    df = pd.DataFrame([i.model_dump() for i in islemler])
    X = ozellik_cikar(df)
    olasiliklar = durum["model"].predict_proba(X)[:, 1]
    shap_degerleri = durum["aciklayici"](X).values
    esik = durum["bilgi"]["esik"]

    sonuclar = []
    for olasilik, satir in zip(olasiliklar, shap_degerleri):
        # Fraud yönünde en çok iten 3 özellik
        en_etkili = sorted(zip(X.columns, satir), key=lambda x: x[1], reverse=True)[:3]
        sonuclar.append(Sonuc(
            olasilik=round(float(olasilik), 4),
            supheli=bool(olasilik >= esik),
            esik=esik,
            nedenler=[Neden(ozellik=o, etki=round(float(e), 3)) for o, e in en_etkili],
        ))
    return sonuclar


@app.get("/saglik")
def saglik():
    return {"durum": "calisiyor", "model": durum["bilgi"]["model"], "esik": durum["bilgi"]["esik"]}


@app.post("/tahmin", response_model=Sonuc)
def tahmin(islem: Islem):
    return degerlendir([islem])[0]


@app.post("/toplu-tahmin", response_model=list[Sonuc])
def toplu_tahmin(islemler: list[Islem]):
    return degerlendir(islemler) if islemler else []
