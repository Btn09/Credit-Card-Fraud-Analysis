import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app

ORNEKLER = json.loads((Path(__file__).parent / "ornek_islemler.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def istemci():
    with TestClient(app) as c:
        yield c


def test_saglik(istemci):
    cevap = istemci.get("/saglik")
    assert cevap.status_code == 200
    assert cevap.json()["durum"] == "calisiyor"


def test_fraud_islem_supheli(istemci):
    cevap = istemci.post("/tahmin", json=ORNEKLER["fraud"]).json()
    assert cevap["supheli"] is True
    assert len(cevap["nedenler"]) == 3


def test_normal_islem_supheli_degil(istemci):
    cevap = istemci.post("/tahmin", json=ORNEKLER["normal"]).json()
    assert cevap["supheli"] is False
    assert cevap["olasilik"] < cevap["esik"]


def test_toplu_tahmin(istemci):
    cevap = istemci.post("/toplu-tahmin", json=[ORNEKLER["normal"], ORNEKLER["fraud"]])
    assert cevap.status_code == 200
    assert [s["supheli"] for s in cevap.json()] == [False, True]


def test_eksik_alan(istemci):
    islem = dict(ORNEKLER["normal"])
    del islem["V14"]
    assert istemci.post("/tahmin", json=islem).status_code == 422


def test_negatif_tutar(istemci):
    islem = dict(ORNEKLER["normal"], Amount=-10)
    assert istemci.post("/tahmin", json=islem).status_code == 422
