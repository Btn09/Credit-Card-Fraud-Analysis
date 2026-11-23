"""Modelleri eğitir, karşılaştırır, en iyisini seçip models/ klasörüne kaydeder.

Çalıştırma:  python -m src.egit
"""
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve, recall_score)

from .degerlendir import (INCELEME_MALIYETI, bootstrap_araligi, en_iyi_esik,
                          maliyet, metrikler)
from .modeller import modelleri_hazirla
from .veri import KOK, OZELLIKLER, ozellik_cikar, veri_oku, zamana_gore_bol

MODEL_KLASORU = KOK / "models"
RAPOR_KLASORU = KOK / "reports"
GRAFIK_KLASORU = RAPOR_KLASORU / "figures"


def main():
    for k in (MODEL_KLASORU, GRAFIK_KLASORU):
        k.mkdir(parents=True, exist_ok=True)

    df = veri_oku()
    egitim, dogrulama, test = zamana_gore_bol(df)
    X_e, y_e = ozellik_cikar(egitim), egitim["Class"].values
    X_d, y_d = ozellik_cikar(dogrulama), dogrulama["Class"].values
    X_t, y_t = ozellik_cikar(test), test["Class"].values
    print(f"Eğitim {len(y_e)} ({y_e.sum()} fraud) | Doğrulama {len(y_d)} ({y_d.sum()}) | Test {len(y_t)} ({y_t.sum()})")

    pozitif_agirlik = (y_e == 0).sum() / y_e.sum()
    modeller = modelleri_hazirla(pozitif_agirlik)

    satirlar, olasiliklar = [], {}
    for ad, model in modeller.items():
        bas = time.time()
        model.fit(X_e, y_e)
        sure = time.time() - bas
        p_d = model.predict_proba(X_d)[:, 1]
        p_t = model.predict_proba(X_t)[:, 1]
        olasiliklar[ad] = (p_d, p_t)
        satirlar.append({
            "model": ad,
            "dogrulama_pr_auc": average_precision_score(y_d, p_d),
            "test_pr_auc": average_precision_score(y_t, p_t),
            "egitim_suresi_sn": round(sure, 1),
        })
        print(f"{ad:20s} doğrulama PR-AUC {satirlar[-1]['dogrulama_pr_auc']:.3f} | "
              f"test PR-AUC {satirlar[-1]['test_pr_auc']:.3f} | {sure:.0f} sn")

    tablo = pd.DataFrame(satirlar).sort_values("dogrulama_pr_auc", ascending=False)
    tablo.to_csv(RAPOR_KLASORU / "model_karsilastirma.csv", index=False, float_format="%.4f")

    # Seçim doğrulama setine göre yapılıyor, test seti sadece son raporda kullanılıyor
    secilen = tablo.iloc[0]["model"]
    model = modeller[secilen]
    p_d, p_t = olasiliklar[secilen]
    print(f"\nSeçilen model: {secilen}")

    esik, esikler, maliyetler = en_iyi_esik(y_d, p_d, dogrulama["Amount"].values)
    print(f"Doğrulama setinde en düşük maliyetli eşik: {esik}")

    test_metrik = metrikler(y_t, p_t, esik)
    tutar_t = test["Amount"].values
    modelsiz_kayip = tutar_t[y_t == 1].sum()
    modelli_maliyet = maliyet(y_t, p_t, tutar_t, esik)
    isaretli = int((p_t >= esik).sum())
    yakalanan_tutar = tutar_t[(y_t == 1) & (p_t >= esik)].sum()

    pr_aralik = bootstrap_araligi(y_t, p_t, average_precision_score)
    duy_aralik = bootstrap_araligi(y_t, p_t, lambda y, p: recall_score(y, (p >= esik).astype(int)))

    # Farklı inceleme maliyetlerinde eşik ve sonuç
    duyarlilik_analizi = []
    for inc in (1, 5, 10, 25):
        e, _, _ = en_iyi_esik(y_d, p_d, dogrulama["Amount"].values, inceleme=inc)
        duyarlilik_analizi.append({
            "inceleme_maliyeti": inc,
            "esik": e,
            "test_duyarlilik": float(recall_score(y_t, (p_t >= e).astype(int))),
            "test_isaretlenen": int((p_t >= e).sum()),
            "test_net_kazanc": float(modelsiz_kayip - maliyet(y_t, p_t, tutar_t, e, inc)),
        })

    sonuc = {
        "secilen_model": secilen,
        "esik": esik,
        "inceleme_maliyeti": INCELEME_MALIYETI,
        "test": {k: round(float(v), 4) for k, v in test_metrik.items()},
        "test_pr_auc_95_aralik": [round(x, 4) for x in pr_aralik],
        "test_duyarlilik_95_aralik": [round(x, 4) for x in duy_aralik],
        "test_fraud_sayisi": int(y_t.sum()),
        "test_isaretlenen_islem": isaretli,
        "test_fraud_toplam_tutar": round(float(modelsiz_kayip), 2),
        "test_yakalanan_tutar": round(float(yakalanan_tutar), 2),
        "test_modelli_maliyet": round(float(modelli_maliyet), 2),
        "test_net_kazanc": round(float(modelsiz_kayip - modelli_maliyet), 2),
        "inceleme_maliyeti_analizi": duyarlilik_analizi,
    }
    with open(RAPOR_KLASORU / "sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))

    joblib.dump(model, MODEL_KLASORU / "model.joblib", compress=3)
    with open(MODEL_KLASORU / "model_bilgisi.json", "w", encoding="utf-8") as f:
        json.dump({"model": secilen, "esik": esik, "ozellikler": OZELLIKLER,
                   "inceleme_maliyeti": INCELEME_MALIYETI}, f, ensure_ascii=False, indent=2)

    grafikler(y_t, olasiliklar, secilen, p_t, esik, esikler, maliyetler)


def grafikler(y_t, olasiliklar, secilen, p_t, esik, esikler, maliyetler):
    plt.figure(figsize=(7, 5))
    for ad, (_, p) in olasiliklar.items():
        kes, duy, _ = precision_recall_curve(y_t, p)
        plt.plot(duy, kes, lw=2.2 if ad == secilen else 1.2,
                 label=f"{ad} (AP={average_precision_score(y_t, p):.3f})")
    plt.axhline(y_t.mean(), color="gray", ls="--", lw=1, label="Rastgele tahmin")
    plt.xlabel("Duyarlılık (yakalanan fraud oranı)")
    plt.ylabel("Kesinlik")
    plt.title("Test seti: Kesinlik-Duyarlılık eğrileri")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAFIK_KLASORU / "pr_egrileri.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(esikler, maliyetler, color="k")
    plt.axvline(esik, color="tab:red", ls="--", label=f"seçilen eşik = {esik}")
    plt.xlabel("Karar eşiği")
    plt.ylabel("Toplam maliyet (€)")
    plt.title("Doğrulama seti: eşiğe göre maliyet\n(kaçan fraud tutarı + işaretlenen işlem × 5 €)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAFIK_KLASORU / "esik_maliyet.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_t, (p_t >= esik).astype(int), display_labels=["Normal", "Fraud"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"{secilen}, eşik {esik} (test)")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(GRAFIK_KLASORU / "karisiklik_matrisi.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
