"""Kayıtlı modelin kararlarını SHAP ile açıklar, grafikleri reports/figures/ altına kaydeder.

Çalıştırma:  python -m src.acikla   (önce python -m src.egit)
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from .egit import GRAFIK_KLASORU, MODEL_KLASORU
from .modeller import agac_modeli
from .veri import ozellik_cikar, veri_oku, zamana_gore_bol


def main():
    model = joblib.load(MODEL_KLASORU / "model.joblib")
    bilgi = json.loads((MODEL_KLASORU / "model_bilgisi.json").read_text(encoding="utf-8"))

    _, _, test = zamana_gore_bol(veri_oku())
    X_t, y_t = ozellik_cikar(test), test["Class"].values
    p_t = model.predict_proba(X_t)[:, 1]

    aciklayici = shap.TreeExplainer(agac_modeli(model))

    # Bütün test setinde SHAP hesaplamak gereksiz yavaş; fraud'ların hepsi + 3000 normal işlem yeterli
    rng = np.random.default_rng(0)
    secim = np.concatenate([np.flatnonzero(y_t == 1),
                            rng.choice(np.flatnonzero(y_t == 0), 3000, replace=False)])
    X_ornek = X_t.iloc[secim]
    degerler = aciklayici(X_ornek)

    plt.figure()
    shap.plots.beeswarm(degerler, max_display=12, show=False)
    plt.title("Hangi özellik kararı ne yönde etkiliyor?")
    plt.tight_layout()
    plt.savefig(GRAFIK_KLASORU / "shap_ozet.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(degerler, max_display=12, show=False)
    plt.title("Ortalama etki büyüklüğü")
    plt.tight_layout()
    plt.savefig(GRAFIK_KLASORU / "shap_onem.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Tek bir işlem: modelin yakaladığı, olasılığı en yüksek fraud
    yakalanan = np.flatnonzero((y_t == 1) & (p_t >= bilgi["esik"]))
    i = yakalanan[np.argmax(p_t[yakalanan])]
    tek = aciklayici(X_t.iloc[[i]])
    plt.figure()
    shap.plots.waterfall(tek[0], max_display=10, show=False)
    plt.title(f"Örnek işlem: fraud olasılığı %{p_t[i] * 100:.1f}, tutar {test['Amount'].iloc[i]:.2f} €")
    plt.tight_layout()
    plt.savefig(GRAFIK_KLASORU / "shap_tek_islem.png", dpi=150, bbox_inches="tight")
    plt.close()

    ortalama = np.abs(degerler.values).mean(axis=0)
    sira = np.argsort(ortalama)[::-1][:5]
    print("En etkili 5 özellik:", [X_ornek.columns[j] for j in sira])


if __name__ == "__main__":
    main()
