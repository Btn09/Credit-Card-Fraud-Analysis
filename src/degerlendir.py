import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# Şüpheli bulunan her işlemin bir analist tarafından incelenmesinin maliyeti (euro).
# Veri setindeki tutarlar euro. 5 euro varsayım, README'de farklı değerler için sonuç var.
INCELEME_MALIYETI = 5.0


def metrikler(y, olasilik, esik):
    tahmin = (olasilik >= esik).astype(int)
    return {
        "pr_auc": average_precision_score(y, olasilik),
        "roc_auc": roc_auc_score(y, olasilik),
        "kesinlik": precision_score(y, tahmin, zero_division=0),
        "duyarlilik": recall_score(y, tahmin),
        "f1": f1_score(y, tahmin),
    }


def maliyet(y, olasilik, tutar, esik, inceleme=INCELEME_MALIYETI):
    # Kaçırılan fraud'un tutarı kayıp, işaretlenen her işlem için de inceleme maliyeti ödeniyor
    y = np.asarray(y)
    tutar = np.asarray(tutar)
    isaretli = olasilik >= esik
    kacan = (y == 1) & ~isaretli
    return tutar[kacan].sum() + isaretli.sum() * inceleme


def en_iyi_esik(y, olasilik, tutar, inceleme=INCELEME_MALIYETI):
    esikler = np.round(np.arange(0.01, 1.0, 0.01), 2)
    maliyetler = np.array([maliyet(y, olasilik, tutar, e, inceleme) for e in esikler])
    return float(esikler[np.argmin(maliyetler)]), esikler, maliyetler


def bootstrap_araligi(y, olasilik, fonksiyon, n=1000, seed=42):
    # Test setinden yerine koyarak örneklem çekip metriğin %95 güven aralığını buluyorum
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    skorlar = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        skorlar.append(fonksiyon(y[i], olasilik[i]))
    return float(np.percentile(skorlar, 2.5)), float(np.percentile(skorlar, 97.5))
