from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def xgb(**ek):
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
        **ek,
    )


def modelleri_hazirla(pozitif_agirlik):
    # pozitif_agirlik: normal işlem sayısı / fraud sayısı.
    # Dengesizlik için iki yolu karşılaştırıyorum: sınıf ağırlığı ve SMOTE ile çoğaltma.
    return {
        "Lojistik Regresyon": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": xgb(scale_pos_weight=pozitif_agirlik),
        "XGBoost + SMOTE": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
            ("model", xgb()),
        ]),
        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=pozitif_agirlik,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
    }


def agac_modeli(model):
    # SHAP için pipeline içindeki asıl modeli çıkar
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model
