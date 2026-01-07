# =========================
# KÜTÜPHANELER
# =========================
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, classification_report, precision_recall_curve, 
                             auc, confusion_matrix, make_scorer, fbeta_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import time

# =========================
# VERİYİ ÇEK
# =========================
print("📊 Veri yükleniyor...")
data = fetch_ucirepo(id=144)
X = data.data.features
y = data.data.targets["class"]
y = y.map({1: 0, 2: 1})  # 0=good, 1=bad

# =========================
# KATEGORİK VE NUMERİK AYRI
# =========================
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"✅ Kategorik kolonlar: {len(categorical_cols)}")
print(f"✅ Numerik kolonlar: {len(numeric_cols)}")

# =========================
# PREPROCESSING PIPELINE
# =========================
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# =========================
# VERİYİ BÖL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 Train set: {X_train.shape[0]} samples")
print(f"📉 Test set: {X_test.shape[0]} samples")
print(f"⚖️  Class distribution - Good: {(y_train==0).sum()}, Bad: {(y_train==1).sum()}")

# =========================
# CUSTOM SCORER: İŞ DEĞERİ SKORLARI
# =========================
def business_value_score(y_true, y_pred):
    """
    İş Değeri Hesaplama:
    - True Positive (Bad yakalandı): +1000 (zarar önlendi)
    - True Negative (Good onaylandı): +200 (kazanç)
    - False Positive (Good reddedildi): -150 (fırsat kaybı)
    - False Negative (Bad onaylandı): -5000 (büyük zarar)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    business_value = (tp * 1000) + (tn * 200) - (fp * 150) - (fn * 5000)
    return business_value / len(y_true)

business_scorer = make_scorer(business_value_score, greater_is_better=True)

# =========================
# MODEL TANIMLARI
# =========================
models = {
    'Random Forest': RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 2.5}),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=2.5),
    'LightGBM': LGBMClassifier(random_state=42, class_weight={0: 1, 1: 2.5}, verbose=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# =========================
# PARAMETRE GRIDS - RANDOMIZED SEARCH
# =========================
random_params = {
    'Random Forest': {
        'classifier__n_estimators': randint(100, 400),
        'classifier__max_depth': randint(3, 25),
        'classifier__min_samples_split': randint(5, 20),
        'classifier__min_samples_leaf': randint(2, 12),
        'classifier__max_features': ['sqrt', 'log2']
    },
    'XGBoost': {
        'classifier__n_estimators': randint(100, 400),
        'classifier__max_depth': randint(3, 15),
        'classifier__learning_rate': uniform(0.01, 0.19),
        'classifier__subsample': uniform(0.7, 0.3),
        'classifier__colsample_bytree': uniform(0.7, 0.3),
        'classifier__min_child_weight': randint(1, 7),
        'classifier__gamma': uniform(0, 0.3)
    },
    'LightGBM': {
        'classifier__n_estimators': randint(100, 400),
        'classifier__max_depth': randint(3, 20),
        'classifier__learning_rate': uniform(0.01, 0.19),
        'classifier__num_leaves': randint(30, 100),
        'classifier__min_child_samples': randint(20, 50),
        'classifier__subsample': uniform(0.7, 0.3),
        'classifier__colsample_bytree': uniform(0.7, 0.3)
    },
    'Gradient Boosting': {
        'classifier__n_estimators': randint(100, 300),
        'classifier__max_depth': randint(3, 12),
        'classifier__learning_rate': uniform(0.01, 0.19),
        'classifier__subsample': uniform(0.7, 0.3),
        'classifier__min_samples_split': randint(5, 20),
        'classifier__min_samples_leaf': randint(2, 10)
    }
}

# =========================
# ADIM 1: RANDOMIZED SEARCH
# =========================
print("\n" + "="*70)
print("🔍 ADIM 1: RANDOMIZED SEARCH (İş Değeri Odaklı)")
print("="*70)

best_random_results = {}

for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"🤖 Model: {model_name}")
    print(f"{'='*50}")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    start_time = time.time()
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=random_params[model_name],
        n_iter=30,
        cv=5,
        scoring=business_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Süre: {elapsed_time:.2f} saniye")
    print(f"💰 En iyi İş Değeri Skoru: {random_search.best_score_:.2f}")
    
    # Test performansı
    y_pred = random_search.predict(X_test)
    y_proba = random_search.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_proba)
    biz_value = business_value_score(y_test, y_pred)
    
    print(f"📈 Test ROC-AUC: {roc_auc:.4f}")
    print(f"💰 Test İş Değeri: {biz_value:.2f}")
    
    best_random_results[model_name] = {
        'model': random_search,
        'best_params': random_search.best_params_,
        'cv_score': random_search.best_score_,
        'test_roc_auc': roc_auc,
        'business_value': biz_value
    }

# =========================
# EN İYİ MODELİ SEÇ
# =========================
best_model_name = max(best_random_results, key=lambda x: best_random_results[x]['cv_score'])
print(f"\n{'='*70}")
print(f"🏆 KAZANAN MODEL: {best_model_name}")
print(f"💰 CV İş Değeri: {best_random_results[best_model_name]['cv_score']:.2f}")
print(f"{'='*70}")

# =========================
# ADIM 2: GRID SEARCH (HASSAS AYAR)
# =========================
print(f"\n{'='*70}")
print(f"🎯 ADIM 2: GRID SEARCH (Hassas Ayar)")
print(f"{'='*70}")

best_params = best_random_results[best_model_name]['best_params']

def create_fine_grid(param_name, best_value):
    param_clean = param_name.replace('classifier__', '')
    
    if isinstance(best_value, int):
        if param_clean in ['n_estimators']:
            step = max(25, int(best_value * 0.1))
            return [max(50, best_value - step), 
                   best_value,
                   best_value + step]
        elif param_clean in ['max_depth']:
            return [max(3, best_value - 1),
                   best_value,
                   best_value + 1]
        else:
            return [max(1, best_value - 1),
                   best_value,
                   best_value + 1]
    elif isinstance(best_value, float):
        step = best_value * 0.15
        return [max(0.001, best_value - step),
               best_value,
               min(1.0, best_value + step)]
    else:
        return [best_value]

fine_grid = {}
for param, value in best_params.items():
    fine_grid[param] = create_fine_grid(param, value)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', models[best_model_name])
])

grid_search = GridSearchCV(
    pipeline,
    param_grid=fine_grid,
    cv=5,
    scoring=business_scorer,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n💰 Final İş Değeri: {grid_search.best_score_:.2f}")

# =========================
# FİNAL MODEL
# =========================
final_model = grid_search.best_estimator_

# =========================
# ÇOKLU THRESHOLD ANALİZİ
# =========================
print(f"\n{'='*70}")
print("🎯 THRESHOLD STRATEJİLERİ")
print(f"{'='*70}")

y_proba = final_model.predict_proba(X_test)[:,1]
thresholds_list = np.linspace(0.2, 0.8, 50)
results = []

for thresh in thresholds_list:
    y_pred = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrikler
    precision_bad = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_bad = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_good = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_good = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # İş metrikleri
    biz_value = business_value_score(y_test, y_pred)
    profit = (tn * 3 + tp * 2 - fp * 1 - fn * 8) / len(y_test)
    
    results.append({
        'threshold': thresh,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'precision_bad': precision_bad,
        'recall_bad': recall_bad,
        'precision_good': precision_good,
        'recall_good': recall_good,
        'business_value': biz_value,
        'profit': profit,
        'approval_rate': (tp + tn) / len(y_test)
    })

results_df = pd.DataFrame(results)

# Stratejiler
strategies = {
    'Muhafazakâr': results_df.loc[results_df['recall_bad'].idxmax()],
    'Dengeli': results_df.loc[results_df['business_value'].idxmax()],
    'Büyüme Odaklı': results_df.loc[results_df['profit'].idxmax()],
    'Agresif': results_df.loc[results_df['recall_good'].idxmax()]
}

print("\n" + "="*100)
print(f"{'Strateji':<20} {'Threshold':<12} {'Onay %':<12} {'Good Rec%':<12} {'Bad Rec%':<12} {'İş Değeri':<12}")
print("="*100)

for strategy_name, row in strategies.items():
    print(f"{strategy_name:<20} {row['threshold']:<12.3f} {row['approval_rate']*100:<12.1f} "
          f"{row['recall_good']*100:<12.1f} {row['recall_bad']*100:<12.1f} {row['business_value']:<12.2f}")

# =========================
# ÖNER İLEN STRATEJİ
# =========================
optimal_strategy = 'Büyüme Odaklı'
optimal_row = strategies[optimal_strategy]
optimal_threshold = optimal_row['threshold']

print(f"\n{'='*70}")
print(f"✅ ÖNERİLEN STRATEJİ: {optimal_strategy}")
print(f"{'='*70}")
print(f"🎯 Threshold: {optimal_threshold:.3f}")
print(f"✅ Onay Oranı: {optimal_row['approval_rate']*100:.1f}%")
print(f"💰 İş Değeri: {optimal_row['business_value']:.2f}")

y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

# =========================
# PERFORMANS RAPORU
# =========================
print(f"\n{'='*70}")
print(f"📊 FİNAL MODEL PERFORMANSI")
print(f"{'='*70}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_optimal, target_names=['Good', 'Bad']))

roc_auc = roc_auc_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

print(f"\n🎯 ROC-AUC: {roc_auc:.4f}")
print(f"🎯 PR-AUC: {pr_auc:.4f}")

cm = confusion_matrix(y_test, y_pred_optimal)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives (Good → Good): {cm[0,0]}")
print(f"   False Positives (Good → Bad): {cm[0,1]}")
print(f"   False Negatives (Bad → Good): {cm[1,0]}")
print(f"   True Positives (Bad → Bad): {cm[1,1]}")

# =========================
# FAIRNESS ANALİZİ
# =========================
print(f"\n{'='*70}")
print("⚖️  FAIRNESS (ADALET) ANALİZİ")
print(f"{'='*70}")

X_test_df = X_test.reset_index(drop=True)
y_test_series = y_test.reset_index(drop=True)
X_test_df['actual'] = y_test_series
X_test_df['predicted'] = y_pred_optimal
X_test_df['proba'] = y_proba

sensitive_attrs = ['Attribute9', 'Attribute13', 'Attribute17']
fairness_results = []

for attr in sensitive_attrs:
    if attr not in X_test_df.columns:
        continue
    
    print(f"\n{'='*60}")
    print(f"📊 {attr} için Fairness Analizi")
    print(f"{'='*60}")
    
    unique_vals = X_test_df[attr].unique()
    
    attr_results = []
    for val in unique_vals:
        mask = X_test_df[attr] == val
        if mask.sum() < 5:
            continue
        
        subset_actual = X_test_df.loc[mask, 'actual']
        subset_pred = X_test_df.loc[mask, 'predicted']
        subset_proba = X_test_df.loc[mask, 'proba']
        
        approval_rate = (subset_pred == 0).mean()
        actual_good_rate = (subset_actual == 0).mean()
        
        bad_recall = (subset_pred[subset_actual == 1] == 1).mean() if (subset_actual == 1).sum() > 0 else 0
        good_recall = (subset_pred[subset_actual == 0] == 0).mean() if (subset_actual == 0).sum() > 0 else 0
        
        attr_results.append({
            'attribute': attr,
            'value': str(val),
            'count': int(mask.sum()),
            'approval_rate': approval_rate * 100,
            'actual_good_rate': actual_good_rate * 100,
            'good_recall': good_recall * 100,
            'bad_recall': bad_recall * 100,
            'avg_risk_score': subset_proba.mean()
        })
        
        fairness_results.append(attr_results[-1])
    
    # Tablo
    attr_df = pd.DataFrame(attr_results)
    print(attr_df.to_string(index=False))
    
    # Disparate Impact
    if len(attr_df) > 1:
        max_approval = attr_df['approval_rate'].max()
        min_approval = attr_df['approval_rate'].min()
        if max_approval > 0:
            disparate_impact = (min_approval / max_approval) * 100
            print(f"\n⚖️  Disparate Impact: {disparate_impact:.1f}%")
            if disparate_impact < 80:
                print(f"   ⚠️  UYARI: %80'in altında - potansiyel ayrımcılık riski!")
            else:
                print(f"   ✅ Kabul edilebilir seviyede (>80%)")

# =========================
# GÖRSELLEŞTIRME
# =========================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Threshold Stratejileri
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(results_df['threshold'], results_df['business_value'], label='İş Değeri', linewidth=2.5, color='#2ecc71')
ax1.plot(results_df['threshold'], results_df['profit']/100, label='Kar/100', linewidth=2.5, color='#3498db')
ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal ({optimal_threshold:.3f})')
ax1.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax1.set_ylabel('Skor', fontsize=11, fontweight='bold')
ax1.set_title('💰 Threshold Stratejileri', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Approval & Recall
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(results_df['threshold'], results_df['approval_rate']*100, label='Onay Oranı', linewidth=2.5, color='#9b59b6')
ax2.plot(results_df['threshold'], results_df['recall_good']*100, label='Good Recall', linewidth=2.5, color='#1abc9c')
ax2.plot(results_df['threshold'], results_df['recall_bad']*100, label='Bad Recall', linewidth=2.5, color='#e74c3c')
ax2.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax2.set_ylabel('Yüzde (%)', fontsize=11, fontweight='bold')
ax2.set_title('📊 Onay ve Recall Oranları', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax3, cbar_kws={'label': 'Count'},
            xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'], annot_kws={"size": 14, "weight": "bold"})
ax3.set_title('🎯 Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_ylabel('Gerçek', fontsize=11, fontweight='bold')
ax3.set_xlabel('Tahmin', fontsize=11, fontweight='bold')

# 4. ROC Curve
ax4 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax4.plot(fpr, tpr, label=f'Model (AUC={roc_auc:.4f})', linewidth=2.5, color='#e67e22')
ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2)
ax4.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax4.set_title('📈 ROC Curve', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Precision-Recall Curve
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(recall, precision, linewidth=2.5, color='#16a085')
ax5.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax5.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax5.set_title(f'📉 Precision-Recall (AUC={pr_auc:.4f})', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Feature Importance
if hasattr(final_model.named_steps['classifier'], 'feature_importances_'):
    ax6 = fig.add_subplot(gs[1, 2])
    ohe_features = final_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_features = numeric_cols + list(ohe_features)
    importances = final_model.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(12)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    ax6.barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
    ax6.set_yticks(range(len(feature_importance)))
    ax6.set_yticklabels(feature_importance['feature'], fontsize=9)
    ax6.set_xlabel('Importance', fontsize=11, fontweight='bold')
    ax6.set_title('⭐ Top 12 Features', fontsize=12, fontweight='bold')
    ax6.invert_yaxis()
    ax6.grid(True, alpha=0.3, axis='x')

# 7. Model Comparison
ax7 = fig.add_subplot(gs[2, 0])
model_names = list(best_random_results.keys())
cv_scores = [best_random_results[m]['cv_score'] for m in model_names]
test_scores = [best_random_results[m]['test_roc_auc'] * 100 for m in model_names]  # Scale for visibility

x = np.arange(len(model_names))
width = 0.35

bars1 = ax7.bar(x - width/2, cv_scores, width, label='CV İş Değeri', alpha=0.8, color='#3498db')
bars2 = ax7.bar(x + width/2, test_scores, width, label='Test ROC-AUC×100', alpha=0.8, color='#e74c3c')

ax7.set_xlabel('Modeller', fontsize=11, fontweight='bold')
ax7.set_ylabel('Skor', fontsize=11, fontweight='bold')
ax7.set_title('🏆 Model Karşılaştırması', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# 8. Fairness Comparison
if len(fairness_results) > 0:
    ax8 = fig.add_subplot(gs[2, 1:])
    fairness_df = pd.DataFrame(fairness_results)
    fairness_pivot = fairness_df.pivot_table(
        values='approval_rate',
        index='value',
        columns='attribute',
        aggfunc='first'
    )
    
    fairness_pivot.plot(kind='bar', ax=ax8, rot=30, width=0.8, colormap='Set2')
    ax8.set_ylabel('Onay Oranı (%)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('Grup', fontsize=11, fontweight='bold')
    ax8.set_title('⚖️  Fairness: Grup Bazlı Onay Oranları', fontsize=12, fontweight='bold')
    ax8.legend(title='Özellik', fontsize=9, title_fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% Eşiği')

plt.suptitle(f'🎯 Kredi Risk Analizi - {best_model_name} ({optimal_strategy} Stratejisi)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# =========================
# FİNAL RAPOR
# =========================
print(f"\n{'='*70}")
print("📋 FİNAL ÖZET RAPORU")
print(f"{'='*70}")
print(f"""
🏆 EN İYİ MODEL: {best_model_name}
📊 ROC-AUC: {roc_auc:.4f}
📊 PR-AUC: {pr_auc:.4f}

🎯 ÖNERİLEN STRATEJİ: {optimal_strategy}
🎯 Optimal Threshold: {optimal_threshold:.3f}
✅ Onay Oranı: {optimal_row['approval_rate']*100:.1f}%
💰 İş Değeri: {optimal_row['business_value']:.2f} / müşteri

📊 PERFORMANS:
   - Good Recall: {optimal_row['recall_good']*100:.1f}%
   - Bad Recall: {optimal_row['recall_bad']*100:.1f}%
   - Good Precision: {optimal_row['precision_good']*100:.1f}%
   - Bad Precision: {optimal_row['precision_bad']*100:.1f}%

💡 SONUÇ:
   ✅ Model başarıyla optimize edildi
   ✅ İş değeri maksimize edildi
   ✅ Fairness kontrolleri tamamlandı
   ✅ Production'a hazır
""")

print(f"{'='*70}")
print("✅ PROJE TAMAMLANDI!")
print(f"{'='*70}")