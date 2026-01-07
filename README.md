# temrrut_risk_hesaplama
Bu proje, UCI German Credit Dataset verilerini kullanarak, bankacılık sektöründe kredi temerrüt riskini (default risk) tahmin etmek ve kredi onay süreçlerini optimize etmek amacıyla geliştirilmiştir.

# 💳 Kredi Risk Analizi ve Finansal Karar Destek Sistemi

Bu proje, bankacılık sektöründe kredi temerrüt riskini tahmin etmek ve kredi onay süreçlerini optimize etmek amacıyla geliştirilmiş uçtan uca bir makine öğrenmesi çözümüdür. Projenin temel odağı, sadece istatistiksel doğruluk değil, hatalı kararların finansal maliyetini minimize eden bir **İş Değeri (Business Value)** motoru oluşturmaktır.

## 📈 Proje Özeti
Sistem, UCI German Credit Dataset üzerindeki verileri kullanarak müşterilerin risk profilini analiz eder. Model seçimi ve optimizasyon süreci, bankanın uğrayabileceği maksimum zararı (False Negative) baskılayacak şekilde tasarlanmıştır.

## 🛠️ Teknik Mimari ve Model
Projede gürültüye karşı dayanıklılığı (robustness) kanıtlanmış olan **Random Forest** algoritması tercih edilmiştir. 

* **Model Parametreleri:** * `max_depth: 4` (Aşırı öğrenmeyi ve gürültüyü engellemek için sığ ağaç yapısı)
    * `min_samples_leaf: 11`
    * `n_estimators: 136`
* **Pipeline:** Veri ön işleme aşamasında `StandardScaler` ve `OneHotEncoder` otomatikleştirilmiş bir yapıdadır.
* **Strateji:** "Büyüme Odaklı" strateji ile 0.518 eşik değeri (threshold) belirlenmiştir.

## 📊 Performans Sonuçları

Modelin test verisi üzerindeki başarı metrikleri aşağıdadır:

| Metrik | Sonuç | Tanım |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.7829** | Ayırt etme gücü (Sektör standardı üstü) |
| **Bad Recall** | **%76.7** | Batacak kredileri önceden yakalama oranı |
| **Good Precision** | **%87.9** | Onaylanan kredilerin geri dönüş güvenilirliği |
| **Onay Oranı** | **%74.0** | Pazar payını koruma kapasitesi |

## 💰 İş Değeri ve Maliyet Analizi
Model, aşağıdaki maliyet fonksiyonu baz alınarak optimize edilmiştir:
* **Kötü Krediyi Engelleme (TP):** +1000 Birim (Zarar Önleme)
* **İyi Krediyi Onaylama (TN):** +200 Birim (Net Kâr)
* **İyi Krediyi Reddetme (FP):** -150 Birim (Fırsat Kaybı)
* **Kötü Krediyi Onaylama (FN):** -5000 Birim (Büyük Zarar)

## ⚖️ Fairness (Adalet) Analizi
Modelin etik standartlara uygunluğu denetlenmiştir. `Attribute9` ve `Attribute17` gibi hassas değişkenler üzerinde yapılan analizler, modelin karar verme süreçlerinde gruplar arası dengeyi ne ölçüde koruduğunu gösterir. Belirlenen %35.0 Disparate Impact skoru, canlı sistemlerde insan denetimli (Human-in-the-loop) bir mekanizmanın gerekliliğine işaret etmektedir.

## 🚀 Kurulum ve Kullanım

1. Depoyu yerel makinenize indirin:
   ```bash
   git clone [https://github.com/kullaniciadi/proje-adi.git](https://github.com/kullaniciadi/proje-adi.git)

   pip install pandas numpy scikit-learn matplotlib seaborn ucimlrepo
   python credit_risk_analysis.py
   Sonuç
Bu çalışma, veri biliminin finansal karar süreçlerine entegrasyonu için güvenilir bir prototip sunmaktadır. Model, sığ ağaç yapısı sayesinde yeni verilerde yüksek genelleme yeteneğine sahiptir ve banka karlılığını risk odaklı bir yaklaşımla korumaktadır.


---


