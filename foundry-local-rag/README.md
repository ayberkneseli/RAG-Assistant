# Building Your First Local RAG Application with Foundry Local

Bu proje, verilen yaz okulu planındaki implementasyon bölümünün çalışan karşılığıdır. Yerel belgeleri parçalara ayırır, Foundry Local ile embedding üretir, vektörleri SQLite'ta saklar, cosine similarity ile ilgili parçaları bulur ve yine Foundry Local üzerindeki yerel sohbet modeliyle kaynaklı cevap üretir.

Normal soru-cevap akışında bulut API'si, API anahtarı veya Azure hesabı kullanılmaz. Modellerin ilk kez indirilmesi için internet gerekir; modeller önbelleğe alındıktan ve Python paketleri kurulduktan sonra uygulama çevrimdışı çalışabilir.

## Neler hazır?

- `.md`, `.txt` ve metin içeren `.pdf` belgelerini okuma
- Tekrarlı, örtüşen metin parçaları (chunk) oluşturma
- `qwen3-embedding-0.6b` ile toplu embedding üretme
- Embeddingleri `BLOB` olarak SQLite'ta saklama
- Dosya SHA-256 değeriyle değişmeyen belgeleri tekrar işlememe
- Silinen belgeleri indeksten kaldırma ve `--rebuild` desteği
- Cosine similarity ile top-K retrieval
- Benzerlik eşiğinin altında LLM'i çağırmadan güvenli fallback
- Prompt içinde `[S1]`, `[S2]` biçiminde kaynak zorlaması
- CLI, etkileşimli terminal ve Streamlit arayüzü
- Retrieval değerlendirmesi ve Foundry gerektirmeyen birim testleri

## Mimari

```text
Belgeler -> Loader -> Chunker -> Foundry Embedding Model -> SQLite
                                                           |
Kullanıcı sorusu -> Query Embedding -> Cosine Search -------+
                                      |
                                      v
                     Kaynaklı context + soru
                                      |
                                      v
                         Foundry Chat Model -> Cevap
```

SQLite küçük eğitim veri kümeleri için bilinçli olarak kullanılmıştır. Arama sırasında vektörler SQLite'tan okunur ve benzerlik Python'da hesaplanır. Çok büyük belge koleksiyonlarında özel bir vector database veya SQLite vector extension tercih edilmelidir.

## Gereksinimler

- 64-bit Python 3.11-3.14
- Windows x64, Linux x64 veya Apple Silicon macOS
- İlk model indirmeleri için internet bağlantısı
- Modeller için yeterli boş disk alanı ve bellek

Foundry Local 2.0.1 doğrudan Python sürecinin içinde çalışır; ayrı Foundry CLI kurulumu gerekmez.

## Windows kurulumu (PowerShell)

```powershell
cd foundry-local-rag
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
cd foundry-local-rag
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Hızlı başlangıç

Projede Türkçe hazırlanmış beş örnek yaz okulu belgesi bulunur. Önce indeksi oluşturun:

```powershell
python main.py ingest --rebuild
```

İlk çalıştırmada embedding modeli indirilir. Sonra tek soru sorun:

```powershell
python main.py ask "Bir öğrenci en fazla kaç laboratuvar oturumunu kaçırabilir?"
```

Etkileşimli terminal:

```powershell
python main.py chat
```

Web arayüzü:

```powershell
streamlit run app.py
```

Tarayıcı açıldığında soldaki **Belgeleri İndeksle / Güncelle** düğmesiyle de indeks oluşturulabilir.

## Kendi belgelerinizi ekleme

1. `data/documents/` içine `.md`, `.txt` veya `.pdf` dosyaları koyun.
2. Aşağıdaki komutu çalıştırın:

```powershell
python main.py ingest
```

Değişmeyen dosyalar atlanır. Embedding modelini değiştirdiyseniz veya indeksi tamamen yenilemek istiyorsanız:

```powershell
python main.py ingest --rebuild
```

Farklı bir belge klasörü de verilebilir:

```powershell
python main.py ingest --documents "C:\Users\YourName\Documents\rag-docs" --rebuild
```

Not: Taranmış/görüntü tabanlı PDF'lerde OCR uygulanmaz. Bu dosyaları önce aranabilir PDF'ye dönüştürün.

## Ayarlar

Varsayılanlar ortam değişkenleriyle değiştirilebilir. PowerShell örneği:

```powershell
$env:RAG_CHAT_MODEL = "phi-3.5-mini"
$env:RAG_TOP_K = "4"
$env:RAG_MIN_SCORE = "0.30"
python main.py chat
```

| Değişken | Varsayılan | Açıklama |
|---|---:|---|
| `RAG_CHAT_MODEL` | `qwen2.5-0.5b` | Cevap üreten model alias'ı |
| `RAG_EMBEDDING_MODEL` | `qwen3-embedding-0.6b` | Belge ve soru embedding modeli |
| `RAG_TOP_K` | `1` | Prompt'a eklenen en ilgili chunk sayısı |
| `RAG_MIN_SCORE` | `0.35` | Cevap üretmek için minimum en iyi cosine skoru |
| `RAG_CHUNK_WORDS` | `180` | Bir chunk'ın yaklaşık kelime üst sınırı |
| `RAG_CHUNK_OVERLAP_WORDS` | `30` | Komşu chunk'ların örtüşmesi |
| `RAG_MAX_OUTPUT_TOKENS` | `96` | Yerel model cevap sınırı |

Eşik belge türüne ve dile göre ayarlanmalıdır. Çok fazla yanlış fallback oluyorsa eşiği biraz düşürün; ilgisiz içerikten cevap veriyorsa yükseltin.

## Test ve değerlendirme

Birim testleri gerçek model indirmeden çalışır:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Örnek soru setinde retrieval ölçümü:

```powershell
python main.py evaluate
```

Komut, beklenen kaynağın top-K sonuçlar içinde olup olmadığını (Hit@K) ve ortalama retrieval süresini raporlar.

## Proje yapısı

```text
foundry-local-rag/
├── app.py                         # Streamlit arayüzü
├── main.py                        # ingest/ask/chat/status/evaluate CLI
├── local_rag/
│   ├── chunking.py                # metin normalizasyonu ve chunking
│   ├── database.py                # SQLite şeması ve cosine search
│   ├── foundry_runtime.py         # Foundry Local 2.x Session adaptörü
│   ├── ingestion.py               # belge -> embedding -> SQLite akışı
│   ├── loaders.py                 # TXT/Markdown/PDF okuyucuları
│   ├── prompts.py                 # kaynaklı cevap talimatları
│   └── rag.py                     # retrieve + augment + generate
├── data/documents/                # örnek veya kullanıcı belgeleri
├── evaluation/cases.json          # retrieval değerlendirme soruları
└── tests/                         # model gerektirmeyen testler
```

## Beklenen ilk çalışma davranışı

- `ingest`: embedding modelini indirip yüklediği için ilk sefer uzun sürebilir.
- İlk `ask/chat`: sohbet modelini indirir; sonraki açılışlarda yerel cache'i kullanır.
- İki model aynı anda bellekte tutulur. Bellek sorunu yaşanırsa daha küçük bir chat modeli seçin veya CLI'ı yeniden başlatın.
- Uygulamayı kapatırken yüklenen modeller `unload()` ile serbest bırakılır.

## Sınırlamalar

- OCR, tabloya özel parsing ve görsel içerik analizi yoktur.
- SQLite araması küçük koleksiyonlar için brute-force yapılır.
- Küçük yerel modeller kaynak formatını bazen eksik uygulayabilir; UI her durumda retrieval kaynaklarını ayrıca gösterir.
- Bu örnek tek kullanıcı/tek cihaz senaryosu içindir; çok kullanıcılı inference sunucusu değildir.

## Temel kaynaklar

- Microsoft Foundry Local: https://github.com/microsoft/Foundry-Local
- Güncel SDK referansı: https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-current
- Foundry Local başlangıç rehberi: https://learn.microsoft.com/en-us/azure/foundry-local/get-started
