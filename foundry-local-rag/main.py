from __future__ import annotations

import argparse
import atexit
import json
import time
from pathlib import Path

from local_rag import (
    FoundryLocalAI,
    RagService,
    Settings,
    VectorStore,
    ingest_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Foundry Local ile çevrimdışı doküman soru-cevap uygulaması"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Yerel belgeleri indeksle")
    ingest_parser.add_argument("--documents", type=Path, help="Belge klasörü")
    ingest_parser.add_argument(
        "--rebuild", action="store_true", help="Vektör indeksini yeniden oluştur"
    )

    ask_parser = subparsers.add_parser("ask", help="Tek bir soru sor")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int)
    ask_parser.add_argument("--min-score", type=float)

    chat_parser = subparsers.add_parser(
        "chat", help="Etkileşimli soru-cevap oturumu başlat"
    )
    chat_parser.add_argument("--top-k", type=int)
    chat_parser.add_argument("--min-score", type=float)

    subparsers.add_parser("status", help="Bilgi tabanı istatistiklerini göster")

    evaluation_parser = subparsers.add_parser(
        "evaluate", help="Bilgi getirme başarısını evaluation/cases.json ile ölç"
    )
    evaluation_parser.add_argument(
        "--cases", type=Path, default=Path("evaluation/cases.json")
    )
    evaluation_parser.add_argument("--top-k", type=int)
    return parser


def create_runtime(settings: Settings) -> FoundryLocalAI:
    def progress(stage: str, percent: float) -> None:
        print(f"\r{stage}: {percent:5.1f}%", end="", flush=True)
        if percent >= 100:
            print()

    runtime = FoundryLocalAI(
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        max_output_tokens=settings.max_output_tokens,
        progress_callback=progress,
    )
    atexit.register(runtime.close)
    return runtime


def print_sources(sources) -> None:
    if not sources:
        return
    print("\nKaynaklar:")
    for index, source in enumerate(sources, start=1):
        print(f"  [S{index}] {source.citation_label} - benzerlik {source.score:.3f}")


def run_evaluation(service: RagService, cases_path: Path, top_k: int) -> int:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    passed = 0
    timings: list[float] = []
    for case in cases:
        started = time.perf_counter()
        results = service.retrieve(case["question"], top_k=top_k)
        timings.append(time.perf_counter() - started)
        returned = {result.source_path for result in results}
        success = case["expected_source"] in returned
        passed += int(success)
        marker = "BAŞARILI" if success else "BAŞARISIZ"
        print(f"[{marker}] {case['question']}")
        print(f"       beklenen: {case['expected_source']}; sonuç: {sorted(returned)}")

    total = len(cases)
    average = sum(timings) / total if total else 0.0
    print(
        f"\nHit@{top_k}: {passed}/{total} ({passed / total:.1%})"
        if total
        else "Değerlendirme sorusu bulunamadı"
    )
    print(f"Ortalama bilgi getirme süresi: {average:.3f} sn")
    return 0 if passed == total else 1


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    if getattr(args, "documents", None):
        settings = Settings(**{**settings.__dict__, "documents_dir": args.documents})
    settings.validate()
    store = VectorStore(settings.database_path)

    if args.command == "status":
        stats = store.stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print(
            f"Embedding modeli: "
            f"{store.get_metadata('embedding_model') or 'ayarlanmamış'}"
        )
        return 0

    runtime = create_runtime(settings)

    if args.command == "ingest":
        report = ingest_documents(
            store,
            runtime,
            settings,
            rebuild=args.rebuild,
            status_callback=print,
        )
        print(
            f"Tamamlandı: {report.files_indexed} dosya indekslendi, "
            f"{report.files_skipped} değişmeyen dosya atlandı, "
            f"{report.files_removed} dosya kaldırıldı, "
            f"{report.chunks_written} metin parçası kaydedildi."
        )
        return 0

    service = RagService(
        store,
        runtime,
        runtime,
        top_k=settings.top_k,
        min_score=settings.min_score,
    )

    if args.command == "ask":
        result = service.answer(args.question, args.top_k, args.min_score)
        print(f"\n{result.answer}")
        print_sources(result.sources)
        return 0

    if args.command == "chat":
        print("Yerel RAG Asistanı. Çıkmak için /exit yazın.")
        while True:
            question = input("\nSiz: ").strip()
            if question.lower() in {"/exit", "/quit"}:
                break
            if not question:
                continue
            result = service.answer(question, args.top_k, args.min_score)
            print(f"\nAsistan: {result.answer}")
            print_sources(result.sources)
        return 0

    if args.command == "evaluate":
        return run_evaluation(service, args.cases, args.top_k or settings.top_k)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
