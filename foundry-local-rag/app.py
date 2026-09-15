from __future__ import annotations

import streamlit as st

from local_rag import (
    FoundryLocalAI,
    RagService,
    Settings,
    VectorStore,
    ingest_documents,
)

st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="📚", layout="centered")
st.title("📚 Yerel Bilgi Asistanı")
st.caption(
    "Foundry Local ve SQLite ile cihazınızda çalışan, kaynaklı doküman soru-cevap uygulaması"
)

settings = Settings.from_env()
store = VectorStore(settings.database_path)


@st.cache_resource
def get_runtime(chat_model: str, embedding_model: str, max_tokens: int):
    return FoundryLocalAI(chat_model, embedding_model, max_tokens)


runtime = get_runtime(
    settings.chat_model,
    settings.embedding_model,
    settings.max_output_tokens,
)
service = RagService(
    store,
    runtime,
    runtime,
    top_k=settings.top_k,
    min_score=settings.min_score,
)

with st.sidebar:
    st.header("Bilgi Tabanı")
    stats = store.stats()
    st.metric("Belgeler", stats["documents"])
    st.metric("Metin Parçaları", stats["chunks"])
    st.caption(f"Sohbet modeli: {settings.chat_model}")
    st.caption(f"Embedding modeli: {settings.embedding_model}")

    if st.button("Belgeleri İndeksle / Güncelle", use_container_width=True):
        status = st.status("Yerel indeks hazırlanıyor...", expanded=True)
        try:
            report = ingest_documents(
                store,
                runtime,
                settings,
                status_callback=status.write,
            )
            status.update(label="İndeks hazır", state="complete")
            st.success(
                f"{report.files_indexed} dosya indekslendi, "
                f"{report.files_skipped} değişmeyen dosya atlandı, "
                f"{report.chunks_written} metin parçası kaydedildi."
            )
            st.rerun()
        except Exception as exc:
            status.update(label="İndeksleme başarısız", state="error")
            st.error(str(exc))

    if st.button("İndeksi Yeniden Oluştur", use_container_width=True):
        status = st.status("Yerel indeks yeniden oluşturuluyor...", expanded=True)
        try:
            report = ingest_documents(
                store,
                runtime,
                settings,
                rebuild=True,
                status_callback=status.write,
            )
            status.update(label="İndeks yeniden oluşturuldu", state="complete")
            st.success(f"{report.chunks_written} metin parçası kaydedildi.")
            st.rerun()
        except Exception as exc:
            status.update(label="İndeks oluşturulamadı", state="error")
            st.error(str(exc))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Getirilen Kaynaklar"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['label']}** — benzerlik `{source['score']:.3f}`"
                    )
                    st.caption(source["preview"])

question = st.chat_input("İndekslenen belgeler hakkında bir soru sorun")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "İlgili bilgiler getiriliyor ve yerel model çalıştırılıyor..."
            ):
                result = service.answer(question)
            st.markdown(result.answer)
            sources = [
                {
                    "label": f"[S{index}] {source.citation_label}",
                    "score": source.score,
                    "preview": source.content[:350]
                    + ("..." if len(source.content) > 350 else ""),
                }
                for index, source in enumerate(result.sources, start=1)
            ]
            if sources:
                with st.expander("Getirilen Kaynaklar"):
                    for source in sources:
                        st.markdown(
                            f"**{source['label']}** — benzerlik `{source['score']:.3f}`"
                        )
                        st.caption(source["preview"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "sources": sources}
            )
        except Exception as exc:
            message = f"Uygulama hatası: {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
