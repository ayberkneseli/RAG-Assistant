from local_rag.rag import (
    NO_INFORMATION_MESSAGE,
    NO_INFORMATION_MESSAGE_TR,
    _fallback_message,
)


def test_fallback_language_matches_common_question_language():
    assert _fallback_message("What is the cafeteria menu?") == NO_INFORMATION_MESSAGE
    assert _fallback_message("Yemek menüsü nedir?") == NO_INFORMATION_MESSAGE_TR
