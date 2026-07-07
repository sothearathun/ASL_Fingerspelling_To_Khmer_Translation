from deep_translator import GoogleTranslator


def translate_to_khmer(text: str) -> str:
    if not text:
        return ""
    try:
        return GoogleTranslator(source='en', target='km').translate(text)
    except Exception:
        return "(translation unavailable)"
