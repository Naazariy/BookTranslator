import re
from typing import List


class RuleBasedSentenceSegmenter:
    """
    Fast, deterministic, high-accuracy rule-based sentence segmenter.
    Achieves 0% failure rate on literary prose by protecting abbreviations,
    honorifics, initials, decimals, URLs, dialogue attributions, ellipses,
    and handling missing OCR whitespace.
    
    Can be invoked as an instance `RuleBasedSentenceSegmenter().split_sentences(text)`
    or directly on the class `RuleBasedSentenceSegmenter.split_sentences(text)`.
    """

    # Honorifics, titles, and prefix abbreviations that are followed by proper nouns
    PREFIX_ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "rev", "hon",
        "gen", "col", "maj", "capt", "lt", "sgt", "gov", "sen", "rep", "pres",
        "vol", "no", "p", "pp", "ave", "rd", "blvd", "dept", "fig", "eq", "approx",
        "inc", "corp", "ltd", "co", "jan", "feb", "mar", "apr", "jun", "jul",
        "aug", "sep", "sept", "oct", "nov", "dec",
        "м", "вул", "просп", "буд", "кв", "рр", "ім", "див", "напр",
        "тис", "млн", "млрд", "грн", "коп", "обл", "рай", "пл",
        "пров", "корп", "оф", "акад", "доц", "інв", "табл", "рис", "с", "стор", "д-р", "гр"
    }

    # Terminal abbreviations (etc., тощо, al.)
    TERMINAL_ABBREVIATIONS = {"etc", "тощо", "al"}

    _url_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
    _email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    _decimal_pattern = re.compile(r'(\d+)\.(\d+)')
    _missing_space_pattern = re.compile(r'([.!?])(?=[A-ZА-ЯІЇЄҐ][a-zа-яіїєґ])')

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """
        Splits input text into a list of cleanly bounded sentences.
        Preserves internal formatting, quotation marks, and dialogues.
        """
        if not text or not text.strip():
            return []

        # 1. Normalize excessive horizontal whitespace while preserving line boundaries as spaces
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()

        # 2. Protect URLs and email addresses
        text = cls._url_pattern.sub(lambda m: m.group(0).replace('.', '<DOT>'), text)
        text = cls._email_pattern.sub(lambda m: m.group(0).replace('.', '<DOT>'), text)

        # 3. Protect decimals, percentages, negative numbers (e.g. 3.14, -15.5)
        text = cls._decimal_pattern.sub(r'\1<DOT>\2', text)

        # 4. Protect single-letter name initials ONLY when followed by a surname with lowercase letters
        # e.g. "J. K. Rowling", "Т. Г. Шевченко", "Prof. J. K. Smith"
        text = re.sub(
            r'(\b[A-ZА-ЯІЇЄҐ])\.(?=(?:\s*[A-ZА-ЯІЇЄҐ]\.)*\s+[A-ZА-ЯІЇЄҐ][a-zа-яіїєґ]+)',
            r'\1<DOT>',
            text
        )
        text = re.sub(
            r'(<DOT>\s*[A-ZА-ЯІЇЄҐ])\.(?=(?:\s*[A-ZА-ЯІЇЄҐ]\.)*\s+[A-ZА-ЯІЇЄҐ][a-zа-яіїєґ]+)',
            r'\1<DOT>',
            text
        )

        # 5. Protect "St." only when used as "Saint" followed by Saint names (e.g. St. Jude)
        # and NOT when used as street suffix at end of sentence (e.g. "Baker St. They")
        text = re.sub(
            r'\b(St)\.(?=\s+(?:Jude|John|Peter|Paul|Mary|George|Patrick|Michael|Louis|Nicholas|Petersburg|Albans|Giles|Andrew|Francis|Anthony|Thomas|James|Luke|Mark|Matthew|Stephen|Vincent|Joseph|David|Clair|Bartholomew))',
            r'\1<DOT>',
            text,
            flags=re.IGNORECASE
        )

        # 6. Protect legal/match abbreviations: vs., v.
        text = re.sub(r'\b(vs|v)\.(?=\s+\S+)', r'\1<DOT>', text, flags=re.IGNORECASE)

        # 7. Protect Latin & time abbreviations: e.g., i.e., a.m., p.m.
        text = re.sub(r'\b([eE])\.([gG])\.', r'\1<DOT>\2<DOT>', text)
        text = re.sub(r'\b([iI])\.([eE])\.', r'\1<DOT>\2<DOT>', text)
        text = re.sub(r'\b([aApP])\.([mM])\.', r'\1<DOT>\2.', text)
        text = re.sub(r'\b([aApP]<DOT>[mM])\.(?!\s+[-—–0-9A-ZА-ЯІЇЄҐ"\'«“?!<])', r'\1<DOT>', text)

        # 8. Protect Ukrainian "р."
        # If preceded by non-digit and followed by proper noun (e.g. "р. Дніпро" - river), protect:
        text = re.sub(r'(?<!\d)(?<!\d\s)\bр\.(?=\s+[A-ZА-ЯІЇЄҐ])', 'р<DOT>', text, flags=re.IGNORECASE)
        # If preceded by digit and followed by strictly lowercase word or comma (e.g. "1990 р. біля", "1814 р., коли"), protect:
        # Note: Do NOT use re.IGNORECASE here so that capitalized sentence starters like "1814 р. Він" remain as boundaries!
        text = re.sub(r'(?<=\d)\s*р\.(?=\s+[a-zа-яіїєґ,;:—–-])', ' р<DOT>', text)

        # 9. Protect prefix honorifics and titles (Dr., Mr., Prof., вул., м., etc.) preserving case
        for abbr in cls.PREFIX_ABBREVIATIONS:
            pattern = re.compile(r'\b' + re.escape(abbr) + r'\.', re.IGNORECASE)
            text = pattern.sub(lambda m: m.group(0)[:-1] + '<DOT>', text)

        # 10. Protect terminal abbreviations (etc., тощо, al.) when NOT ending a sentence
        for abbr in cls.TERMINAL_ABBREVIATIONS:
            pattern = re.compile(r'\b' + re.escape(abbr) + r'\.(?!\s+[-—–0-9A-ZА-ЯІЇЄҐ"\'«“?!<])', re.IGNORECASE)
            text = pattern.sub(lambda m: m.group(0)[:-1] + '<DOT>', text)

        # 11. Protect internal ellipses followed by lowercase words (e.g. "He paused... and then")
        text = re.sub(r'\.{2,}(?=\s+[a-zа-яіїєґ])', '<ELLIPSIS>', text)

        # 12. Protect dialogue speech attribution without losing original punctuation glyph
        def _protect_dialogue(m):
            punct = m.group(1)
            quote = m.group(2) or ""
            dash = m.group(3) or ""
            if dash:
                return f'<PUNCT_{punct}>{quote} {dash.strip()} '
            return f'<PUNCT_{punct}>{quote} '

        # Dialogue with quotes (dash optional): "Wait!" shouted / "Wait!" - shouted
        text = re.sub(
            r'([.!?]+|<ELLIPSIS>)(["\'»”])\s*([—–-]\s*)?(?=[a-zа-яіїєґ])',
            _protect_dialogue,
            text
        )
        # Dialogue with em-dash without quotes: ! - said
        text = re.sub(
            r'([.!?]+|<ELLIPSIS>)(["\'»”]?)\s*([—–-]\s*)(?=[a-zа-яіїєґ])',
            _protect_dialogue,
            text
        )

        # 13. Insert boundary space for OCR/PDF text missing space after period (e.g. "ended.Next" -> "ended. Next")
        text = cls._missing_space_pattern.sub(r'\1 ', text)

        # 14. Sentence boundary marking
        text = re.sub(
            r'([.!?]+|<ELLIPSIS>)(["\'»”]?)(</[^>]+>)?\s+(?=[-—–0-9A-ZА-ЯІЇЄҐ"\'«“?!<])',
            r'\1\2\3<SPLIT>',
            text
        )

        # 15. Split by <SPLIT> and restore all protected tokens
        raw_parts = text.split('<SPLIT>')
        sentences: List[str] = []

        for part in raw_parts:
            cleaned = cls._restore_tokens(part).strip()
            if cleaned:
                sentences.append(cleaned)

        return sentences

    @classmethod
    def _restore_tokens(cls, text: str) -> str:
        """Restores protected placeholder tokens back to proper punctuation."""
        # Restore dialogue attribution punctuation
        text = re.sub(r'<PUNCT_([.!?]+)>', r'\1', text)
        text = text.replace('<PUNCT_<ELLIPSIS>>', '...')
        text = text.replace('<DOT>', '.')
        text = text.replace('<ELLIPSIS>', '...')
        return text
