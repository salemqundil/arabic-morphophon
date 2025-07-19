from phonology.classifier import classify_letter

SHORT_VOWELS = {"َ": "A", "ُ": "O", "ِ": "E"}
LONG_VOWEL_MAP = {"َ": "ا", "ُ": "و", "ِ": "ي"}
SUKOON = "ْ"
SHADDA = "ّ"
TANWEEN = {"ً": "An", "ٌ": "On", "ٍ": "En"}


def encode_syllables(text: str):
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        entry = {"letter": char}

        # Skip anything not a letter
        if not char.isalpha():
            i += 1
            continue

        # Morphological classification
        entry["classification"] = classify_letter(char)

        # Default values
        entry["vowel"] = ""  # Initialize as empty string instead of None
        entry["syllable_code"] = ""  # Initialize as empty string instead of None

        # Look ahead
        next1 = text[i + 1] if i + 1 < len(text) else ""
        next2 = text[i + 2] if i + 2 < len(text) else ""

        # Tanween check
        if next1 in TANWEEN:
            entry["vowel"] = next1
            entry["syllable_code"] = f"CV{TANWEEN[next1]}"
            i += 2
            result.append(entry)
            continue

        # Shadda detection (repeat + vowel)
        if next1 == SHADDA:
            # Look for vowel after shadda
            vowel = text[i + 2] if i + 2 < len(text) else ""
            entry["vowel"] = vowel
            if vowel in SHORT_VOWELS:
                entry["syllable_code"] = f"CV{SHORT_VOWELS[vowel]}SH"
            i += 3
            result.append(entry)
            continue

        # Long vowel detection
        if next1 in SHORT_VOWELS and next2 == LONG_VOWEL_MAP.get(next1):
            entry["vowel"] = next1
            entry["syllable_code"] = f"CVV{SHORT_VOWELS[next1]}"
            i += 3
            result.append(entry)
            continue

        # Short vowel only
        if next1 in SHORT_VOWELS:
            entry["vowel"] = next1
            next_vowel = text[i + 2] if i + 2 < len(text) else ""
            if next_vowel == SUKOON:
                entry["syllable_code"] = f"CV{SHORT_VOWELS[next1]}C"
                i += 3
            else:
                entry["syllable_code"] = f"CV{SHORT_VOWELS[next1]}"
                i += 2
            result.append(entry)
            continue

        # Sukoon alone
        if next1 == SUKOON:
            entry["vowel"] = SUKOON
            entry["syllable_code"] = "C"
            i += 2
            result.append(entry)
            continue

        # Unknown or unmarked letter
        entry["syllable_code"] = "unknown"
        i += 1
        result.append(entry)

    return result
