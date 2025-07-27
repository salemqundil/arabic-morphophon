"""
🔤 Arabic Text Normalizer (محسن النصوص العربية)
Comprehensive normalization for historical, dialectal, and classical Arabic texts,
    Handles spelling variants, diacritics, and era specific conventions
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
    class ArabicEra(Enum):
    """Historical eras of Arabic language"""

    PRE_ISLAMIC = "الجاهلية"
    EARLY_ISLAMIC = "صدر الإسلام"
    UMAYYAD = "العصر الأموي"
    ABBASID = "العصر العباسي"
    ANDALUSIAN = "الأندلسي"
    MAMLUK = "المملوكي"
    OTTOMAN = "العثماني"
    MODERN = "الحديث"
    CONTEMPORARY = "المعاصر"


class ArabicDialect(Enum):
    """Major Arabic dialect groups"""

    CLASSICAL = "فصحى التراث"
    MODERN_STANDARD = "فصحى العصر"
    EGYPTIAN = "مصري"
    LEVANTINE = "شامي"
    GULF = "خليجي"
    MAGHREBI = "مغربي"
    IRAQI = "عراقي"
    SUDANESE = "سوداني"
    YEMENI = "يمني"


@dataclass,
    class NormalizationRule:
    """Rule for text normalization"""

    pattern: str,
    replacement: str,
    era: Optional[ArabicEra]
    dialect: Optional[ArabicDialect]
    context: List[str]
    priority: int,
    description: str,
    class ArabicNormalizer:
    """Comprehensive Arabic text normalizer"""

    def __init__(self):
    self.historical_variants = self._initialize_historical_variants()
    self.dialect_mappings = self._initialize_dialect_mappings()
    self.diacritic_rules = self._initialize_diacritic_rules()
    self.orthographic_rules = self._initialize_orthographic_rules()
    self.normalization_rules = self._initialize_normalization_rules()

    def _initialize_historical_variants(self) -> Dict[ArabicEra, Dict[str, str]]:
    """Initialize historical spelling variants by era"""
    return {
    ArabicEra.PRE_ISLAMIC: {
                # Ancient script variations
    'لاه': 'الله',
    'الرحمن': 'الرحمان',
    'إله': 'إلاه',
    'كلم': 'كلام',
    'جبل': 'جبال',
    'نهر': 'أنهار',
                # Old poetry conventions
    'سيف': 'سيوف',
    'بيت': 'بيوت',
    'فرس': 'أفراس',
    },
    ArabicEra.EARLY_ISLAMIC: {
                # Quranic spelling conventions
    'الصلوة': 'الصلاة',
    'الزكوة': 'الزكاة',
    'الحيوة': 'الحياة',
    'مشكوة': 'مشكاة',
    'نجوة': 'نجاة',
    'رحمة': 'رحمت',  # Ottoman influence
    'حكمة': 'حكمت',
    },
    ArabicEra.ABBASID: {
                # Classical Arabic standardization
    'إبن': 'ابن',
    'إبنة': 'ابنة',
    'إثنان': 'اثنان',
    'إثنين': 'اثنين',
    'إسم': 'اسم',
    'إستطاع': 'استطاع',
                # Philosophical/scientific terms
    'الفلسفة': 'الفلسفه',
    'الطبيعة': 'الطبيعه',
    'الكيمياء': 'الكيمياأ',
    },
    ArabicEra.ANDALUSIAN: {
                # Andalusian Arabic variants
    'قرطبة': 'قرطبه',
    'إشبيلية': 'إشبيليه',
    'غرناطة': 'غرناطه',
    'الموشح': 'الموشحات',
    'الزجل': 'الزجال',
                # Romance language influence
    'الأندلس': 'الأندلوس',
    'البرتقال': 'النارنج',
    },
    ArabicEra.MODERN: {
                # Modern spelling reforms
    'هذه': 'هاذه',
    'هؤلاء': 'هاؤلاء',
    'أولئك': 'أولائك',
    'إمرأة': 'امرأة',
    'مرأة': 'امرأة',
                # Borrowed terms standardization
    'التلفزيون': 'التلفزيون',
    'الكمبيوتر': 'الحاسوب',
    'الإنترنت': 'الشابكة',
    },
    }

    def _initialize_dialect_mappings(self) -> Dict[ArabicDialect, Dict[str, str]]:
    """Initialize dialect-to standard mappings"""
    return {
    ArabicDialect.EGYPTIAN: {
                # Egyptian colloquial to MSA
    'إيه': 'ما',
    'ده': 'هذا',
    'دي': 'هذه',
    'دول': 'هؤلاء',
    'كده': 'هكذا',
    'فين': 'أين',
    'إمتى': 'متى',
    'ليه': 'لماذا',
    'ازاي': 'كيف',
    'عايز': 'أريد',
    'عاوز': 'أريد',
    'جاي': 'آت',
    'رايح': 'ذاهب',
    'قاعد': 'جالس',
    'واقف': 'واقف',
    'نايم': 'نائم',
    'صاحي': 'مستيقظ',
    'جعان': 'جائع',
    'عطشان': 'عطش',
    },
    ArabicDialect.LEVANTINE: {
                # Levantine to MSA
    'شو': 'ما',
    'هاد': 'هذا',
    'هاي': 'هذه',
    'هدول': 'هؤلاء',
    'هيك': 'هكذا',
    'وين': 'أين',
    'إيمتا': 'متى',
    'ليش': 'لماذا',
    'كيف': 'كيف',
    'بدي': 'أريد',
    'بدو': 'يريد',
    'جاي': 'آت',
    'رايح': 'ذاهب',
    'قاعد': 'جالس',
    'واقف': 'واقف',
    'نايم': 'نائم',
    'صاحي': 'مستيقظ',
    },
    ArabicDialect.GULF: {
                # Gulf Arabic to MSA
    'شنو': 'ما',
    'ذا': 'هذا',
    'ذي': 'هذه',
    'ذول': 'هؤلاء',
    'جذي': 'هكذا',
    'وين': 'أين',
    'متا': 'متى',
    'ليش': 'لماذا',
    'شلون': 'كيف',
    'أبي': 'أريد',
    'يبي': 'يريد',
    'جاي': 'آت',
    'رايح': 'ذاهب',
    'قاعد': 'جالس',
    'واقف': 'واقف',
    'نايم': 'نائم',
    'صاحي': 'مستيقظ',
    },
    ArabicDialect.MAGHREBI: {
                # Maghrebi to MSA (simplified)
    'آش': 'ما',
    'هاذا': 'هذا',
    'هاذي': 'هذه',
    'هاذو': 'هؤلاء',
    'هكذا': 'هكذا',
    'فين': 'أين',
    'فوقتاش': 'متى',
    'علاش': 'لماذا',
    'كيفاش': 'كيف',
    'بغيت': 'أريد',
    'جاي': 'آت',
    'ماشي': 'ذاهب',
    'قاعد': 'جالس',
    'واقف': 'واقف',
    'راقد': 'نائم',
    'فايق': 'مستيقظ',
    },
    }

    def _initialize_diacritic_rules(self) -> Dict[str, NormalizationRule]:
    """Initialize diacritic normalization rules"""
    return {
    'remove_all_diacritics': NormalizationRule(
    pattern=r'[َُِّْٰٱٲٳٴٵٶٷٸٹٺٻټٽپٿ]',
    replacement='',
    era=None,
    dialect=None,
    context=['casual_text', 'modern_writing'],
    priority=1,
    description='Remove all diacritical marks',
    ),
    'preserve_essential_diacritics': NormalizationRule(
    pattern=r'[َُِ]',
    replacement='',
    era=None,
    dialect=None,
    context=['classical_text', 'quranic_text'],
    priority=10,  # High priority to preserve,
    description='Preserve essential diacritics in classical texts',
    ),
    'normalize_tanween': NormalizationRule(
    pattern=r'[ًٌٍ]',
    replacement='',
    era=None,
    dialect=None,
    context=['simplified_text'],
    priority=2,
    description='Remove tanween marks',
    ),
    'normalize_shadda': NormalizationRule(
    pattern=r'ّ',
    replacement='',
    era=None,
    dialect=None,
    context=['simplified_text'],
    priority=3,
    description='Remove shadda (gemination) marks',
    ),
    }

    def _initialize_orthographic_rules(self) -> List[NormalizationRule]:
    """Initialize orthographic normalization rules"""
    return [
            # Alef variations,
    NormalizationRule(
    pattern=r'[آأإ]',
    replacement='ا',
    era=None,
    dialect=None,
    context=['general'],
    priority=1,
    description='Normalize alef variations',
    ),
            # Yeh variations,
    NormalizationRule(
    pattern=r'[ىئي]',
    replacement='ي',
    era=None,
    dialect=None,
    context=['general'],
    priority=1,
    description='Normalize yeh variations',
    ),
            # Teh marbuta,
    NormalizationRule(
    pattern=r'ة',
    replacement='ه',
    era=None,
    dialect=None,
    context=['phonetic_matching'],
    priority=2,
    description='Normalize teh marbuta to heh',
    ),
            # Waw hamza,
    NormalizationRule(
    pattern=r'ؤ',
    replacement='و',
    era=None,
    dialect=None,
    context=['simplified'],
    priority=2,
    description='Normalize waw with hamza',
    ),
            # Yeh hamza,
    NormalizationRule(
    pattern=r'ئ',
    replacement='ي',
    era=None,
    dialect=None,
    context=['simplified'],
    priority=2,
    description='Normalize yeh with hamza',
    ),
    ]

    def _initialize_normalization_rules(self) -> List[NormalizationRule]:
    """Initialize comprehensive normalization rules"""
    rules = []

        # Add diacritic rules,
    rules.extend(self.diacritic_rules.values())

        # Add orthographic rules,
    rules.extend(self.orthographic_rules)

        # Sort by priority,
    rules.sort(key=lambda r: r.priority)

    return rules,
    def normalize_text(
    self,
    text: str,
    target_era: Optional[ArabicEra] = None,
    target_dialect: Optional[ArabicDialect] = None,
    context: List[str] = None,
    preserve_diacritics: bool = False,
    ) -> Dict:
    """Comprehensive text normalization"""

        if context is None:
    context = ['general']

    original_text = text,
    normalized_text = text

        # Step 1: Historical variant normalization,
    if target_era:
    normalized_text = self._normalize_historical_variants(
    normalized_text, target_era
    )

        # Step 2: Dialect normalization,
    if target_dialect and target_dialect != ArabicDialect.CLASSICAL:
    normalized_text = self._normalize_dialect(normalized_text, target_dialect)

        # Step 3: Orthographic normalization,
    normalized_text = self._apply_orthographic_rules(normalized_text, context)

        # Step 4: Diacritic handling,
    if not preserve_diacritics:
    normalized_text = self._normalize_diacritics(normalized_text, context)

        # Step 5: Final cleanup,
    normalized_text = self._final_cleanup(normalized_text)

        # Generate normalization report,
    changes = self._generate_change_report(original_text, normalized_text)

    return {
    'original_text': original_text,
    'normalized_text': normalized_text,
    'target_era': target_era.value if target_era else None,
    'target_dialect': target_dialect.value if target_dialect else None,
    'context': context,
    'preserve_diacritics': preserve_diacritics,
    'changes_made': changes,
    'similarity_score': self._calculate_similarity(
    original_text, normalized_text
    ),
    }

    def _normalize_historical_variants(self, text: str, era: ArabicEra) -> str:
    """Normalize historical spelling variants"""
        if era not in self.historical_variants:
    return text,
    normalized = text,
    variants = self.historical_variants[era]

        for old_form, new_form in variants.items():
    normalized = re.sub(rf'\b{re.escape(old_form)}\b', new_form, normalized)

    return normalized,
    def _normalize_dialect(self, text: str, target_dialect: ArabicDialect) -> str:
    """Normalize dialectal forms to standard Arabic"""
        if target_dialect not in self.dialect_mappings:
    return text,
    normalized = text,
    mappings = self.dialect_mappings[target_dialect]

        for dialect_form, standard_form in mappings.items():
    normalized = re.sub(
    rf'\b{re.escape(dialect_form)}\b', standard_form, normalized
    )

    return normalized,
    def _apply_orthographic_rules(self, text: str, context: List[str]) -> str:
    """Apply orthographic normalization rules"""
    normalized = text,
    for rule in self.orthographic_rules:
            if not rule.context or any(ctx in context for ctx in rule.context):
    normalized = re.sub(rule.pattern, rule.replacement, normalized)

    return normalized,
    def _normalize_diacritics(self, text: str, context: List[str]) -> str:
    """Normalize diacritical marks based on context"""
    normalized = text

        # Apply diacritic rules based on context,
    for rule_name, rule in self.diacritic_rules.items():
            if rule.context and any(ctx in context for ctx in rule.context):
                if (
    rule_name != 'preserve_essential_diacritics'
    ):  # Skip preservation rule,
    normalized = re.sub(rule.pattern, rule.replacement, normalized)

    return normalized,
    def _final_cleanup(self, text: str) -> str:
    """Final text cleanup and standardization"""
        # Remove extra whitespace,
    cleaned = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace,
    cleaned = cleaned.strip()

        # Normalize punctuation,
    cleaned = re.sub(r'[،؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ]', '،', cleaned)
    cleaned = re.sub(r'[؛؞]', '؛', cleaned)
    cleaned = re.sub(r'[؟]', '؟', cleaned)

    return cleaned,
    def _generate_change_report(self, original: str, normalized: str) -> List[Dict]:
    """Generate report of changes made during normalization"""
    changes = []

        # Simple word-level comparison,
    original_words = original.split()
    normalized_words = normalized.split()

        for i, (orig_word, norm_word) in enumerate(
    zip(original_words, normalized_words)
    ):
            if orig_word != norm_word:
    changes.append(
    {
    'position': i,
    'original': orig_word,
    'normalized': norm_word,
    'change_type': self._classify_change(orig_word, norm_word),
    }
    )

    return changes,
    def _classify_change(self, original: str, normalized: str) -> str:
    """Classify the type of change made"""
        # Remove diacritics for comparison,
    orig_no_diac = re.sub(r'[َُِّْٰ]', '', original)
    norm_no_diac = re.sub(r'[َُِّْٰ]', '', normalized)

        if orig_no_diac == norm_no_diac:
    return 'diacritic_removal'
        elif len(original) != len(normalized):
    return 'length_change'
        elif original[0] != normalized[0]:
    return 'initial_change'
        else:
    return 'character_substitution'

    def _calculate_similarity(self, text1: str, text2: str) -> float:
    """Calculate similarity between original and normalized text"""
        if not text1 or not text2:
    return 0.0

        # Simple character-level similarity,
    max_len = max(len(text1), len(text2))
    matches = sum(1 for i, j in zip(text1, text2) if i == j)

    return matches / max_len,
    def detect_era(self, text: str) -> Tuple[ArabicEra, float]:
    """Detect the most likely historical era of a text"""
    era_scores = {}

        for era, variants in self.historical_variants.items():
    score = 0,
    for old_form in variants.keys():
                if re.search(rf'\b{re.escape(old_form)}\b', text):
    score += 1

            # Normalize by total possible variants,
    era_scores[era] = score / len(variants) if variants else 0,
    if not era_scores:
    return ArabicEra.MODERN, 0.0,
    best_era = max(era_scores, key=era_scores.get)
    confidence = era_scores[best_era]

    return best_era, confidence,
    def detect_dialect(self, text: str) -> Tuple[ArabicDialect, float]:
    """Detect the most likely dialect of a text"""
    dialect_scores = {}

        for dialect, mappings in self.dialect_mappings.items():
    score = 0,
    for dialect_form in mappings.keys():
                if re.search(rf'\b{re.escape(dialect_form)}\b', text):
    score += 1

            # Normalize by total possible forms,
    dialect_scores[dialect] = score / len(mappings) if mappings else 0,
    if not dialect_scores:
    return ArabicDialect.MODERN_STANDARD, 0.0,
    best_dialect = max(dialect_scores, key=dialect_scores.get)
    confidence = dialect_scores[best_dialect]

    return best_dialect, confidence,
    def convert_between_eras(
    self, text: str, source_era: ArabicEra, target_era: ArabicEra
    ) -> Dict:
    """Convert text between different historical eras"""

        # First normalize from source era,
    intermediate = self._normalize_historical_variants(text, source_era)

        # Then apply target era conventions (reverse mapping)
        if target_era in self.historical_variants:
    target_variants = {
    v: k for k, v in self.historical_variants[target_era].items()
    }
    final_text = intermediate,
    for standard_form, era_form in target_variants.items():
    final_text = re.sub(
    rf'\b{re.escape(standard_form)}\b', era_form, final_text
    )
        else:
    final_text = intermediate,
    return {
    'original_text': text,
    'source_era': source_era.value,
    'target_era': target_era.value,
    'converted_text': final_text,
    'conversion_quality': self._assess_conversion_quality(text, final_text),
    }

    def _assess_conversion_quality(self, original: str, converted: str) -> Dict:
    """Assess the quality of era conversion"""
    return {
    'character_similarity': self._calculate_similarity(original, converted),
    'word_count_original': len(original.split()),
    'word_count_converted': len(converted.split()),
    'length_ratio': len(converted) / len(original) if original else 0,
    'preservation_score': 1.0
    - abs(len(original) - len(converted))
    / max(len(original), len(converted), 1),
    }
