#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Arabic Interrogative Pronouns Deep Learning System
=========================================================
نظام محسن للتعلم العميق لأسماء الاستفهام العربية

Optimized version with improved Transformer architecture and
enhanced training strategies for Arabic interrogative pronouns.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 2.0.0 - ENHANCED MODELS
Date: 2025-07-24
Encoding: UTF 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import random
import logging
from typing import Di        # إعدادات النماذج المحسنة
    self.model_configs = {
    'enhanced_transformer': {
    'vocab_size': self.processor.vocab_size,
    'feature_dim': self.processor.feature_dim,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'num_classes': len(INTERROGATIVE_PRONOUNS),
    'dropout': 0.1,
    'max_seq_len': 15
    }
    }le, Any, Optional
from dataclasses import dataclass
import json
import math
from collections import Counter

# إعداد البذرة للتكرار
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════


INTERROGATIVE_PRONOUNS = {
    0: "مَن",  # للسؤال عن الأشخاص
    1: "مَا",  # للسؤال عن الأشياء
    2: "مَتَى",  # للسؤال عن الزمان
    3: "أَيْنَ",  # للسؤال عن المكان
    4: "كَيْفَ",  # للسؤال عن الكيفية
    5: "كَمْ",  # للسؤال عن الكمية
    6: "أَيّ",  # للاختيار
    7: "لِمَاذَا",  # للسؤال عن السبب
    8: "مَاذَا",  # للسؤال عن الأشياء (مباشر)
    9: "أَيَّانَ",  # للسؤال عن الزمان المستقبلي
    10: "أَنَّى",  # للسؤال عن المكان والطريقة
    11: "لِمَ",  # للسؤال عن السبب (مختصر)
    12: "كَأَيِّنْ",  # للسؤال عن العدد الكثير
    13: "أَيُّهَا",  # النداء الاستفهامي
    14: "مَهْمَا",  # للسؤال عن أي شيء
    15: "أَيْنَمَا",  # للسؤال عن أي مكان
    16: "كَيْفَمَا",  # للسؤال عن أي طريقة
    17: "مَنْ ذَا",  # للسؤال عن الأشخاص بتأكيد
}

PRONOUN_TO_ID = {v: k for k, v in INTERROGATIVE_PRONOUNS.items()}


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED PHONETIC PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════════


class EnhancedPhoneticProcessor:
    """معالج صوتي محسن لأسماء الاستفهام"""

    def __init__(self):

    self.arabic_phonemes = {
            # الصوامت الأساسية
    'ب': 0,
    'ت': 1,
    'ث': 2,
    'ج': 3,
    'ح': 4,
    'خ': 5,
    'د': 6,
    'ذ': 7,
    'ر': 8,
    'ز': 9,
    'س': 10,
    'ش': 11,
    'ص': 12,
    'ض': 13,
    'ط': 14,
    'ظ': 15,
    'ع': 16,
    'غ': 17,
    'ف': 18,
    'ق': 19,
    'ك': 20,
    'ل': 21,
    'م': 22,
    'ن': 23,
    'ه': 24,
    'و': 25,
    'ي': 26,
    'ء': 27,
    'آ': 28,
    'أ': 29,
    'إ': 30,
    'ئ': 31,
    'ؤ': 32,
    'ة': 33,
            # الحركات
    'َ': 34,
    'ُ': 35,
    'ِ': 36,
    'ْ': 37,
    'ً': 38,
    'ٌ': 39,
    'ٍ': 40,
            # أصوات مركبة شائعة في أسماء الاستفهام
    'مَ': 41,
    'أَ': 42,
    'كَ': 43,
    'لِ': 44,
    'نْ': 45,
    'تَ': 46,
    'يْ': 47,
    'ذَ': 48,
    'ى': 49,
    'اذَ': 50,
    'نَ': 51,
    'فَ': 52,
    'يَّ': 53,
    'ان': 54,
    'هْ': 55,
            # رموز خاصة
    '<PAD>': 56,
    '<UNK>': 57,
    '<START>': 58,
    '<END>': 59,
    }

    self.vocab_size = len(self.arabic_phonemes)
    self.feature_dim = 16  # زيادة أبعاد الخصائص
    self.phoneme_features = self._initialize_enhanced_features()

    def _initialize_enhanced_features(self) -> Dict[str, np.ndarray]:
    """تهيئة خصائص صوتية محسنة"""

    features = {}

        for phoneme, idx in self.arabic_phonemes.items():
            # خصائص محسنة (16 بعد)
    feature_vector = np.zeros(self.feature_dim)

            # تصنيف الأصوات الأساسي
    consonants = [
    'ب',
    'ت',
    'ث',
    'ج',
    'ح',
    'خ',
    'د',
    'ذ',
    'ر',
    'ز',
    'س',
    'ش',
    'ص',
    'ض',
    'ط',
    'ظ',
    'ع',
    'غ',
    'ف',
    'ق',
    'ك',
    'ل',
    'م',
    'ن',
    'ه',
    ]

            if phoneme in consonants:
    feature_vector[0] = 1.0  # صامت

    vowels = ['و', 'ي', 'ا', 'َ', 'ُ', 'ِ', 'آ', 'ى']
            if phoneme in vowels:
    feature_vector[1] = 1.0  # صائت

            # أصوات الاستفهام الأساسية
    primary_interrogatives = ['م', 'أ', 'ك', 'ل']
            if any(c in phoneme for c in primary_interrogatives):
    feature_vector[2] = 1.0

            # أصوات الاستفهام المركبة
            if phoneme in ['مَ', 'أَ', 'كَ', 'لِ']:
    feature_vector[3] = 1.0

            # الصوامت الجانبية والأنفية
            if phoneme in ['ل', 'ر', 'ن', 'م']:
    feature_vector[4] = 1.0

            # الصوامت الاحتكاكية
    fricatives = ['ف', 'ث', 'ذ', 'س', 'ز', 'ش', 'ص', 'خ', 'غ', 'ح', 'ع', 'ه']
            if phoneme in fricatives:
    feature_vector[5] = 1.0

            # الصوامت الانفجارية
    stops = ['ب', 'ت', 'د', 'ط', 'ك', 'ق', 'ء']
            if phoneme in stops:
    feature_vector[6] = 1.0

            # الحركات الطويلة
    long_vowels = ['ا', 'و', 'ي', 'آ', 'ى']
            if phoneme in long_vowels:
    feature_vector[7] = 1.0

            # الحركات القصيرة
    short_vowels = ['َ', 'ُ', 'ِ']
            if phoneme in short_vowels:
    feature_vector[8] = 1.0

            # موضع النطق (محسن)
    labial = ['ب', 'ف', 'م', 'و']
    dental = ['ت', 'د', 'ث', 'ذ', 'ل', 'ن', 'ر', 'ز', 'س']
    palatal = ['ش', 'ج', 'ي']
    velar = ['ك', 'ق', 'غ', 'خ']
    pharyngeal = ['ح', 'ع']

            if phoneme in labial:
    feature_vector[9] = 1.0  # شفوي
            elif phoneme in dental:
    feature_vector[10] = 1.0  # لثوي
            elif phoneme in palatal:
    feature_vector[11] = 1.0  # حلقي
            elif phoneme in velar:
    feature_vector[12] = 1.0  # طبقي
            elif phoneme in pharyngeal:
    feature_vector[13] = 1.0  # بلعومي

            # التشديد والسكون
            if phoneme in ['ً', 'ٌ', 'ٍ', 'ْ', 'ّ']:
    feature_vector[14] = 1.0

            # أصوات مركبة خاصة بالاستفهام
    special_interrogatives = ['مَ', 'أَ', 'كَ', 'لِ', 'اذَ', 'يَّ', 'ان']
            if phoneme in special_interrogatives:
    feature_vector[15] = 1.0

    features[phoneme] = feature_vector

    return features

    def encode_syllables_enhanced(self, syllables: List[str], max_length: int = 12) -> np.ndarray:
    """ترميز محسن للمقاطع"""

        # إنشاء مصفوفة الترميز
    encoded = np.zeros((max_length, self.vocab_size + self.feature_dim))

        # إضافة رمز البداية
    start_idx = self.arabic_phonemes['<START>']
    encoded[0, start_idx] = 1.0

        # ترميز المقاطع
    pos = 1
        for syllable in syllables:
            if pos >= max_length - 1:  # ترك مكان لرمز النهاية
    break

            # تفكيك المقطع
    phonemes = self._decompose_syllable(syllable)

            for phoneme in phonemes:
                if pos >= max_length - 1:
    break

                if phoneme in self.arabic_phonemes:
                    # One hot encoding
    idx = self.arabic_phonemes[phoneme]
    encoded[pos, idx] = 1.0

                    # إضافة الخصائص الصوتية
                    if phoneme in self.phoneme_features:
    features = self.phoneme_features[phoneme]
    encoded[pos, self.vocab_size :] = features

    pos += 1

        # إضافة رمز النهاية
        if pos < max_length:
    end_idx = self.arabic_phonemes['<END>']
    encoded[pos, end_idx] = 1.0

    return encoded

    def _decompose_syllable(self, syllable: str) -> List[str]:
    """تفكيك المقطع إلى أصوات"""

        if syllable in self.arabic_phonemes:
    return [syllable]

        # تفكيك تدريجي
    phonemes = []
    i = 0
        while i < len(syllable):
            # محاولة مطابقة أطول صوت مركب
    found = False
            for length in range(min(3, len(syllable) - i), 0, -1):
    candidate = syllable[i : i + length]
                if candidate in self.arabic_phonemes:
    phonemes.append(candidate)
    i += length
    found = True
    break

            if not found:
                # إضافة رمز غير معروف
    phonemes.append('<UNK>')
    i += 1

    return phonemes


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED MODELS
# ═══════════════════════════════════════════════════════════════════════════════════


class EnhancedTransformer(nn.Module):
    """نموذج Transformer محسن لأسماء الاستفهام"""

    def __init__()
    self,
    vocab_size: int,
    feature_dim: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    num_classes: int = 18,
    dropout: float = 0.1, max_seq_len: int = 12):
    super(EnhancedTransformer, self).__init__()

    self.d_model = d_model
    self.vocab_size = vocab_size
    self.feature_dim = feature_dim

        # تضمين محسن
    self.phoneme_embedding = nn.Embedding(vocab_size, d_model // 2)
    self.feature_projection = nn.Linear(feature_dim, d_model // 2)

        # طبقة دمج التضمين
    self.embedding_fusion = nn.Linear(d_model, d_model)

        # ترميز الموضع المحسن
    self.pos_encoding = EnhancedPositionalEncoding(d_model, dropout, max_seq_len)

        # طبقة تطبيع أولية
    self.input_norm = nn.LayerNorm(d_model)

        # طبقات Transformer محسنة
    encoder_layer = nn.TransformerEncoderLayer()
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=d_model * 4,
    dropout=dropout,
    activation='gelu',
    batch_first=True,
    norm_first=True,  # Pre-norm للتدريب الأكثر استقراراً
    )

    self.transformer_encoder = nn.TransformerEncoder()
    encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
    )

        # آلية انتباه عالمية
    self.global_attention = nn.MultiheadAttention()
    embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
    )

        # رمز التصنيف القابل للتعلم
    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # طبقات التصنيف المتدرجة
    self.classifier = nn.Sequential()
    nn.LayerNorm(d_model),
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, d_model // 4),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 4, num_classes))

        # تهيئة الأوزان
    self._init_weights()

    def _init_weights(self):
    """تهيئة أوزان النموذج"""

        for module in self.modules():
            if isinstance(module, nn.Linear):
    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
    nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
    def forward(self, x):
    batch_size, seq_len, feature_size = x.shape

        # فصل الأصوات والخصائص
    phoneme_part = x[: : : self.vocab_size]  # One-hot phonemes
    feature_part = x[: : self.vocab_size :]  # Phonetic features

        # تحويل one-hot إلى indices
    phoneme_indices = torch.argmax(phoneme_part, dim=-1)

        # تضمين الأصوات
    phoneme_emb = self.phoneme_embedding(phoneme_indices)  # [batch, seq, d_model//2]

        # إسقاط الخصائص
    feature_emb = self.feature_projection(feature_part)  # [batch, seq, d_model//2]

        # دمج التضمين
    combined_emb = torch.cat([phoneme_emb, feature_emb], dim=-1)  # [batch, seq, d_model]
    combined_emb = self.embedding_fusion(combined_emb)

        # إضافة رمز التصنيف
    cls_tokens = self.cls_token.expand(batch_size, -1,  1)
    x = torch.cat([cls_tokens, combined_emb], dim=1)

        # ترميز الموضع
    x = self.pos_encoding(x)

        # تطبيع أولي
    x = self.input_norm(x)

        # Transformer
    transformer_out = self.transformer_encoder(x)

        # انتباه عالمي
    attended_out, _ = self.global_attention(transformer_out, transformer_out, transformer_out)

        # أخذ رمز التصنيف
    cls_output = attended_out[: 0]

        # التصنيف
    output = self.classifier(cls_output)

    return output


class EnhancedPositionalEncoding(nn.Module):
    """ترميز موضع محسن"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 15):

    super(EnhancedPositionalEncoding, self).__init__()

    self.dropout = nn.Dropout(p=dropout)

        # ترميز موضع مطلق
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[: 0::2] = torch.sin(position * div_term)
    pe[: 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

        # ترميز موضع قابل للتعلم
    self.learned_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
    def forward(self, x):
    seq_len = x.size(1)

        # التأكد من توافق الأبعاد
        if seq_len <= self.pe.size(1) and seq_len <= self.learned_pe.size(1):
            # دمج الترميز المطلق والقابل للتعلم
    pos_encoding = (self.pe[: :seq_len] + self.learned_pe[: :seq_len]) * 0.5
        else:
            # استخدام الترميز المطلق فقط إذا كان الطول أكبر من المتوقع
    pos_encoding = self.pe[: :seq_len] if seq_len <= self.pe.size(1) else self.pe

        # التأكد من توافق أبعاد الإضافة
        if pos_encoding.size(1) == x.size(1):
    x = x + pos_encoding

    return self.dropout(x)
# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED TRAINING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


def create_enhanced_synthetic_data()
    processor: EnhancedPhoneticProcessor, num_samples: int = 2000
) -> Tuple[List[np.ndarray], List[int]]:
    """إنشاء بيانات تدريب محسنة"""

    # البيانات الأساسية المحسنة
    enhanced_data = {
    "مَن": [["مَنْ"], ["مَ", "نْ"], ["مِنْ"], ["مُنْ"]],
    "مَا": [["مَا"], ["مَ", "ا"], ["مُا"], ["مِا"]],
    "مَتَى": [["مَ", "تَى"], ["مَ", "تَ", "ى"], ["مِ", "تَى"], ["مُ", "تَى"]],
    "أَيْنَ": [["أَيْ", "نَ"], ["أَ", "يْ", "نَ"], ["أُيْ", "نَ"], ["أِيْ", "نَ"]],
    "كَيْفَ": [["كَيْ", "فَ"], ["كَ", "يْ", "فَ"], ["كُيْ", "فَ"], ["كِيْ", "فَ"]],
    "كَمْ": [["كَمْ"], ["كَ", "مْ"], ["كُمْ"], ["كِمْ"]],
    "أَيّ": [["أَيّ"], ["أَ", "يّ"], ["أُيّ"], ["أِيّ"]],
    "لِمَاذَا": [["لِ", "مَا", "ذَا"], ["لِ", "مَ", "ا", "ذَا"], ["لُ", "مَا", "ذَا"]],
    "مَاذَا": [["مَا", "ذَا"], ["مَ", "ا", "ذَا"], ["مُا", "ذَا"]],
    "أَيَّانَ": [["أَيْ", "يَا", "نَ"], ["أَ", "يَّ", "ا", "نَ"], ["أُيَّ", "ا", "نَ"]],
    "أَنَّى": [["أَنْ", "نَى"], ["أَ", "نَّ", "ى"], ["أُنَّ", "ى"]],
    "لِمَ": [["لِمَ"], ["لِ", "مَ"], ["لُمَ"], ["لِ", "مُ"]],
    "كَأَيِّنْ": [["كَأَيْ", "يِنْ"], ["كَ", "أَ", "يِّ", "نْ"], ["كُأَيِّ", "نْ"]],
    "أَيُّهَا": [["أَيْ", "يُ", "هَا"], ["أَ", "يُّ", "هَا"], ["أُيُّ", "هَا"]],
    "مَهْمَا": [["مَهْ", "مَا"], ["مَ", "هْ", "مَا"], ["مُهْ", "مَا"]],
    "أَيْنَمَا": [["أَيْ", "نَ", "مَا"], ["أَ", "يْ", "نَ", "مَا"], ["أُيْ", "نَ", "مَا"]],
    "كَيْفَمَا": [["كَيْ", "فَ", "مَا"], ["كَ", "يْ", "فَ", "مَا"], ["كُيْ", "فَ", "مَا"]],
    "مَنْ ذَا": [["مَنْ", "ذَا"], ["مَ", "نْ", "ذَا"], ["مُنْ", "ذَا"]],
    }

    X = []
    y = []

    # حساب الأوزان للتوازن
    class_weights = {}
    for pronoun in enhanced_data.keys():
        class_weights[PRONOUN_TO_ID[pronoun]] = 1.0 / len(enhanced_data)

    # توليد العينات مع توازن الفئات
    samples_per_class = num_samples // len(enhanced_data)

    for pronoun, syllable_variants in enhanced_data.items():
    pronoun_id = PRONOUN_TO_ID[pronoun]

        for _ in range(samples_per_class):
            # اختيار تنويع عشوائي
    syllables = random.choice(syllable_variants).copy()

            # إضافة تشويش معتدل (20% من الوقت)
            if random.random() < 0.2:
    syllables = apply_phonetic_noise(syllables)

            # استخراج الخصائص المحسنة
    features = processor.encode_syllables_enhanced(syllables)

    X.append(features)
    y.append(pronoun_id)

    # خلط البيانات
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    return list(X), list(y)


def apply_phonetic_noise(syllables: List[str]) -> List[str]:
    """تطبيق تشويش صوتي طفيف"""

    noisy_syllables = syllables.copy()

    # تنويعات الحركات
    vowel_variations = {'َ': ['ُ', 'ِ'], 'ُ': ['َ', 'ِ'], 'ِ': ['َ', 'ُ']}

    for i, syllable in enumerate(noisy_syllables):
        if random.random() < 0.3:  # 30% احتمال تغيير
            for orig_vowel, alt_vowels in vowel_variations.items():
                if orig_vowel in syllable:
    new_vowel = random.choice(alt_vowels)
    noisy_syllables[i] = syllable.replace(orig_vowel, new_vowel)
    break

    return noisy_syllables


def train_enhanced_model()
    model: nn.Module, X: List[np.ndarray], y: List[int], epochs: int = 50, batch_size: int = 32, lr: float = 0.0005
) -> Dict[str, List[float]]:
    """تدريب محسن للنموذج"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # تحويل البيانات
    X_tensor = torch.stack([torch.FloatTensor(x) for x in X])
    y_tensor = torch.LongTensor(y)

    # تقسيم البيانات مع طبقية
    from sklearn.model_selection import train_test_split

    train_X, test_X, train_y, test_y = train_test_split()
    X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )

    # حساب أوزان الفئات للتعامل مع عدم التوازن
    class_counts = Counter(train_y.numpy())
    total_samples = len(train_y)
    class_weights = {}
    for class_id in range(len(INTERROGATIVE_PRONOUNS)):
        if class_id in class_counts:
            class_weights[class_id] = total_samples / (len(class_counts) * class_counts[class_id])
        else:
            class_weights[class_id] = 1.0

    # تحويل أوزان الفئات إلى tensor
    weight_tensor = torch.FloatTensor([class_weights[i] for i in range(len(INTERROGATIVE_PRONOUNS))])

    # DataLoader مع أخذ عينات موزونة
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)

    # حساب أوزان العينات
    sample_weights = [class_weights[label.item()] for label in train_y]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # Optimizer محسن مع جدولة معدل التعلم
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))

    # Loss function مع أوزان الفئات
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))

    # جدولة معدل التعلم
    scheduler = optim.lr_scheduler.OneCycleLR()
    optimizer, max_lr=lr * 5, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
    )

    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}

    best_test_acc = 0
    patience = 10
    patience_counter = 0

    model.train()
    for epoch in range(epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0

        for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    optimizer.zero_grad()

            # Forward pass
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

            # Backward pass
    loss.backward()

            # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    total_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total_train += batch_y.size(0)
    correct_train += (predicted == batch_y).sum().item()

        # تقييم على بيانات الاختبار
    model.eval()
        with torch.no_grad():
    test_X_device = test_X.to(device)
    test_outputs = model(test_X_device)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_accuracy = (test_predicted == test_y.to(device)).float().mean().item()

    model.train()

    train_accuracy = correct_train / total_train
    avg_loss = total_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]

    history['train_loss'].append(avg_loss)
    history['train_acc'].append(train_accuracy)
    history['test_acc'].append(test_accuracy)
    history['lr'].append(current_lr)

        # Early stopping
        if test_accuracy > best_test_acc:
    best_test_acc = test_accuracy
    patience_counter = 0
        else:
    patience_counter += 1

        if (epoch + 1) % 10 == 0:
    print()
    f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
    f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f} | }"
    f"LR: {current_lr:.6f}"
    )

        if patience_counter >= patience and epoch > 20:
    print(f"Early stopping at epoch {epoch+1}")
    break

    return history


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED INFERENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


class EnhancedInterrogativeInference:
    """نظام استنتاج محسن لأسماء الاستفهام"""

    def __init__(self):

    self.processor = EnhancedPhoneticProcessor()
    self.models = {}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # إعدادات النماذج المحسنة
    self.model_configs = {
    'enhanced_transformer': {
    'vocab_size': self.processor.vocab_size,
    'feature_dim': self.processor.feature_dim,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'num_classes': len(INTERROGATIVE_PRONOUNS),
    'dropout': 0.1,
    'max_seq_len': 12,
    }
    }

    self._initialize_models()

    def _initialize_models(self):
    """تهيئة النماذج المحسنة"""

    self.models['enhanced_transformer'] = EnhancedTransformer(**self.model_configs['enhanced_transformer'])

        for model in self.models.values():
    model.to(self.device)

    def train_enhanced_models(self, num_samples: int = 3000):
    """تدريب النماذج المحسنة"""

    print("🧠 إنشاء بيانات التدريب المحسنة...")
    X, y = create_enhanced_synthetic_data(self.processor, num_samples)

    results = {}

        for model_name, model in self.models.items():
    print(f"\n🚀 تدريب النموذج المحسن {model_name.upper()}...")
    history = train_enhanced_model(model, X, y, epochs=40, lr=0.0005)

    results[model_name] = {
    'final_train_acc': history['train_acc'][ 1],
    'final_test_acc': history['test_acc'][ 1],
    'best_test_acc': max(history['test_acc']),
    'history': history,
    }

    print()
    f"✅ {model_name.upper()} - دقة التدريب: {history['train_acc'][-1]:.3f, }"
    f"أفضل دقة اختبار: {max(history['test_acc']):.3f}"
    )

    return results

    def predict_enhanced(self, syllables: List[str], model_type: str = 'enhanced_transformer') -> Dict[str, Any]:
    """تنبؤ محسن مع تحليل مفصل"""

        if model_type not in self.models:
    raise ValueError(f"نوع النموذج غير مدعوم: {model_type}")

    model = self.models[model_type]
    model.eval()

        # استخراج الخصائص المحسنة
    features = self.processor.encode_syllables_enhanced(syllables)
    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # التنبؤ
        with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]

        # ترتيب النتائج
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    results = {
    'input_syllables': syllables,
    'best_prediction': INTERROGATIVE_PRONOUNS[sorted_indices[0].item()],
    'confidence': sorted_probs[0].item(),
    'alternatives': [],
    'entropy': -torch.sum(probabilities * torch.log(probabilities + 1e 10)).item(),
    'max_prob': sorted_probs[0].item(),
    'prediction_strength': ()
    'high' if sorted_probs[0].item()  > 0.8 else 'medium' if sorted_probs[0].item()  > 0.5 else 'low'
    ),
    }

        # إضافة البدائل
        for i in range(min(5, len(sorted_indices))):
    idx = sorted_indices[i].item()
    prob = sorted_probs[i].item()
    results['alternatives'].append()
    {
    'pronoun': INTERROGATIVE_PRONOUNS[idx],
    'confidence': prob,
    'relative_strength': prob / sorted_probs[0].item() if sorted_probs[0].item()  > 0 else 0,
    }
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════


def main():
    """التشغيل الرئيسي للنظام المحسن"""

    print("🧠 النظام المحسن للتعلم العميق - أسماء الاستفهام العربية")
    print("=" * 65)

    # إنشاء نظام الاستنتاج المحسن
    enhanced_inference = EnhancedInterrogativeInference()

    # تدريب النماذج المحسنة
    print("🚀 بدء تدريب النماذج المحسنة...")
    results = enhanced_inference.train_enhanced_models(num_samples=2500)

    # عرض النتائج
    print(f"\n📊 نتائج التدريب المحسن:")
    for model_name, result in results.items():
    print(f"   {model_name.upper()}: {result['best_test_acc']:.1% أفضل} دقة اختبار}")

    # اختبار التنبؤ المحسن
    print(f"\n🔬 اختبار التنبؤ المحسن:")
    test_cases = [
    ["مَنْ"],  # مَن
    ["مَا"],  # ما
    ["مَ", "تَى"],  # متى
    ["أَيْ", "نَ"],  # أين
    ["كَيْ", "فَ"],  # كيف
    ["لِ", "مَا", "ذَا"],  # لماذا
    ]

    for syllables in test_cases:
    result = enhanced_inference.predict_enhanced(syllables)
    print(f"\n   المقاطع: {syllables}")
    print(f"     التنبؤ: {result['best_prediction']} (ثقة: {result['confidence']:.3f)}")
    print(f"     قوة التنبؤ: {result['prediction_strength']}")
    print(f"     البدائل الأولى:")
        for alt in result['alternatives'][:3]:
    print(f"       - {alt['pronoun']: {alt['confidence']:.3f}}")

    print(f"\n✅ اكتمل النظام المحسن!")


if __name__ == "__main__":
    main()

