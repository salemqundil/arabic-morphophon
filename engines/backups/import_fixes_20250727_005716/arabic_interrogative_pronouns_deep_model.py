#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Interrogative Pronouns Deep Learning Models
================================================
نماذج التعلم العميق لأسماء الاستفهام العربية

Advanced deep learning models (LSTM, GRU, Transformer) for Arabic interrogative
pronouns classification from phonetic features and syllable patterns.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - DEEP LEARNING MODELS
Date: 2025-07-24
Encoding: UTF 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Any
import math

# إعداد البذرة للتكرار
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# INTERROGATIVE PRONOUNS MAPPING
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
# PHONETIC PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativePhoneticProcessor:
    """معالج صوتي متقدم لأسماء الاستفهام"""

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
    }

    self.phoneme_features = self._initialize_phoneme_features()
    self.vocab_size = len(self.arabic_phonemes)

    def _initialize_phoneme_features(self) -> Dict[str, np.ndarray]:
    """تهيئة الخصائص الصوتية للأصوات"""

    features = {}

        # خصائص صوتية أساسية (12 أبعاد)
        for phoneme, idx in self.arabic_phonemes.items():
    feature_vector = np.zeros(12)

            # تصنيف الأصوات
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

    vowels = ['و', 'ي', 'ا', 'َ', 'ُ', 'ِ', 'آ']
            if phoneme in vowels:
    feature_vector[1] = 1.0  # صائت

            # أصوات شائعة في أسماء الاستفهام
    common_interrogatives = ['م', 'أ', 'ك', 'ل', 'ن', 'ت', 'ي', 'ذ', 'ه', 'ف']
            if any(c in phoneme for c in common_interrogatives):
    feature_vector[2] = 1.0

            # أصوات الاستفهام المميزة
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

            # موضع النطق
    labial = ['ب', 'ف', 'م', 'و']
    dental = ['ت', 'د', 'ث', 'ذ', 'ل', 'ن', 'ر', 'ز', 'س', 'ش']
    velar = ['ك', 'ق', 'غ', 'خ']

            if phoneme in labial:
    feature_vector[9] = 1.0  # شفوي
            elif phoneme in dental:
    feature_vector[9] = 0.5  # لثوي
            elif phoneme in velar:
    feature_vector[9] = 0.0  # خلفي

            # التشديد والسكون
            if phoneme in ['ً', 'ٌ', 'ٍ', 'ْ', 'ّ']:
    feature_vector[10] = 1.0

            # أصوات مركبة خاصة بالاستفهام
    interrogative_compounds = ['مَ', 'أَ', 'كَ', 'لِ', 'اذَ', 'يَّ']
            if phoneme in interrogative_compounds:
    feature_vector[11] = 1.0

    features[phoneme] = feature_vector

    return features

    def syllables_to_phonemes(self, syllables: List[str]) -> List[str]:
    """تحويل المقاطع إلى أصوات"""

    phonemes = []

        for syllable in syllables:
            # تفكيك المقطع إلى أصوات
            if syllable in self.arabic_phonemes:
                # مقطع مركب معروف
    phonemes.append(syllable)
            else:
                # تفكيك المقطع حرف بحرف
                for char in syllable:
                    if char in self.arabic_phonemes:
    phonemes.append(char)

    return phonemes

    def encode_phonemes(self, phonemes: List[str], max_length: int = 15) -> np.ndarray:
    """ترميز الأصوات إلى تمثيل رقمي"""

        # إنشاء مصفوفة الترميز
    encoded = np.zeros((max_length, self.vocab_size + 12))  # vocab + features

        for i, phoneme in enumerate(phonemes[:max_length]):
            if phoneme in self.arabic_phonemes:
                # One-hot encoding للصوت
    idx = self.arabic_phonemes[phoneme]
    encoded[i, idx] = 1.0

                # إضافة الخصائص الصوتية
                if phoneme in self.phoneme_features:
    features = self.phoneme_features[phoneme]
    encoded[i, self.vocab_size : self.vocab_size + 12] = features

    return encoded

    def extract_features(self, syllables: List[str]) -> np.ndarray:
    """استخراج الخصائص من المقاطع"""

    phonemes = self.syllables_to_phonemes(syllables)
    encoded = self.encode_phonemes(phonemes)

    return encoded


# ═══════════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativeLSTM(nn.Module):
    """نموذج LSTM لتصنيف أسماء الاستفهام"""

    def __init__()
    self,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    num_classes: int, dropout: float = 0.3):
    super(InterrogativeLSTM, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers

        # طبقة LSTM ثنائية الاتجاه
    self.lstm = nn.LSTM()
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=True,
    dropout=dropout if num_layers > 1 else 0)

        # طبقة الانتباه
    self.attention = nn.Linear(hidden_size * 2, 1)

        # طبقات التصنيف
    self.classifier = nn.Sequential()
    nn.Linear(hidden_size * 2, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, hidden_size // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size // 2, num_classes))

    def forward(self, x):
    def forward(self, x):
    batch_size = x.size(0)

        # تهيئة الحالات المخفية
    h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # LSTM
    lstm_out, _ = self.lstm(x, (h0, c0))

        # آلية الانتباه
    attention_weights = F.softmax(self.attention(lstm_out), dim=1)
    attended_output = torch.sum(attention_weights * lstm_out, dim=1)

        # التصنيف
    output = self.classifier(attended_output)

    return output


class InterrogativeGRU(nn.Module):
    """نموذج GRU لتصنيف أسماء الاستفهام"""

    def __init__()
    self,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    num_classes: int, dropout: float = 0.3):
    super(InterrogativeGRU, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers

        # طبقة GRU ثنائية الاتجاه
    self.gru = nn.GRU()
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=True,
    dropout=dropout if num_layers > 1 else 0)

        # طبقة الانتباه متعدد الرؤوس
    self.multihead_attention = nn.MultiheadAttention()
    embed_dim=hidden_size * 2, num_heads=4, dropout=dropout, batch_first=True
    )

        # طبقات التصنيف
    self.classifier = nn.Sequential()
    nn.Linear(hidden_size * 2, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, num_classes))

    def forward(self, x):
    def forward(self, x):
    batch_size = x.size(0)

        # تهيئة الحالة المخفية
    h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        # GRU
    gru_out, _ = self.gru(x, h0)

        # الانتباه متعدد الرؤوس
    attn_out, _ = self.multihead_attention(gru_out, gru_out, gru_out)

        # تجميع الإخراج (متوسط)
    pooled_output = torch.mean(attn_out, dim=1)

        # التصنيف
    output = self.classifier(pooled_output)

    return output


class InterrogativeTransformer(nn.Module):
    """نموذج Transformer لتصنيف أسماء الاستفهام"""

    def __init__()
    self,
    input_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    num_classes: int, dropout: float = 0.1):
    super(InterrogativeTransformer, self).__init__()

    self.d_model = d_model

        # تضمين الدخل
    self.input_projection = nn.Linear(input_size, d_model)

        # ترميز الموضع
    self.pos_encoding = PositionalEncoding(d_model, dropout)

        # طبقات Transformer
    encoder_layer = nn.TransformerEncoderLayer()
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=d_model * 4,
    dropout=dropout,
    activation='gelu',
    batch_first=True)

    self.transformer_encoder = nn.TransformerEncoder()
    encoder_layer, num_layers=num_layers
    )

        # رمز التصنيف
    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # طبقة التصنيف النهائية
    self.classifier = nn.Sequential()
    nn.LayerNorm(d_model),
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(d_model // 2, num_classes))

    def forward(self, x):
    def forward(self, x):
    batch_size = x.size(0)

        # إسقاط الدخل
    x = self.input_projection(x)

        # إضافة رمز التصنيف
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)

        # ترميز الموضع
    x = self.pos_encoding(x)

        # Transformer
    transformer_out = self.transformer_encoder(x)

        # أخذ رمز التصنيف
    cls_output = transformer_out[: 0]

        # التصنيف
    output = self.classifier(cls_output)

    return output


class PositionalEncoding(nn.Module):
    """ترميز الموضع للـ Transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):

    super(PositionalEncoding, self).__init__()

    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    div_term = torch.exp()
    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[: 0::2] = torch.sin(position * div_term)
    pe[: 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

    def forward(self, x):
    def forward(self, x):
    x = x + self.pe[: x.size(1), :].transpose(0, 1)
    return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════════════════════
# TRAINING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


def create_synthetic_data()
    processor: InterrogativePhoneticProcessor, num_samples: int = 1200
) -> Tuple[List[np.ndarray], List[int]]:
    """إنشاء بيانات تدريب اصطناعية"""

    # البيانات الأساسية لأسماء الاستفهام
    base_data = {
    "مَن": [["مَنْ"], ["مَ", "نْ"]],
    "مَا": [["مَا"], ["مَ", "ا"]],
    "مَتَى": [["مَ", "تَى"], ["مَ", "تَ", "ى"]],
    "أَيْنَ": [["أَيْ", "نَ"], ["أَ", "يْ", "نَ"]],
    "كَيْفَ": [["كَيْ", "فَ"], ["كَ", "يْ", "فَ"]],
    "كَمْ": [["كَمْ"], ["كَ", "مْ"]],
    "أَيّ": [["أَيّ"], ["أَ", "يّ"]],
    "لِمَاذَا": [["لِ", "مَا", "ذَا"], ["لِ", "مَ", "ا", "ذَا"]],
    "مَاذَا": [["مَا", "ذَا"], ["مَ", "ا", "ذَا"]],
    "أَيَّانَ": [["أَيْ", "يَا", "نَ"], ["أَ", "يَّ", "ا", "نَ"]],
    "أَنَّى": [["أَنْ", "نَى"], ["أَ", "نَّ", "ى"]],
    "لِمَ": [["لِمَ"], ["لِ", "مَ"]],
    "كَأَيِّنْ": [["كَأَيْ", "يِنْ"], ["كَ", "أَ", "يِّ", "نْ"]],
    "أَيُّهَا": [["أَيْ", "يُ", "هَا"], ["أَ", "يُّ", "هَا"]],
    "مَهْمَا": [["مَهْ", "مَا"], ["مَ", "هْ", "مَا"]],
    "أَيْنَمَا": [["أَيْ", "نَ", "مَا"], ["أَ", "يْ", "نَ", "مَا"]],
    "كَيْفَمَا": [["كَيْ", "فَ", "مَا"], ["كَ", "يْ", "فَ", "مَا"]],
    "مَنْ ذَا": [["مَنْ", "ذَا"], ["مَ", "نْ", "ذَا"]],
    }

    X = []
    y = []

    # توليد العينات
    for _ in range(num_samples):
        # اختيار اسم استفهام عشوائي
    pronoun = random.choice(list(base_data.keys()))
    syllables = random.choice(base_data[pronoun])

        # إضافة تشويش طفيف (تنويع)
        if random.random() < 0.15:  # 15% تشويش
    syllables = syllables.copy()

            # تنويعات صوتية طفيفة
            if random.random() < 0.5 and len(len(syllables) -> 1) > 1:
                # تبديل حركة أحياناً
    variations = {'َ': ['ُ', 'ِ'], 'ُ': ['َ', 'ِ'], 'ِ': ['َ', 'ُ']}

                for i, syll in enumerate(syllables):
                    for orig, alts in variations.items():
                        if orig in syll and random.random() < 0.3:
    syllables[i] = syll.replace(orig, random.choice(alts))

        # استخراج الخصائص
    features = processor.extract_features(syllables)
    label = PRONOUN_TO_ID[pronoun]

    X.append(features)
    y.append(label)

    return X, y


def train_model()
    model: nn.Module,
    X: List[np.ndarray],
    y: List[int],
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.001) -> Dict[str, List[float]]:
    """تدريب النموذج"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # تحويل البيانات إلى tensors
    X_tensor = torch.stack([torch.FloatTensor(x) for x in X])
    y_tensor = torch.LongTensor(y)

    # تقسيم البيانات
    train_size = int(0.8 * len(X_tensor))
    train_X, test_X = X_tensor[:train_size], X_tensor[train_size:]
    train_y, test_y = y_tensor[:train_size], y_tensor[train_size:]

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer وCriterion
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    model.train()
    for epoch in range(epochs):
    total_loss = 0
    correct_train = 0
    total_train = 0

        for batch_X, batch_y in train_loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

    optimizer.zero_grad()
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)
    loss.backward()

            # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    total_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total_train += batch_y.size(0)
    correct_train += (predicted == batch_y).sum().item()

    scheduler.step()

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

    history['train_loss'].append(avg_loss)
    history['train_acc'].append(train_accuracy)
    history['test_acc'].append(test_accuracy)

        if (epoch + 1) % 10 == 0:
    print()
    f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f | Test} Acc: {test_accuracy:.4f}}"
    )

    return history


# ═══════════════════════════════════════════════════════════════════════════════════
# INFERENCE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════


class InterrogativePronounInference:
    """نظام الاستنتاج لأسماء الاستفهام"""

    def __init__(self):

    self.processor = InterrogativePhoneticProcessor()
    self.models = {}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # معايير النماذج
    self.model_configs = {
    'lstm': {
    'input_size': self.processor.vocab_size + 12,
    'hidden_size': 64,
    'num_layers': 2,
    'num_classes': len(INTERROGATIVE_PRONOUNS),
    'dropout': 0.3,
    },
    'gru': {
    'input_size': self.processor.vocab_size + 12,
    'hidden_size': 64,
    'num_layers': 2,
    'num_classes': len(INTERROGATIVE_PRONOUNS),
    'dropout': 0.3,
    },
    'transformer': {
    'input_size': self.processor.vocab_size + 12,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'num_classes': len(INTERROGATIVE_PRONOUNS),
    'dropout': 0.1,
    },
    }

    self._initialize_models()

    def _initialize_models(self):
    """تهيئة النماذج"""

        # إنشاء النماذج
    self.models['lstm'] = InterrogativeLSTM(**self.model_configs['lstm'])
    self.models['gru'] = InterrogativeGRU(**self.model_configs['gru'])
    self.models['transformer'] = InterrogativeTransformer()
    **self.model_configs['transformer']
    )

        # نقل النماذج إلى الجهاز
        for model in self.models.values():
    model.to(self.device)

    def train_all_models(self, num_samples: int = 1800):
    """تدريب جميع النماذج"""

    print("🧠 إنشاء بيانات التدريب لأسماء الاستفهام...")
    X, y = create_synthetic_data(self.processor, num_samples)

    results = {}

        for model_name, model in self.models.items():
    print(f"\n🚀 تدريب نموذج {model_name.upper()}...")
    history = train_model(model, X, y, epochs=35)
    results[model_name] = {
    'final_train_acc': history['train_acc'][ 1],
    'final_test_acc': history['test_acc'][ 1],
    'history': history,
    }
    print()
    f"✅ {model_name.upper()} - دقة التدريب: {history['train_acc'][ 1]:.3f}, دقة الاختبار: {history['test_acc'][-1]:.3f}"
    )

    return results

    def predict_syllables()
    self, syllables: List[str], model_type: str = 'transformer'
    ) -> str:
    """التنبؤ باسم الاستفهام من المقاطع"""

        if model_type not in self.models:
    raise ValueError(f"نوع النموذج غير مدعوم: {model_type}")

    model = self.models[model_type]
    model.eval()

        # استخراج الخصائص
    features = self.processor.extract_features(syllables)

        # تحويل إلى tensor
    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # التنبؤ
        with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(outputs, dim=1).item()
    probabilities[0][predicted_class].item()

    predicted_pronoun = INTERROGATIVE_PRONOUNS[predicted_class]

    return predicted_pronoun

    def predict_with_confidence()
    self, syllables: List[str], model_type: str = 'transformer'
    ) -> Dict[str, Any]:
    """التنبؤ مع درجة الثقة والبدائل"""

        if model_type not in self.models:
    raise ValueError(f"نوع النموذج غير مدعوم: {model_type}")

    model = self.models[model_type]
    model.eval()

        # استخراج الخصائص
    features = self.processor.extract_features(syllables)

        # تحويل إلى tensor
    input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # التنبؤ
        with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]

        # ترتيب النتائج حسب الثقة
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    results = {
    'best_prediction': INTERROGATIVE_PRONOUNS[sorted_indices[0].item()],
    'confidence': sorted_probs[0].item(),
    'alternatives': [],
    }

        # إضافة البدائل
        for i in range(min(3, len(sorted_indices))):
    idx = sorted_indices[i].item()
    prob = sorted_probs[i].item()
    results['alternatives'].append()
    {'pronoun': INTERROGATIVE_PRONOUNS[idx], 'confidence': prob}
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════


def main():
    """التشغيل الرئيسي للنظام"""

    print("🧠 نماذج التعلم العميق لأسماء الاستفهام العربية")
    print("=" * 55)

    # إنشاء نظام الاستنتاج
    inference = InterrogativePronounInference()

    # تدريب النماذج
    print("🚀 بدء تدريب النماذج...")
    results = inference.train_all_models(num_samples=1600)

    # عرض النتائج
    print("\n📊 نتائج التدريب:")
    for model_name, result in results.items():
    print(f"   {model_name.upper()}: {result['final_test_acc']:.1%} دقة اختبار}")

    # اختبار التنبؤ
    print("\n🔬 اختبار التنبؤ:")
    test_cases = [
    ["مَنْ"],  # مَن
    ["مَا"],  # ما
    ["مَ", "تَى"],  # متى
    ["أَيْ", "نَ"],  # أين
    ["كَيْ", "فَ"],  # كيف
    ["كَمْ"],  # كم
    ["لِ", "مَا", "ذَا"],  # لماذا
    ["أَيّ"],  # أي
    ]

    for syllables in test_cases:
    print(f"\n   المقاطع: {syllables}")

        for model_type in ['lstm', 'gru', 'transformer']:
    prediction = inference.predict_syllables(syllables, model_type)
    print(f"     {model_type.upper()}: {prediction}")

    # اختبار مفصل
    print("\n🎯 اختبار مفصل (Transformer):")
    detailed_result = inference.predict_with_confidence(["مَ", "تَى"], 'transformer')
    print(f"   أفضل تنبؤ: {detailed_result['best_prediction']}")
    print(f"   الثقة: {detailed_result['confidence']:.3f}")
    print("   البدائل:")
    for alt in detailed_result['alternatives']:
    print(f"     - {alt['pronoun']: {alt['confidence']:.3f}}")

    print("\n✅ اكتمل التدريب والاختبار!")


if __name__ == "__main__":
    main()

