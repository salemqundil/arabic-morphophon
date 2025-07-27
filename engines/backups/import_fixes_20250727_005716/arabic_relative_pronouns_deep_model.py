#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Relative Pronouns Deep Learning Models
===========================================
نماذج التعلم العميق للأسماء الموصولة العربية

Deep learning models for Arabic relative pronouns classification from syllable sequences
using RNN, LSTM, GRU, and Transformer architectures with MFCC audio features.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 1.0.0 - DEEP LEARNING MODELS
Date: 2025-07-24
Encoding: UTF 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from sklearn.metrics import ()
    precision_recall_fscore_support,
    confusion_matrix)
import librosa
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════
# PHONETIC PROCESSOR FOR ARABIC RELATIVE PRONOUNS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicPhoneticProcessor:
    """معالج الفونيمات العربية للأسماء الموصولة"""

    def __init__(self):

        # مخزون الفونيمات العربية الشامل
    self.arabic_phonemes = [
            # أصوات صامتة
    'b',
    't',
    'th',
    'j',
    'h',
    'kh',
    'd',
    'dh',
    'r',
    'z',
    's',
    'sh',
    'ṣ',
    'ḍ',
    'ṭ',
    'ẓ',
    'ʕ',
    'gh',
    'f',
    'q',
    'k',
    'l',
    'm',
    'n',
    'h',
    'w',
    'y',
            # أصوات صائتة قصيرة
    'a',
    'i',
    'u',
            # أصوات صائتة طويلة
    'aa',
    'ii',
    'uu',
            # أصوات مركبة
    'ay',
    'aw',
            # أصوات خاصة
    'ʔ',  # همزة
            # رموز خاصة
    '<PAD>',
    '<UNK>',
    '<START>',
    '<END>',
    ]

    self.phoneme_to_idx = {p: idx for idx, p in enumerate(self.arabic_phonemes)}
    self.idx_to_phoneme = {idx: p for idx, p in enumerate(self.arabic_phonemes)}
    self.vocab_size = len(self.arabic_phonemes)

        # تحويل الأحرف العربية إلى فونيمات
    self.arabic_to_phoneme_map = {
    'ا': 'aa',
    'أ': 'a',
    'إ': 'i',
    'آ': 'aa',
    'ب': 'b',
    'ت': 't',
    'ث': 'th',
    'ج': 'j',
    'ح': 'h',
    'خ': 'kh',
    'د': 'd',
    'ذ': 'dh',
    'ر': 'r',
    'ز': 'z',
    'س': 's',
    'ش': 'sh',
    'ص': 'ṣ',
    'ض': 'ḍ',
    'ط': 'ṭ',
    'ظ': 'ẓ',
    'ع': 'ʕ',
    'غ': 'gh',
    'ف': 'f',
    'ق': 'q',
    'ك': 'k',
    'ل': 'l',
    'م': 'm',
    'ن': 'n',
    'ه': 'h',
    'و': 'w',
    'ي': 'y',
    'ى': 'aa',
    'ة': 'h',
    'ء': 'ʔ',
            # حركات
    'َ': 'a',
    'ِ': 'i',
    'ُ': 'u',
    'ً': 'an',
    'ٍ': 'in',
    'ٌ': 'un',
    'ْ': '',  # سكون
    'ّ': '',  # شدة (مضاعفة)
    }

    logger.info(f"📚 تم تهيئة معالج الفونيمات: {self.vocab_size فونيم}")

    def text_to_phonemes(self, text: str) -> List[str]:
    """تحويل النص العربي إلى فونيمات"""

    phonemes = []
    i = 0

        while i < len(text):
    char = text[i]

            if char in self.arabic_to_phoneme_map:
    phoneme = self.arabic_to_phoneme_map[char]
                if phoneme:  # تجاهل الرموز الفارغة
    phonemes.append(phoneme)

    i += 1

    return phonemes

    def syllables_to_phonemes(self, syllables: List[str]) -> List[str]:
    """تحويل المقاطع إلى فونيمات"""

    all_phonemes = []

        for syllable in syllables:
    syllable_phonemes = self.text_to_phonemes(syllable)
    all_phonemes.extend(syllable_phonemes)

    return all_phonemes

    def encode_phonemes(self, phonemes: List[str]) -> List[int]:
    """تحويل الفونيمات إلى أرقام"""

    return [
    self.phoneme_to_idx.get(p, self.phoneme_to_idx['<UNK>']) for p in phonemes
    ]

    def decode_phonemes(self, indices: List[int]) -> List[str]:
    """تحويل الأرقام إلى فونيمات"""

    return [self.idx_to_phoneme.get(idx, '<UNK>') for idx in indices]

    def pad_sequence(self, sequence: List[int], max_length: int) -> List[int]:
    """إضافة حشو للتسلسل"""

    pad_token = self.phoneme_to_idx['<PAD>']

        if len(sequence) >= max_length:
    return sequence[:max_length]
        else:
    return sequence + [pad_token] * (max_length - len(sequence))


# ═══════════════════════════════════════════════════════════════════════════════════
# AUDIO FEATURE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════════


class MFCCFeatureProcessor:
    """معالج خصائص MFCC للصوت"""

    def __init__(self, n_mfcc: int = 40, sample_rate: int = 16000):

    self.n_mfcc = n_mfcc
    self.sample_rate = sample_rate
    self.n_fft = 2048
    self.hop_length = 512

    def extract_mfcc_features(self, audio_file: str) -> np.ndarray:
    """استخراج خصائص MFCC من ملف صوتي"""

        try:
            # قراءة الملف الصوتي
    y, sr = librosa.load(audio_file, sr=self.sample_rate)

            # استخراج MFCC
    mfccs = librosa.feature.mfcc()
    y=y,
    sr=sr,
    n_mfcc=self.n_mfcc,
    n_fft=self.n_fft,
    hop_length=self.hop_length)

            # تطبيع الخصائص
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / ()
    np.std(mfccs, axis=1, keepdims=True) + 1e 8
    )

    return mfccs.T  # التبديل للحصول على (time_steps, features)

        except Exception as e:
    logger.error(f"خطأ في استخراج MFCC: {e}")
    return np.zeros((100, self.n_mfcc))  # إرجاع خصائص فارغة

    def simulate_mfcc_from_phonemes(self, phonemes: List[str]) -> np.ndarray:
    """محاكاة خصائص MFCC من الفونيمات (للاختبار)"""

        # إنشاء خصائص صوتية محاكاة بناءً على الفونيمات
    sequence_length = len(phonemes) * 10  # 10 إطارات لكل فونيم
    features = np.random.randn(sequence_length, self.n_mfcc) * 0.1

        # إضافة خصائص مميزة لكل فونيم
        for i, phoneme in enumerate(phonemes):
    start_frame = i * 10
    end_frame = (i + 1) * 10

            # خصائص مميزة لكل نوع فونيم
            if phoneme in ['a', 'i', 'u', 'aa', 'ii', 'uu']:  # أصوات صائتة
    features[start_frame:end_frame, :5] += 0.5
            elif phoneme in ['m', 'n']:  # أصوات أنفية
    features[start_frame:end_frame, 5:10] += 0.3
            elif phoneme in ['l', 'r']:  # أصوات سائلة
    features[start_frame:end_frame, 10:15] += 0.4

    return features


# ═══════════════════════════════════════════════════════════════════════════════════
# DATASET CLASSES
# ═══════════════════════════════════════════════════════════════════════════════════


class RelativePronounDataset(Dataset):
    """مجموعة بيانات الأسماء الموصولة"""

    def __init__()
    self,
    data: List[Tuple[List[str], int]],
    processor: ArabicPhoneticProcessor, max_length: int = 20):
    self.data = data
    self.processor = processor
    self.max_length = max_length

    def __len__(self):
    def __len__(self):
    return len(self.data)

    def __getitem__(self, idx):
    def __getitem__(self, idx):
    syllables, label = self.data[idx]

        # تحويل المقاطع إلى فونيمات
    phonemes = self.processor.syllables_to_phonemes(syllables)

        # تحويل إلى أرقام وإضافة حشو
    encoded = self.processor.encode_phonemes(phonemes)
    padded = self.processor.pad_sequence(encoded, self.max_length)

    return torch.tensor(padded, dtype=torch.long), torch.tensor()
    label, dtype=torch.long
    )


class RelativePronounAudioDataset(Dataset):
    """مجموعة بيانات الأسماء الموصولة مع الصوت"""

    def __init__()
    self,
    data: List[Tuple[List[str], str, int]],
    phonetic_processor: ArabicPhoneticProcessor,
    audio_processor: MFCCFeatureProcessor,
    max_phoneme_length: int = 20, max_audio_length: int = 200):
    self.data = data
    self.phonetic_processor = phonetic_processor
    self.audio_processor = audio_processor
    self.max_phoneme_length = max_phoneme_length
    self.max_audio_length = max_audio_length

    def __len__(self):
    def __len__(self):
    return len(self.data)

    def __getitem__(self, idx):
    def __getitem__(self, idx):
    syllables, audio_file, label = self.data[idx]

        # معالجة الفونيمات
    phonemes = self.phonetic_processor.syllables_to_phonemes(syllables)
    encoded_phonemes = self.phonetic_processor.encode_phonemes(phonemes)
    padded_phonemes = self.phonetic_processor.pad_sequence()
    encoded_phonemes, self.max_phoneme_length
    )

        # معالجة الصوت
        if audio_file and Path(audio_file).exists():
    audio_features = self.audio_processor.extract_mfcc_features(audio_file)
        else:
            # محاكاة خصائص صوتية
    audio_features = self.audio_processor.simulate_mfcc_from_phonemes(phonemes)

        # حشو أو قطع الخصائص الصوتية
        if len(audio_features) >= self.max_audio_length:
    audio_features = audio_features[: self.max_audio_length]
        else:
    padding = np.zeros()
    (self.max_audio_length - len(audio_features), audio_features.shape[1])
    )
    audio_features = np.vstack([audio_features, padding])

    return ()
    torch.tensor(padded_phonemes, dtype=torch.long),
    torch.tensor(audio_features, dtype=torch.float32),
    torch.tensor(label, dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════════


class ArabicRelativePronounLSTM(nn.Module):
    """نموذج LSTM للأسماء الموصولة العربية"""

    def __init__()
    self,
    vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    num_classes: int,
    num_layers: int = 2, dropout: float = 0.1):
    super().__init__()

    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    self.lstm = nn.LSTM()
    embed_dim,
    hidden_size,
    num_layers,
    batch_first=True,
    bidirectional=True,
    dropout=dropout)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
    def forward(self, x):
        # x: (batch_size, seq_len)
    embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # LSTM forward pass
    lstm_out, (hidden, cell) = self.lstm(embedded)

        # استخدام آخر إخراج من الاتجاهين
        forward_out = lstm_out[:  1, : self.lstm.hidden_size]
    backward_out = lstm_out[: 0, self.lstm.hidden_size :]
    combined = torch.cat([forward_out, backward_out], dim=1)

        # Dropout وطبقة التصنيف
    dropped = self.dropout(combined)
    output = self.fc(dropped)

    return output


class ArabicRelativePronounTransformer(nn.Module):
    """نموذج Transformer للأسماء الموصولة العربية"""

    def __init__()
    self,
    vocab_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_classes: int,
    dim_feedforward: int = 2048,
    dropout: float = 0.1, max_seq_length: int = 50):
    super().__init__()

    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
    self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

    encoder_layer = nn.TransformerEncoderLayer()
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True)

    self.transformer_encoder = nn.TransformerEncoder()
    encoder_layer, num_encoder_layers
    )
    self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, src_key_padding_mask=None):
    def forward(self, x, src_key_padding_mask=None):
        # x: (batch_size, seq_len)
    embedded = self.embedding(x) * np.sqrt(self.d_model)
    embedded = self.pos_encoding(embedded)

        # إنشاء mask للحشو
        if src_key_padding_mask is None:
    src_key_padding_mask = x == 0  # افتراض أن 0 هو رمز الحشو

        # Transformer encoding
    encoded = self.transformer_encoder()
    embedded, src_key_padding_mask=src_key_padding_mask
    )

        # استخدام متوسط الإخراج (تجاهل الحشو)
    mask = (~src_key_padding_mask).float().unsqueeze( 1)
    pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1)

        # التصنيف
    output = self.classifier(pooled)

    return output


class PositionalEncoding(nn.Module):
    """ترميز الموضع للـ Transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):

    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp()
    torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
    )

    pe[: 0::2] = torch.sin(position * div_term)
    pe[: 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)

    self.register_buffer('pe', pe)

    def forward(self, x):
    def forward(self, x):
    x = x + self.pe[: x.size(1), :].transpose(0, 1)
    return self.dropout(x)


class MultimodalRelativePronounModel(nn.Module):
    """نموذج متعدد الوسائط للأسماء الموصولة (نص + صوت)"""

    def __init__()
    self,
    vocab_size: int,
    text_embed_dim: int,
    audio_input_dim: int,
    hidden_size: int,
    num_classes: int, dropout: float = 0.1):
    super().__init__()

        # معالج النص
    self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
    self.text_lstm = nn.LSTM()
    text_embed_dim, hidden_size, batch_first=True, bidirectional=True
    )

        # معالج الصوت
    self.audio_lstm = nn.LSTM()
    audio_input_dim, hidden_size, batch_first=True, bidirectional=True
    )

        # آلية الانتباه
    self.attention = nn.MultiheadAttention()
    hidden_size * 2, num_heads=8, batch_first=True
    )

        # طبقات التصنيف
    self.dropout = nn.Dropout(dropout)
    self.fusion_layer = nn.Linear(hidden_size * 4, hidden_size)
    self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, text_input, audio_input):
    def forward(self, text_input, audio_input):
        # معالجة النص
    text_embedded = self.text_embedding(text_input)
    text_output, _ = self.text_lstm(text_embedded)
    text_pooled = text_output.mean(dim=1)  # متوسط التسلسل

        # معالجة الصوت
    audio_output, _ = self.audio_lstm(audio_input)
    audio_pooled = audio_output.mean(dim=1)  # متوسط التسلسل

        # دمج الميزات
    combined = torch.cat([text_pooled, audio_pooled], dim=1)

        # طبقة الدمج
    fused = F.relu(self.fusion_layer(combined))
    fused = self.dropout(fused)

        # التصنيف
    output = self.classifier(fused)

    return output


# ═══════════════════════════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════


class RelativePronounTrainer:
    """محرك التدريب والتقييم للأسماء الموصولة"""

    def __init__(self, model, device='cpu'):

    self.model = model.to(device)
    self.device = device
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(model.parameters(), lr=0.001)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau()
    self.optimizer, patience=5, factor=0.5
    )

    self.train_losses = []
    self.val_losses = []
    self.train_accuracies = []
    self.val_accuracies = []

    def train_epoch(self, dataloader):
    """تدريب حقبة واحدة"""

    self.model.train()
    total_loss = 0
    correct = 0
    total = 0

        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 2:  # نموذج النص فقط
    inputs, labels = batch_data
    inputs, labels = inputs.to(self.device), labels.to(self.device)
    outputs = self.model(inputs)
            else:  # نموذج متعدد الوسائط
    text_inputs, audio_inputs, labels = batch_data
    text_inputs = text_inputs.to(self.device)
    audio_inputs = audio_inputs.to(self.device)
    labels = labels.to(self.device)
    outputs = self.model(text_inputs, audio_inputs)

    loss = self.criterion(outputs, labels)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    total_loss += loss.item()
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    self.train_losses.append(avg_loss)
    self.train_accuracies.append(accuracy)

    return avg_loss, accuracy

    def validate(self, dataloader):
    """تقييم النموذج"""

    self.model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

        with torch.no_grad():
            for batch_data in dataloader:
                if len(batch_data) == 2:  # نموذج النص فقط
    inputs, labels = batch_data
    inputs, labels = inputs.to(self.device), labels.to(self.device)
    outputs = self.model(inputs)
                else:  # نموذج متعدد الوسائط
    text_inputs, audio_inputs, labels = batch_data
    text_inputs = text_inputs.to(self.device)
    audio_inputs = audio_inputs.to(self.device)
    labels = labels.to(self.device)
    outputs = self.model(text_inputs, audio_inputs)

    loss = self.criterion(outputs, labels)
    total_loss += loss.item()

    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

    all_predictions.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    self.val_losses.append(avg_loss)
    self.val_accuracies.append(accuracy)

    return avg_loss, accuracy, all_predictions, all_labels

    def train(self, train_dataloader, val_dataloader, epochs=50):
    """التدريب الكامل"""

    logger.info(f"🚀 بدء التدريب لـ {epochs} حقبة")

    best_val_accuracy = 0

        for epoch in range(epochs):
            # تدريب
    train_loss, train_acc = self.train_epoch(train_dataloader)

            # تقييم
    val_loss, val_acc, _, _ = self.validate(val_dataloader)

            # تحديث معدل التعلم
    self.scheduler.step(val_loss)

            # حفظ أفضل نموذج
            if val_acc > best_val_accuracy:
    best_val_accuracy = val_acc
    torch.save(self.model.state_dict(), 'best_relative_pronoun_model.pth')

            if epoch % 10 == 0:
    logger.info()
    f"حقبة {epoch+1}/{epochs}: "
    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
    )

    logger.info(f"✅ اكتمل التدريب! أفضل دقة: {best_val_accuracy:.2f}%")

    return best_val_accuracy

    def evaluate_detailed(self, dataloader, class_names):
    """تقييم مفصل مع مقاييس متقدمة"""

    _, accuracy, predictions, labels = self.validate(dataloader)

        # حساب مقاييس التقييم
    precision, recall, f1, _ = precision_recall_fscore_support()
    labels, predictions, average='weighted'
    )
    conf_matrix = confusion_matrix(labels, predictions)

    results = {
    'accuracy': accuracy / 100,  # تحويل إلى نسبة
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': conf_matrix.tolist(),
    'class_names': class_names,
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════


class RelativePronounInferenceEngine:
    """محرك الاستنتاج للأسماء الموصولة"""

    def __init__()
    self,
    model,
    processor: ArabicPhoneticProcessor,
        class_names: List[str], device='cpu'):
    self.model = model.to(device)
    self.processor = processor
    self.class_names = class_names
    self.device = device

    self.model.eval()

    def predict_from_syllables()
    self, syllables: List[str], max_length: int = 20
    ) -> Dict[str, Any]:
    """التنبؤ من المقاطع"""

        # تحويل المقاطع إلى فونيمات
    phonemes = self.processor.syllables_to_phonemes(syllables)

        # تحويل إلى أرقام وحشو
    encoded = self.processor.encode_phonemes(phonemes)
    padded = self.processor.pad_sequence(encoded, max_length)

        # تحويل إلى tensor
    input_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)

        with torch.no_grad():
    outputs = self.model(input_tensor)
    probabilities = F.softmax(outputs, dim=1)
    predicted_class = outputs.argmax(1).item()
    confidence = probabilities.max().item()

    return {
    'predicted_class': self.class_names[predicted_class],
    'confidence': confidence,
    'class_probabilities': {
    name: prob
                for name, prob in zip(self.class_names, probabilities[0].cpu().numpy())
    },
    'input_syllables': syllables,
    'phonemes': phonemes,
    }

    def predict_batch()
    self, syllables_list: List[List[str]], max_length: int = 20
    ) -> List[Dict[str, Any]]:
    """التنبؤ لمجموعة من المقاطع"""

    results = []

        for syllables in syllables_list:
    result = self.predict_from_syllables(syllables, max_length)
    results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════════


def demonstrate_deep_models():
    """عرض توضيحي للنماذج العميقة"""

    print("🧠 نماذج التعلم العميق للأسماء الموصولة العربية")
    print("=" * 60)

    # إعداد المعالج
    processor = ArabicPhoneticProcessor()

    # إعداد البيانات التجريبية
    training_data = [
    (["الْ", "ذِي"], 0),  # الذي
    (["الْ", "تِي"], 1),  # التي
    (["الْ", "لَ", "ذَا", "نِ"], 2),  # اللذان
    (["الْ", "لَ", "تَا", "نِ"], 3),  # اللتان
    (["الْ", "ذِي", "نَ"], 4),  # الذين
    (["الْ", "لَا", "تِي"], 5),  # اللاتي
    (["مَنْ"], 6),  # مَن
    (["مَا"], 7),  # ما
    ]

    class_names = ["الذي", "التي", "اللذان", "اللتان", "الذين", "اللاتي", "مَن", "ما"]

    # إنشاء مجموعة البيانات
    dataset = RelativePronounDataset(training_data, processor, max_length=15)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # إنشاء النماذج
    models = {
    "LSTM": ArabicRelativePronounLSTM()
    vocab_size=processor.vocab_size,
    embed_dim=64,
    hidden_size=128,
    num_classes=len(class_names),
    num_layers=2),
    "Transformer": ArabicRelativePronounTransformer()
    vocab_size=processor.vocab_size,
    d_model=128,
    nhead=8,
    num_encoder_layers=3,
    num_classes=len(class_names),
    dim_feedforward=512),
    }

    # تدريب واختبار النماذج
    for model_name, model in models.items():
    print(f"\n🔬 اختبار نموذج {model_name}")
    print(" " * 30)

    trainer = RelativePronounTrainer(model)

        # تدريب سريع
        for epoch in range(5):
    train_loss, train_acc = trainer.train_epoch(dataloader)
    print(f"حقبة {epoch+1}: Loss={train_loss:.4f}, Accuracy={train_acc:.2f}%")

        # اختبار الاستنتاج
    inference_engine = RelativePronounInferenceEngine(model, processor, class_names)

    test_cases = [
    ["الْ", "ذِي"],  # الذي
    ["الْ", "تِي"],  # التي
    ["مَنْ"],  # مَن
    ]

    print(f"\n📊 نتائج الاستنتاج لنموذج {model_name}:")
        for syllables in test_cases:
    result = inference_engine.predict_from_syllables(syllables)
    print()
    f"   {syllables} → {result['predicted_class']} (ثقة: {result['confidence']:.2f})"
    )

    print("\n✅ اكتمل العرض التوضيحي للنماذج العميقة!")


if __name__ == "__main__":
    demonstrate_deep_models()

