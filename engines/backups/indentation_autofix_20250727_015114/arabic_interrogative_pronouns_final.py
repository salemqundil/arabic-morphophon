#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Arabic Interrogative Pronouns Deep Learning System
=======================================================
النظام النهائي للتعلم العميق لأسماء الاستفهام العربية

Simplified and optimized version for Arabic interrogative pronouns classification.

Author: Arabic NLP Expert Team - GitHub Copilot
Version: 3.0.0 - FINAL SYSTEM
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
import time
from typing import Dict, List, Tuple, Any
import json

# إعداد البذرة للتكرار
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATIONS
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

# خريطة المقاطع للأسماء
SYLLABLE_TO_PRONOUN = {
    ("مَنْ"): "مَن",
    ("مَا"): "مَا",
    ("مَ", "تَى"): "مَتَى",
    ("أَيْ", "نَ"): "أَيْنَ",
    ("كَيْ", "فَ"): "كَيْفَ",
    ("كَمْ"): "كَمْ",
    ("أَيّ"): "أَيّ",
    ("لِ", "مَاذَا"): "لِمَاذَا",
    ("لِ", "مَا", "ذَا"): "لِمَاذَا",
    ("مَا", "ذَا"): "مَاذَا",
    ("أَيَّ", "ا", "نَ"): "أَيَّانَ",
    ("أَنَّ", "ى"): "أَنَّى",
    ("لِ", "مَ"): "لِمَ",
    ("كَأَيْ", "يِنْ"): "كَأَيِّنْ",
    ("أَيُّ", "هَا"): "أَيُّهَا",
    ("مَهْ", "مَا"): "مَهْمَا",
    ("أَيْ", "نَ", "مَا"): "أَيْنَمَا",
    ("كَيْ", "فَ", "مَا"): "كَيْفَمَا",
    ("مَنْ", "ذَا"): "مَنْ ذَا",
}


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════════════


class OptimizedLSTM(nn.Module):
    """نموذج LSTM محسن لأسماء الاستفهام"""

    def __init__()
        self, vocab_size: int = 60, hidden_size: int = 128, num_classes: int = 18
    ):
        super(OptimizedLSTM, self).__init__()

        self.hidden_size = hidden_size

        # طبقة التضمين
        self.embedding = nn.Embedding(vocab_size, 64)

        # طبقة LSTM ثنائية الاتجاه
        self.lstm = nn.LSTM()
            input_size=64,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3)

        # طبقة الانتباه
        self.attention = nn.Sequential()
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1))

        # طبقات التصنيف
        self.classifier = nn.Sequential()
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes))

    def forward(self, x):
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)

        # LSTM
        lstm_out, _ = self.lstm(embedded)

        # آلية الانتباه
        attention_weights = self.attention(lstm_out)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)

        # التصنيف
        output = self.classifier(attended_output)

        return output


# ═══════════════════════════════════════════════════════════════════════════════════
# ENHANCED GRU MODEL
# ═══════════════════════════════════════════════════════════════════════════════════


class OptimizedGRU(nn.Module):
    """نموذج GRU محسن لأسماء الاستفهام"""

    def __init__()
        self, vocab_size: int = 60, hidden_size: int = 128, num_classes: int = 18
    ):
        super(OptimizedGRU, self).__init__()

        self.hidden_size = hidden_size

        # طبقة التضمين
        self.embedding = nn.Embedding(vocab_size, 64)

        # طبقة GRU ثنائية الاتجاه
        self.gru = nn.GRU()
            input_size=64,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3)

        # انتباه متعدد الرؤوس
        self.multihead_attention = nn.MultiheadAttention()
            embed_dim=hidden_size * 2, num_heads=8, dropout=0.2, batch_first=True
        )

        # طبقات التصنيف
        self.classifier = nn.Sequential()
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)

        # GRU
        gru_out, _ = self.gru(embedded)

        # انتباه متعدد الرؤوس
        attn_out, _ = self.multihead_attention(gru_out, gru_out, gru_out)

        # تجميع الإخراج
        pooled_output = torch.mean(attn_out, dim=1)

        # التصنيف
        output = self.classifier(pooled_output)

        return output


# ═══════════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════════


class SimplePhoneticProcessor:
    """معالج صوتي مبسط"""

    def __init__(self):

        self.char_to_id = {}
        self.id_to_char = {}
        self._build_vocabulary()

    def _build_vocabulary(self):
        """بناء المفردات من المقاطع"""

        chars = set()

        # جمع جميع الأحرف من المقاطع
        for syllables in SYLLABLE_TO_PRONOUN.keys():
            for syllable in syllables:
                chars.update(syllable)

        # إضافة رموز خاصة
        chars.update(['<PAD>', '<UNK>', '<START>', '<END>'])

        # بناء الخرائط
        for i, char in enumerate(sorted(chars)):
            self.char_to_id[char] = i
            self.id_to_char[i] = char

        self.vocab_size = len(self.char_to_id)

    def encode_syllables(self, syllables: List[str], max_length: int = 10) -> List[int]:
        """ترميز المقاطع إلى أرقام"""

        encoded = [self.char_to_id.get('<START>', 0)]

        for syllable in syllables:
            for char in syllable:
                char_id = self.char_to_id.get(char, self.char_to_id.get('<UNK>', 1))
                encoded.append(char_id)

        encoded.append(self.char_to_id.get('<END>', 3))

        # padding أو قص
        if len(encoded) < max_length:
            encoded.extend()
                [self.char_to_id.get('<PAD>', 2)] * (max_length - len(encoded))
            )
        else:
            encoded = encoded[:max_length]

        return encoded


def create_training_data()
    processor: SimplePhoneticProcessor, num_samples: int = 3000
) -> Tuple[List[List[int]], List[int]]:
    """إنشاء بيانات التدريب"""

    X = []
    y = []

    # البيانات الأساسية مع تنويعات
    data_variants = {
        "مَن": [["مَنْ"], ["مَ", "نْ"], ["مِنْ"], ["مُنْ"]],
        "مَا": [["مَا"], ["مَ", "ا"], ["مُا"], ["مِا"]],
        "مَتَى": [["مَ", "تَى"], ["مَ", "تَ", "ى"], ["مِ", "تَى"]],
        "أَيْنَ": [["أَيْ", "نَ"], ["أَ", "يْ", "نَ"], ["أُيْ", "نَ"]],
        "كَيْفَ": [["كَيْ", "فَ"], ["كَ", "يْ", "فَ"], ["كُيْ", "فَ"]],
        "كَمْ": [["كَمْ"], ["كَ", "مْ"], ["كُمْ"]],
        "أَيّ": [["أَيّ"], ["أَ", "يّ"], ["أُيّ"]],
        "لِمَاذَا": [["لِ", "مَا", "ذَا"], ["لِ", "مَ", "ا", "ذَا"]],
        "مَاذَا": [["مَا", "ذَا"], ["مَ", "ا", "ذَا"]],
        "أَيَّانَ": [["أَيَّ", "ا", "نَ"], ["أَ", "يَّ", "ا", "نَ"]],
        "أَنَّى": [["أَنَّ", "ى"], ["أَ", "نَّ", "ى"]],
        "لِمَ": [["لِمَ"], ["لِ", "مَ"]],
        "كَأَيِّنْ": [["كَأَيْ", "يِنْ"], ["كَ", "أَ", "يِّ", "نْ"]],
        "أَيُّهَا": [["أَيُّ", "هَا"], ["أَ", "يُّ", "هَا"]],
        "مَهْمَا": [["مَهْ", "مَا"], ["مَ", "هْ", "مَا"]],
        "أَيْنَمَا": [["أَيْ", "نَ", "مَا"], ["أَ", "يْ", "نَ", "مَا"]],
        "كَيْفَمَا": [["كَيْ", "فَ", "مَا"], ["كَ", "يْ", "فَ", "مَا"]],
        "مَنْ ذَا": [["مَنْ", "ذَا"], ["مَ", "نْ", "ذَا"]],
    }

    samples_per_pronoun = num_samples // len(data_variants)

    for pronoun, variants in data_variants.items():
        pronoun_id = PRONOUN_TO_ID[pronoun]

        for _ in range(samples_per_pronoun):
            syllables = random.choice(variants)
            encoded = processor.encode_syllables(syllables)

            X.append(encoded)
            y.append(pronoun_id)

    return X, y


def train_model()
    model: nn.Module, X: List[List[int]], y: List[int], epochs: int = 30
) -> Dict[str, List[float]]:
    """تدريب النموذج"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # تحويل البيانات
    X_tensor = torch.LongTensor(X)
    y_tensor = torch.LongTensor(y)

    # تقسيم البيانات
    train_size = int(0.8 * len(X_tensor))
    train_X, test_X = X_tensor[:train_size], X_tensor[train_size:]
    train_y, test_y = y_tensor[:train_size], y_tensor[train_size:]

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Optimizer والمعيار
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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


class FinalInterrogativeSystem:
    """النظام النهائي لأسماء الاستفهام"""

    def __init__(self):

        self.processor = SimplePhoneticProcessor()
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # إنشاء النماذج
        self.models['lstm'] = OptimizedLSTM()
            vocab_size=self.processor.vocab_size,
            hidden_size=128,
            num_classes=len(INTERROGATIVE_PRONOUNS))

        self.models['gru'] = OptimizedGRU()
            vocab_size=self.processor.vocab_size,
            hidden_size=128,
            num_classes=len(INTERROGATIVE_PRONOUNS))

        # نقل النماذج إلى الجهاز
        for model in self.models.values():
            model.to(self.device)

    def train_all_models(self, num_samples: int = 4000):
        """تدريب جميع النماذج"""

        print("🧠 إنشاء بيانات التدريب...")
        X, y = create_training_data(self.processor, num_samples)

        results = {}

        for model_name, model in self.models.items():
            print(f"\n🚀 تدريب نموذج {model_name.upper()}...")
            history = train_model(model, X, y, epochs=25)

            results[model_name] = {
                'final_train_acc': history['train_acc'][ 1],
                'final_test_acc': history['test_acc'][ 1],
                'best_test_acc': max(history['test_acc']),
                'history': history,
            }

            print()
                f"✅ {model_name.upper() - أفضل دقة} اختبار: {max(history['test_acc']):.3f}}"
            )

        return results

    def predict_syllables()
        self, syllables: List[str], model_type: str = 'gru'
    ) -> Dict[str, Any]:
        """التنبؤ باسم الاستفهام من المقاطع"""

        if model_type not in self.models:
            model_type = 'gru'  # افتراضي

        model = self.models[model_type]
        model.eval()

        # ترميز المقاطع
        encoded = self.processor.encode_syllables(syllables)
        input_tensor = torch.LongTensor(encoded).unsqueeze(0).to(self.device)

        # التنبؤ
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        # ترتيب النتائج
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

    def comprehensive_test(self):
        """اختبار شامل للنظام"""

        print("\n🔬 تشغيل الاختبار الشامل:")

        test_cases = [
            ["مَنْ"],  # مَن
            ["مَا"],  # ما
            ["مَ", "تَى"],  # متى
            ["أَيْ", "نَ"],  # أين
            ["كَيْ", "فَ"],  # كيف
            ["كَمْ"],  # كم
            ["أَيّ"],  # أي
            ["لِ", "مَا", "ذَا"],  # لماذا
            ["مَا", "ذَا"],  # ماذا
            ["لِ", "مَ"],  # لم
        ]

        expected = ["مَن", "مَا", "مَتَى", "أَيْنَ", "كَيْفَ", "كَمْ", "أَيّ", "لِمَاذَا", "مَاذَا", "لِمَ"]

        for model_name in ['lstm', 'gru']:
            print(f"\n   📊 اختبار نموذج {model_name.upper()}:")
            correct = 0

            for i, (syllables, exp) in enumerate(zip(test_cases, expected)):
                result = self.predict_syllables(syllables, model_name)
                prediction = result['best_prediction']
                confidence = result['confidence']

                is_correct = prediction == exp
                if is_correct:
                    correct += 1

                status = "✅" if is_correct else "❌"
                print()
                    f"     {status} {syllables} → {prediction} (ثقة: {confidence:.3f}) [متوقع: {exp}]"
                )

            accuracy = correct / len(test_cases)
            print()
                f"     📈 الدقة الإجمالية: {accuracy:.1%} ({correct}/{len(test_cases)})"
            )


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════


def main():
    """التشغيل الرئيسي للنظام النهائي"""

    print("🧠 النظام النهائي للتعلم العميق - أسماء الاستفهام العربية")
    print("=" * 65)

    # إنشاء النظام
    system = FinalInterrogativeSystem()

    # تدريب النماذج
    print("🚀 بدء تدريب النماذج النهائية...")
    start_time = time.time()

    results = system.train_all_models(num_samples=3500)

    training_time = time.time() - start_time

    # عرض النتائج
    print("\n📊 نتائج التدريب النهائية:")
    print(f"   ⏱️  زمن التدريب: {training_time:.1f} ثانية")

    for model_name, result in results.items():
        print()
            f"   🏆 {model_name.upper()}: {result['best_test_acc']:.1% أفضل} دقة اختبار}"
        )

    # اختبار شامل
    system.comprehensive_test()

    # اختبار تفصيلي
    print("\n🎯 اختبار تفصيلي:")
    detailed_test = system.predict_syllables(["لِ", "مَا", "ذَا"], 'gru')
    print("   المقاطع: ['لِ', 'مَا', 'ذَا']")
    print(f"   التنبؤ: {detailed_test['best_prediction']}")
    print(f"   الثقة: {detailed_test['confidence']:.3f}")
    print("   البدائل:")
    for alt in detailed_test['alternatives']:
        print(f"     - {alt['pronoun']: {alt['confidence']:.3f}}")

    # حفظ النتائج
    with open('final_interrogative_results.json', 'w', encoding='utf 8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print("\n✅ اكتمل النظام النهائي!")
    print("💾 النتائج محفوظة في: final_interrogative_results.json")


if __name__ == "__main__":
    main()

