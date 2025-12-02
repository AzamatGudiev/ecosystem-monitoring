#!/usr/bin/env python3
"""
Скрипт для классификации звуков птиц.
Можно подавать URL с Xeno-Canto или локальный файл.

Использование:
    python classify.py --url "https://xeno-canto.org/..."
    python classify.py --file "recording.mp3"
"""

import argparse
import os
import sys
import tempfile
import requests

# Пороги для оповещений
LOW_CONFIDENCE_THRESHOLD = 0.5
UNKNOWN_SOUND_THRESHOLD = 0.3

# Редкие виды (можно дополнить)
RARE_SPECIES = [
    "spotted owl",
    "california condor",
    "whooping crane",
    "ivory-billed woodpecker"
]


def download_audio(url, output_path):
    """Скачивает аудио по URL"""
    print(f"Скачиваю: {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"Сохранено: {output_path}")
        return True
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        return False


def convert_to_wav(input_path, output_path):
    """Конвертирует аудио в wav"""
    import librosa
    import soundfile as sf
    
    print(f"Конвертирую в wav...")
    y, sr = librosa.load(input_path, sr=None)
    sf.write(output_path, y, sr)
    print(f"Конвертировано: {output_path}")
    return output_path


def classify_audio(file_path, classifier):
    """Классифицирует аудио и возвращает результаты"""
    print(f"Классифицирую...")
    preds = classifier(file_path)
    preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
    return preds_sorted


def check_alerts(pred_label, pred_score):
    """Проверяет нужно ли генерировать оповещение"""
    alerts = []
    
    # Редкий вид
    if pred_label.lower() in [s.lower() for s in RARE_SPECIES]:
        alerts.append(f"🦉 RARE_SPECIES: Обнаружен редкий вид!")
    
    # Низкая уверенность
    if pred_score < LOW_CONFIDENCE_THRESHOLD:
        alerts.append(f"⚠️ LOW_CONFIDENCE: Модель не уверена в результате")
    
    # Возможно неизвестный звук
    if pred_score < UNKNOWN_SOUND_THRESHOLD:
        alerts.append(f"❓ UNKNOWN_SOUND: Возможно неизвестный вид или шум")
    
    return alerts


def print_results(predictions, top_n=3):
    """Выводит результаты классификации"""
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 50)
    
    top1 = predictions[0]
    print(f"\n🐦 Вид: {top1['label']}")
    print(f"📊 Уверенность: {top1['score']:.1%}")
    
    # Проверяем оповещения
    alerts = check_alerts(top1['label'], top1['score'])
    if alerts:
        print("\n" + "-" * 30)
        print("ОПОВЕЩЕНИЯ:")
        for alert in alerts:
            print(f"  {alert}")
    
    # Топ-3 предсказания
    print("\n" + "-" * 30)
    print(f"Топ-{top_n} предсказания:")
    for i, pred in enumerate(predictions[:top_n], 1):
        print(f"  {i}. {pred['label']} ({pred['score']:.1%})")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Классификация звуков птиц")
    parser.add_argument("--url", help="URL аудиофайла для скачивания")
    parser.add_argument("--file", help="Путь к локальному аудиофайлу")
    args = parser.parse_args()
    
    # Проверяем аргументы
    if not args.url and not args.file:
        print("Ошибка: укажи --url или --file")
        print("Пример: python classify.py --file recording.mp3")
        sys.exit(1)
    
    # Загружаем модель
    print("Загружаю модель...")
    from transformers import pipeline
    classifier = pipeline("audio-classification", model="dima806/bird_sounds_classification")
    print("Модель загружена!")
    
    # Определяем путь к файлу
    if args.url:
        # Скачиваем файл во временную папку
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.mp3")
        if not download_audio(args.url, audio_path):
            sys.exit(1)
    else:
        audio_path = args.file
        if not os.path.exists(audio_path):
            print(f"Ошибка: файл не найден: {audio_path}")
            sys.exit(1)
    
    # Конвертируем в wav если нужно
    if audio_path.endswith(".mp3"):
        wav_path = audio_path.replace(".mp3", ".wav")
        convert_to_wav(audio_path, wav_path)
        audio_path = wav_path
    
    # Классифицируем
    predictions = classify_audio(audio_path, classifier)
    
    # Выводим результаты
    print_results(predictions)


if __name__ == "__main__":
    main()
