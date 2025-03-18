import argparse
from ultralytics import YOLO
import cv2
import numpy as np

def load_image_with_padding(image, stride=32):
    # Вычисляем размеры, кратные 32
    height, width, _ = image.shape
    new_height = (height + stride - 1) // stride * stride
    new_width = (width + stride - 1) // stride * stride

    # Добавляем отступы
    padded_image = np.full((new_height, new_width, 3), 114, dtype=np.uint8)  # Цвет заполнения (114 - серый)
    padded_image[:height, :width] = image  # Вставляем изображение в верхний левый угол

    return padded_image

def resize_to_640x360(image):
    return cv2.resize(image, (900, 600), interpolation=cv2.INTER_LINEAR)

def draw_boxes(original_image, detections, model):
    # Рисуем рамки и добавляем метки на оригинальном изображении
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты рамки
        confidence = box.conf[0] * 100  # Уверенность в процентах
        class_id = int(box.cls[0])  # Идентификатор класса
        label = f"{model.names[class_id]}: {confidence:.2f}%"  # Название класса и уверенность

        # Рисуем прямоугольник вокруг найденного объекта
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # Добавляем текст с названием класса и уверенностью
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return original_image

def run_detection(model_path, source, num_iterations=5):
    # Загружаем модель YOLO
    model = YOLO(model_path)

    # Если источник — изображение
    if source.endswith(('.jpg', '.jpeg', '.png')):
        # Загружаем и изменяем размер изображения
        original_image = cv2.imread(source)
        resized_image = resize_to_640x360(original_image)
        padded_image = load_image_with_padding(resized_image)

        for i in range(num_iterations):
            # Выполняем детекцию на изображении
            print(f"Итерация {i + 1}/{num_iterations}...")
            results = model(padded_image, conf=0.1)  # Порог уверенности 10%
            annotated_image = draw_boxes(resized_image.copy(), results[0], model)  # Рисуем рамки и метки

            # Отображаем изображение с аннотациями
            cv2.imshow(f'Annotated Image - Итерация {i + 1}', annotated_image)
            key = cv2.waitKey(0)  # Ждем нажатия любой клавиши
            if key == 27:  # Если нажата клавиша ESC, прерываем цикл
                break

        cv2.destroyAllWindows()  # Закрываем все окна

    # Если источник — видео
    elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Не удалось открыть видео.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = resize_to_640x360(frame)
            padded_frame = load_image_with_padding(resized_frame)
            results = model(padded_frame, conf=0.1)
            annotated_frame = draw_boxes(resized_frame.copy(), results[0], model)
            cv2.imshow('Annotated Video', annotated_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Выход по ESC
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Для данного скрипта поддерживаются только изображения и видео.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection Script")
    parser.add_argument('model', type=str, help='Путь к модели (.pt)')
    parser.add_argument('source', type=str, help='Путь к изображению или видео')
    parser.add_argument('--iterations', type=int, default=5, help='Количество итераций обработки изображения')

    args = parser.parse_args()

    run_detection(args.model, args.source, args.iterations)
