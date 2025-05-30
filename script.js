// Глобальные переменные
let session = null;
let classNames = {};

// Инициализация приложения
async function initApp() {
    try {
        // 1. Загрузка модели
        document.getElementById('results').innerHTML = '<p>Loading model...</p>';
        session = await ort.InferenceSession.create('./model/mobilenetv2.onnx');
        
        // 2. Загрузка классов ImageNet
        const response = await fetch('imagenet_classes.json');
        classNames = await response.json();
        
        // 3. Настройка обработчиков событий
        setupEventListeners();
        
        document.getElementById('results').innerHTML = '<p>Model loaded. Select an image to classify.</p>';
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('results').innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

// Настройка обработчиков событий
function setupEventListeners() {
    // Загрузка файла
    document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
    
    // Загрузка по URL
    document.getElementById('loadFromUrl').addEventListener('click', loadImageFromUrl);
    
    // Загрузка примера
    document.getElementById('loadSample').addEventListener('click', loadSampleImage);
}

// Обработка загрузки изображения
async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const image = new Image();
    image.src = URL.createObjectURL(file);
    await processImage(image);
}

// Загрузка изображения по URL
async function loadImageFromUrl() {
    const url = document.getElementById('imageUrl').value.trim();
    if (!url) return;
    
    try {
        document.getElementById('results').innerHTML = '<p>Loading image...</p>';
        const image = new Image();
        image.crossOrigin = 'Anonymous';
        image.src = url;
        
        image.onload = async () => {
            await processImage(image);
        };
        
        image.onerror = () => {
            throw new Error('Failed to load image from URL');
        };
    } catch (error) {
        console.error('URL load error:', error);
        document.getElementById('results').innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

// Загрузка примера изображения
async function loadSampleImage() {
    try {
        document.getElementById('results').innerHTML = '<p>Loading sample image...</p>';
        const image = new Image();
        image.src = './assets/test_image.jpg';
        await processImage(image);
    } catch (error) {
        console.error('Sample load error:', error);
        document.getElementById('results').innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

// Обработка изображения и запуск модели
async function processImage(image) {
    try {
        // Показать превью
        const preview = document.getElementById('preview');
        preview.src = image.src;
        
        document.getElementById('results').innerHTML = '<p>Processing image...</p>';
        
        // Подождать пока изображение полностью загрузится
        await new Promise((resolve) => {
            if (image.complete) resolve();
            else image.onload = resolve;
        });
        
        // Препроцессинг
        const tensorData = preprocessImage(image);
        
        // Создаем тензор с правильными размерностями [1, 224, 224, 3]
        const inputTensor = new ort.Tensor('float32', tensorData, [1, 224, 224, 3]);
        
        // Получаем имена входов и выходов
        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];
        
        // Запуск модели
        const outputs = await session.run({ [inputName]: inputTensor });
        const predictions = outputs[outputName].data;
        
        // Отображение результатов
        displayPredictions(predictions);
    } catch (error) {
        console.error('Processing error:', error);
        document.getElementById('results').innerHTML = <p class="error">Error: ${error.message}</p>;
    }
}

// Препроцессинг изображения
function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;
    
    // Рисуем изображение с ресайзом (центрированная обрезка)
    const aspect = image.width / image.height;
    let drawWidth, drawHeight, offsetX, offsetY;
    
    if (aspect > 1) {
        drawWidth = 224 * aspect;
        drawHeight = 224;
        offsetX = (224 - drawWidth) / 2;
        offsetY = 0;
    } else {
        drawWidth = 224;
        drawHeight = 224 / aspect;
        offsetX = 0;
        offsetY = (224 - drawHeight) / 2;
    }
    
    ctx.fillStyle = 'rgb(128, 128, 128)'; // Серый фон для padding
    ctx.fillRect(0, 0, 224, 224);
    ctx.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);
    
    // Получаем пиксели в формате NHWC
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const data = imageData.data;
    const tensor = new Float32Array(224 * 224 * 3);
    
    // Преобразование RGB → NHWC (нормализация [0, 1])
    for (let h = 0; h < 224; h++) {
        for (let w = 0; w < 224; w++) {
            const pixelOffset = (h * 224 + w) * 3;
            const srcOffset = (h * 224 + w) * 4;
            
            tensor[pixelOffset] = data[srcOffset] / 255.0;     // R
            tensor[pixelOffset + 1] = data[srcOffset + 1] / 255.0; // G
            tensor[pixelOffset + 2] = data[srcOffset + 2] / 255.0; // B
        }
    }
    
    return tensor;
}

// Отображение предсказаний
function displayPredictions(predictions) {
    // Создаем массив с индексами и вероятностями
    const results = Array.from(predictions)
        .map((prob, index) => ({
            index,
            prob,
            className: classNames[index] || `Class ${index}`
        }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);
    
    // Создаем HTML для результатов
    const resultsHTML = results.map(item => `
        <div class="prediction-item">
            <div class="prediction-class">${item.className}</div>
            <div class="prediction-prob">${(item.prob * 100).toFixed(2)}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${item.prob * 100}%"></div>
            </div>
        </div>
    `).join('');
    
    document.getElementById('results').innerHTML = `
        <h3>Top Predictions:</h3>
        ${resultsHTML}
    `;
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', initApp);