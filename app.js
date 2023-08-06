const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const { createCanvas, loadImage } = require('canvas');

const app = express();
const PORT = 80;

// Konfigurasi penyimpanan file menggunakan Multer
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Gunakan middleware untuk mengurai data formulir dari permintaan POST
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Muat model TensorFlow.js dari file yang sudah dikonversi
const modelPath = 'tfjsmodel/model.json';
let model;

tf.loadGraphModel(`file://${modelPath}`)
  .then(loadedModel => {
    model = loadedModel;
    console.log('Model TensorFlow.js berhasil dimuat.');
  })
  .catch(err => {
    console.error('Gagal memuat model TensorFlow.js:', err);
  });

// Fungsi deteksi objek
async function detectObjects(imgBuffer) {
  const inputTensor = tf.node.decodeImage(imgBuffer);
  const inputShape = [320, 320]; // Ganti dengan bentuk input model Anda
  const inputArray = inputTensor.toFloat().div(255);
  const inputBatch = inputArray.expandDims();

  // Convert the dtype of the input tensor to int32
  const inputBatchInt32 = inputBatch.toInt();

  const detections = await model.executeAsync({ 'input_tensor': inputBatchInt32 });
  const numDetections = detections[3].dataSync()[0]; // Use dataSync() to get scalar value
  const detectionBoxes = detections[0].dataSync().slice(0, numDetections * 4); // Flatten the tensor
  const detectionScores = detections[1].dataSync().slice(0, numDetections);
  const detectionClasses = detections[2].dataSync().slice(0, numDetections);

  const detectedObjects = [];
  for (let i = 0; i < numDetections; i++) {
    const classIndex = detectionClasses[i];
    const className = class_names[classIndex];
    const score = detectionScores[i];
    const [ymin, xmin, ymax, xmax] = detectionBoxes.slice(i * 4, (i + 1) * 4);

    detectedObjects.push({
      class: className,
      score: score,
      boundingBox: { ymin, xmin, ymax, xmax }
    });
  }

  return detectedObjects;
}

// Konstanta class_names dan class_colors yang Anda gunakan sebelumnya
const class_names = [
  "akar_Patah-mati",
  "batang-akar_patah",
  "batang_pecah",
  "brum akar atau batang",
  "cabang patah mati",
  "daun berubah warna",
  "daun pucuk tunas rusak",
  "gerowong",
  "hilang pucuk dominan",
  "kanker",
  "konk",
  "liana",
  "luka terbuka",
  "percabangan brum berlebihan",
  "resinosis gumosis",
  "sarang rayap"
];

// Endpoint untuk deteksi objek pada gambar
app.post('/detect', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Tidak ada file yang diunggah.' });
  }

  const imgBuffer = req.file.buffer;

  try {
    const detectedObjects = await detectObjects(imgBuffer);

    res.json({ objects: detectedObjects });
  } catch (err) {
    console.error('Gagal melakukan deteksi objek:', err);
    res.status(500).json({ error: 'Terjadi kesalahan saat melakukan deteksi objek.' });
  }
});

// Fungsi untuk menggambar bounding boxes pada gambar
async function drawBoundingBoxes(imgBuffer, detectedObjects) {
  const image = await loadImage(imgBuffer);
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.width, image.height);

  detectedObjects.forEach(obj => {
    const { ymin, xmin, ymax, xmax } = obj.boundingBox;
    const color = getRandomColor();
    const thickness = 2;
    const fontScale = 1;

    ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.lineWidth = thickness;
    ctx.beginPath();
    ctx.rect(xmin * image.width, ymin * image.height, (xmax - xmin) * image.width, (ymax - ymin) * image.height);
    ctx.stroke();
    ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    ctx.fillText(`${obj.class} (${(obj.score * 100).toFixed(2)}%)`, xmin * image.width, ymin * image.height - 10);
  });

  return canvas.toBuffer();
}

// Fungsi untuk mendapatkan warna acak
function getRandomColor() {
  const r = Math.floor(Math.random() * 256);
  const g = Math.floor(Math.random() * 256);
  const b = Math.floor(Math.random() * 256);
  return [r, g, b];
}



// Jalankan server
app.listen(PORT, () => {
  console.log(`Server berjalan di http://localhost:${PORT}`);
});