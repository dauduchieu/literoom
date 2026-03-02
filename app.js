const K = 5;

const toneNames = [
    "Blacks",
    "Shadows",
    "Midtones",
    "Highlights",
    "Whites"
];

let paramModel, curveModel;
let fullImageTensor = null;
let lastOutput = null;

let luma = new Float32Array(K);
let color = new Float32Array(K * 3);

const img = document.getElementById("inputImg");
const upload = document.getElementById("upload");
const canvasOut = document.getElementById("outputCanvas");
const controls = document.getElementById("controls");

function showLoading(flag) {
    document.getElementById("loading").style.display =
        flag ? "flex" : "none";
}

// ===== LOAD MODELS =====
async function loadModels() {
    showLoading(true);
    paramModel = await tf.loadGraphModel("param_tfjs/model.json");
    curveModel = await tf.loadGraphModel("curve_tfjs/model.json");
    showLoading(false);
}
loadModels();

// ===== UPLOAD =====
upload.onchange = e => {
    img.src = URL.createObjectURL(e.target.files[0]);
};

img.onload = async () => {
    showLoading(true);

    fullImageTensor?.dispose();

    // Full resolution tensor
    fullImageTensor = tf.browser.fromPixels(img)
        .toFloat()
        .div(255)
        .expandDims(0);

    drawHistogram(img, "histBefore");

    await runParamModel();
    showLoading(false);
};

// ===== PARAM MODEL =====
async function runParamModel() {

    const input224 = tf.browser.fromPixels(img)
        .toFloat().div(255)
        .resizeBilinear([224, 224])
        .expandDims(0);

    const params = paramModel.predict(input224);
    const data = await params.data();

    luma.set(data.slice(0, K));
    color.set(data.slice(K));

    input224.dispose();
    params.dispose();

    createSliders();
    await renderImage();
}

// ===== SLIDERS =====
function createGroup(title, array, offset) {
    const div = document.createElement("div");
    div.className = "slider-group";
    div.innerHTML = "<b>" + title + "</b>";

    for (let i = 0; i < K; i++) {
        const row = document.createElement("div");
        row.className = "slider-row";

        // Tone label
        const label = document.createElement("label");
        label.textContent = toneNames[i];

        // Slider
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = 0;
        slider.max = 1;
        slider.step = 0.01;
        slider.value = array[offset + i];

        // Value text
        const valueText = document.createElement("span");
        valueText.textContent = parseFloat(slider.value).toFixed(2);

        slider.oninput = () => {
            const v = parseFloat(slider.value);
            array[offset + i] = v;
            valueText.textContent = v.toFixed(2);
            debounceRender();
        };

        row.appendChild(label);
        row.appendChild(slider);
        row.appendChild(valueText);
        div.appendChild(row);
    }

    controls.appendChild(div);
}

function createSliders() {
    controls.innerHTML = "";
    createGroup("Luma", luma, 0);
    createGroup("Red", color, 0);
    createGroup("Green", color, K);
    createGroup("Blue", color, 2 * K);
}

// ===== RENDER FULL RES =====
let renderTimer = null;
function debounceRender() {
    if (renderTimer) clearTimeout(renderTimer);
    renderTimer = setTimeout(renderImage, 80);
}

async function renderImage() {
    if (!fullImageTensor) return;

    showLoading(true);

    const lumaT = tf.tensor(luma, [1, K]);
    const colorT = tf.tensor(color, [1, K, 3]);

    const output = curveModel.predict([
        lumaT,
        colorT,
        fullImageTensor
    ]);

    lastOutput?.dispose();
    lastOutput = output;

    const h = output.shape[1];
    const w = output.shape[2];

    canvasOut.width = w;
    canvasOut.height = h;

    await tf.browser.toPixels(output.squeeze(), canvasOut);
    drawHistogram(canvasOut, "histAfter");

    tf.dispose([lumaT, colorT]);

    showLoading(false);
}

// ===== HISTOGRAM =====
function drawHistogram(source, canvasId) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");

    const tmp = document.createElement("canvas");
    tmp.width = source.width;
    tmp.height = source.height;
    const tctx = tmp.getContext("2d");
    tctx.drawImage(source, 0, 0);

    const data = tctx.getImageData(0, 0, tmp.width, tmp.height).data;
    const hist = new Array(256).fill(0);

    for (let i = 0; i < data.length; i += 4) {
        const v = (data[i] + data[i + 1] + data[i + 2]) / 3;
        hist[Math.floor(v)]++;
    }

    const max = Math.max(...hist);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < 256; i++) {
        const h = hist[i] / max * canvas.height;
        ctx.fillRect(i, canvas.height - h, 1, h);
    }
}

// ===== SAVE =====
document.getElementById("saveBtn").onclick = () => {
    if (!lastOutput) return;

    const link = document.createElement("a");
    link.download = "result.png";
    link.href = canvasOut.toDataURL("image/png");
    link.click();
};