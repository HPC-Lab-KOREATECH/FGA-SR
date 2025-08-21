const toLoadedImage = (src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
});

const waitImage = (image) => new Promise((resolve) => {
    if (image.complete) {
        resolve(image);
    } else {
        const onLoad = () => {
            image.removeEventListener('load', onLoad);
            resolve(image);
        };
        image.addEventListener('load', onLoad);
    }
});

const waitImages = (images) => Promise.all(images.map(waitImage));

const getImageCanvas = (image) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', {willReadFrequently: true});
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0);

    return {canvas, ctx};
}

const extractImageData = (image) => {
    const {canvas, ctx} = getImageCanvas(image);
    const imageData = ctx.getImageData(0, 0, image.naturalWidth, image.naturalHeight);
    return imageData;
}

const getDiffImage = (baseImage, targetImage) => {
    const {canvas, ctx} = getImageCanvas(baseImage);
    const baseData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(targetImage, 0, 0);
    const compareData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const diffData = ctx.createImageData(canvas.width, canvas.height);

    for (let i = 0; i < diffData.data.length; i += 4) {
        diffData.data[i] = Math.abs(baseData.data[i] - compareData.data[i]); // R
        diffData.data[i + 1] = Math.abs(baseData.data[i + 1] - compareData.data[i + 1]); // G
        diffData.data[i + 2] = Math.abs(baseData.data[i + 2] - compareData.data[i + 2]); // B
        diffData.data[i + 3] = 255;
    }

    ctx.putImageData(diffData, 0, 0);
    const diffImage = new Image();
    diffImage.src = canvas.toDataURL();
    return diffImage;
}

const getPSNRImage = ({totalPSNR, minPSNR, maxPSNR, psnrs, gridWidth, gridHeight}, baseImage) => {
    const {canvas, ctx} = getImageCanvas(baseImage);

    psnrs.forEach((row, y) => {
        row.forEach((psnr, x) => {
            if (psnr === Infinity) {
                ctx.fillStyle = `rgba(0, 0, 255, 1)`;
            } else if (psnr > totalPSNR) {
                const alpha = (psnr - totalPSNR) / (maxPSNR - totalPSNR);
                ctx.fillStyle = `rgba(0, 255, 0, ${alpha})`;
            } else if (psnr < totalPSNR) {
                const alpha = (totalPSNR - psnr) / (totalPSNR - minPSNR);
                ctx.fillStyle = `rgba(255, 0, 0, ${alpha})`;
            }
            const rectX = Math.round(x * gridWidth);
            const rectY = Math.round(y * gridHeight);
            const rectWidth = Math.ceil(gridWidth);
            const rectHeight = Math.ceil(gridHeight);

            ctx.fillRect(rectX, rectY, rectWidth, rectHeight);
        });
    });

    const psnrImage = new Image();
    psnrImage.src = canvas.toDataURL();
    return psnrImage;
}

function calculatePatchPSNR(gtData, targetData, width, height, startX, startY, patchWidth, patchHeight) {
    let mse = 0;
    let numValidPixels = 0;

    const endX = Math.min(startX + patchWidth, width);
    const endY = Math.min(startY + patchHeight, height);

    for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
            const index = (y * width + x) * 4;
            for (let j = 0; j < 3; j++) {
                const diff = gtData[index + j] - targetData[index + j];
                mse += diff * diff;
            }
            numValidPixels++;
        }
    }

    mse /= (numValidPixels * 3);

    if (mse === 0) return Infinity;

    const maxPixelValue = 255;
    const psnr = 10 * Math.log10((maxPixelValue ** 2) / mse);
    return psnr;
}

const calculatePSNR = (gtImage, targetImage, patchWidth, patchHeight) => {
    const gtImageData = extractImageData(gtImage);
    const targetImageData = extractImageData(targetImage);

    patchWidth = patchWidth || gtImage.naturalWidth;
    patchHeight = patchHeight || gtImage.naturalHeight;

    const rows = Math.ceil(gtImage.naturalHeight / patchHeight);
    const cols = Math.ceil(gtImage.naturalWidth / patchWidth);
    const psnrs = [];
    psnrs.max = -Infinity;
    psnrs.realMax = -Infinity;
    psnrs.min = Infinity;
    for (let y = 0; y < rows; y++) {
        const row = [];
        for (let x = 0; x < cols; x++) {
            const psnr = calculatePatchPSNR(gtImageData.data, targetImageData.data, gtImageData.width, gtImageData.height, x * patchWidth, y * patchHeight, patchWidth, patchHeight);
            psnrs.max = Math.max(psnrs.max, psnr);
            psnrs.realMax = psnr === Infinity ? psnrs.realMax : Math.max(psnrs.realMax, psnr);
            psnrs.min = Math.min(psnrs.min, psnr);
            row.push(psnr);
        }
        psnrs.push(row);
    }

    return psnrs;
};

const calculateSSIM = (gtImage, targetImage, windowSize = 8) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', {willReadFrequently: true});

    canvas.width = gtImage.naturalWidth;
    canvas.height = gtImage.naturalHeight;

    const C1 = (0.01 * 255) ** 2;
    const C2 = (0.03 * 255) ** 2;

    function calculateStats(data, x, y, width) {
        let sum = 0, sumSq = 0, n = windowSize * windowSize;
        for (let i = 0; i < windowSize; i++) {
            for (let j = 0; j < windowSize; j++) {
                const idx = ((y + i) * width + (x + j)) * 4;
                const luminance = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]; // RGB to luminance
                sum += luminance;
                sumSq += luminance * luminance;
            }
        }
        const mean = sum / n;
        const variance = Math.max(0, (sumSq / n) - (mean * mean));  // Ensure variance is not negative
        return {mean, variance, n};
    }

    ctx.drawImage(gtImage, 0, 0);
    const gtData = ctx.getImageData(0, 0, gtImage.naturalWidth, gtImage.naturalHeight).data;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(targetImage, 0, 0);
    const targetData = ctx.getImageData(0, 0, targetImage.naturalWidth, targetImage.naturalHeight).data;

    let ssim = 0;
    let windows = 0;

    for (let y = 0; y <= gtImage.naturalHeight - windowSize; y += windowSize) {
        for (let x = 0; x <= gtImage.naturalWidth - windowSize; x += windowSize) {
            const gtStats = calculateStats(gtData, x, y, gtImage.naturalWidth);
            const targetStats = calculateStats(targetData, x, y, gtImage.naturalWidth);

            const numerator = (2 * gtStats.mean * targetStats.mean + C1) * (2 * Math.sqrt(Math.max(0, gtStats.variance)) * Math.sqrt(Math.max(0, targetStats.variance)) + C2);
            const denominator = (gtStats.mean ** 2 + targetStats.mean ** 2 + C1) * (gtStats.variance + targetStats.variance + C2);
            ssim += numerator / denominator;
            windows++;
        }
    }

    return ssim / windows;
}


export {toLoadedImage, waitImage, waitImages, getDiffImage, calculatePSNR, calculateSSIM, getPSNRImage};
