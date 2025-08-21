import {downloadURI} from "./utils.js";

export default class ImageContainer extends HTMLDivElement {
    constructor(target) {
        super();
        if (!target) {
            return;
        }
        this.target = target;
        if (this.target.hide) {
            this.style.display = 'none';
        }
        this.label = document.createElement('div');
        this.label.classList.add('label');
        this.label.textContent = target.labelTextContent;
        this.label.style.color = target.labelColor;
        this.label.style.backgroundColor = target.labelBackgroundColor;
        this.label.onclick = (e) => {
            downloadURI(this.image.src, (this.image.rawImage || this.image).src.split('/').pop());
        };
        this.appendChild(this.label);

        this.infoLabel = document.createElement('div');
        this.infoLabel.classList.add('info-label');
        this.infoLabel.style.display = 'none';
        this.appendChild(this.infoLabel);

        this.classList.add('image-container');
    }

    setImage(image, viewer) {
        if (this.image) {
            this.removeChild(this.image);
        }
        this.image = image || new Image();
        this.image.classList.add('image');
        this.appendChild(this.image);

        if (this.canvas) {
            this.removeChild(this.canvas);
        }
        this.canvas = document.createElement('canvas');
        this.canvas.classList.add('overlay-canvas');

        const ctx = this.canvas.getContext('2d', {willReadFrequently: true});
        let isPainting = false;
        let isErasing = false;
        let hasDrawing = false;

        const conditionalHandler = (expression, handler) => {
            return (e) => {
                if (expression()) {
                    handler(e);
                }
            }
        };
        const ifZoomMode = handler => conditionalHandler(_ => viewer.zoomMode === true, handler);
        const ifNotZoomMode = handler => conditionalHandler(_ => viewer.zoomMode === false, handler);

        this.canvas.addEventListener('mousedown', ifNotZoomMode((e) => {
            isPainting = true;
            const zoomLevel = parseFloat(document.body.style.zoom) || 1;

            ctx.beginPath();
            ctx.moveTo(e.offsetX / zoomLevel, e.offsetY / zoomLevel);

            if (e.button === 2) {
                isErasing = true;
                ctx.globalCompositeOperation = 'destination-out';
                ctx.lineWidth = viewer.canvasThickness * 10;
            } else if (e.button === 0) {
                isErasing = false;
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = viewer.canvasLeftColor;
                ctx.lineWidth = viewer.canvasThickness;
            } else if (e.button === 1) {
                isErasing = false;
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = viewer.canvasWheelColor;
                ctx.lineWidth = viewer.canvasThickness;
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
            }
        }));

        this.canvas.addEventListener('mousemove', ifNotZoomMode((e) => {
            if (isPainting) {
                hasDrawing = true;
                const zoomLevel = parseFloat(document.body.style.zoom) || 1;
                ctx.lineTo(e.offsetX / zoomLevel, e.offsetY / zoomLevel);
                ctx.stroke();
            }
        }));

        this.canvas.addEventListener('mouseup', ifNotZoomMode(() => {
            isPainting = false;
        }));

        this.canvas.addEventListener('contextmenu', ifNotZoomMode((e) => {
            e.preventDefault();
        }));

        this.canvas.addEventListener('mouseleave', ifNotZoomMode(() => {
            isPainting = false;
        }));

        this.canvas.drawZoomInterface = (drawParams) => {
            ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            ctx.strokeStyle = viewer.zoomAreaColor === 'label' ? this.target.labelBackgroundColor : viewer.zoomAreaColor;
            ctx.lineWidth = viewer.zoomAreaThickness;
            ctx.setLineDash([5, 3]);
            ctx.strokeRect(...drawParams.zoomAreaRect);

            ctx.globalAlpha = viewer.zoomMouseDown ? 1 : viewer.zoomAlpha;
            ctx.drawImage(this.image, ...drawParams.zoomImage);

            ctx.setLineDash([]);
            ctx.strokeRect(...drawParams.zoomImageRect);
            ctx.globalAlpha = 1;
        }

        this.canvas.updateZoomInterfaceData = (e) => {
            const zoomLevel = parseFloat(document.body.style.zoom) || 1;
            let mouseX = e.offsetX / zoomLevel;
            let mouseY = e.offsetY / zoomLevel;

            const canvasZoomAreaWidth = viewer.zoomAreaWidth / image.naturalWidth * image.width;
            const canvasZoomAreaHeight = viewer.zoomAreaHeight / image.naturalHeight * image.height;

            mouseX = Math.max(canvasZoomAreaWidth / 2, Math.min(this.canvas.width - canvasZoomAreaWidth / 2, mouseX));
            mouseY = Math.max(canvasZoomAreaHeight / 2, Math.min(this.canvas.height - canvasZoomAreaHeight / 2, mouseY));

            const sx = Math.max(0, mouseX - canvasZoomAreaWidth / 2);
            const sy = Math.max(0, mouseY - canvasZoomAreaHeight / 2);

            const displayWidth = this.canvas.width * viewer.zoomAreaWidthRatio;
            const displayHeight = this.canvas.height * viewer.zoomAreaHeightRatio;

            const dx = this.canvas.width * (1 - viewer.zoomAreaWidthRatio) / 2;
            const dy = this.canvas.height * (1 - viewer.zoomAreaHeightRatio) / 2;

            const ex = Math.min(this.canvas.width, mouseX + canvasZoomAreaWidth / 2);
            const ey = Math.min(this.canvas.height, mouseY + canvasZoomAreaHeight / 2);

            const sourceAspectRatio = canvasZoomAreaWidth / canvasZoomAreaHeight;
            let targetWidth, targetHeight;

            if (sourceAspectRatio > (displayWidth / displayHeight)) {
                targetWidth = displayWidth;
                targetHeight = displayWidth / sourceAspectRatio;
            } else {
                targetHeight = displayHeight;
                targetWidth = displayHeight * sourceAspectRatio;
            }

            const targetX = dx + (displayWidth - targetWidth) / 2;
            const targetY = dy + (displayHeight - targetHeight) / 2;

            const renderWidthRatio = image.naturalWidth / image.width;
            const renderHeightRatio = image.naturalHeight / image.height;

            viewer.zoomDrawParams = {
                zoomAreaRect: [sx, sy, ex - sx, ey - sy],
                zoomImage: [sx * renderWidthRatio, sy * renderHeightRatio, canvasZoomAreaWidth * renderWidthRatio, canvasZoomAreaHeight * renderHeightRatio, targetX, targetY, targetWidth, targetHeight],
                zoomImageRect: [targetX, targetY, targetWidth, targetHeight],
                crop: {
                    x: parseInt(sx * renderWidthRatio),
                    y: parseInt(sy * renderHeightRatio),
                    w: Math.round(canvasZoomAreaWidth * renderWidthRatio),
                    h: Math.round(canvasZoomAreaHeight * renderHeightRatio)
                },
                event: e,
                zoomRate: targetWidth / (ex - sx)
            };
        }

        const handleZoomMode = (e) => {
            this.canvas.updateZoomInterfaceData(e);
            for (const container of viewer.imageContainers) {
                if (!container.target.hide) {
                    container.canvas.drawZoomInterface(viewer.zoomDrawParams);
                }
            }
        };

        this.canvas.addEventListener('mousemove', ifZoomMode(handleZoomMode));

        this.canvas.addEventListener('mousedown', ifZoomMode(e => {
            if (e.button === 0) {
                viewer.zoomMouseDown = true;
                handleZoomMode(e);
            }
        }));

        this.canvas.addEventListener('mouseup', ifZoomMode(e => {
            viewer.zoomMouseDown = false;
            handleZoomMode(e);
        }));

        this.canvas.addEventListener('wheel', ifZoomMode(e => {
            if (e.deltaY < 0) {
                const limitSize = Math.min(image.naturalWidth, image.naturalHeight);
                if (viewer.zoomWidthOnly) {
                    viewer.zoomAreaWidth = Math.min(image.naturalWidth, viewer.zoomAreaWidth + viewer.zoomAreaDelta);
                } else if (viewer.zoomHeightOnly) {
                    viewer.zoomAreaHeight = Math.min(image.naturalHeight, viewer.zoomAreaHeight + viewer.zoomAreaDelta);
                } else {
                    viewer.zoomAreaWidth = Math.min(limitSize, viewer.zoomAreaWidth + viewer.zoomAreaDelta);
                    viewer.zoomAreaHeight = Math.min(limitSize, viewer.zoomAreaHeight + viewer.zoomAreaDelta);
                }
            } else if (e.deltaY > 0) {
                if (viewer.zoomWidthOnly) {
                    viewer.zoomAreaWidth = Math.max(viewer.zoomAreaDelta, viewer.zoomAreaWidth - viewer.zoomAreaDelta);
                } else if (viewer.zoomHeightOnly) {
                    viewer.zoomAreaHeight = Math.max(viewer.zoomAreaDelta, viewer.zoomAreaHeight - viewer.zoomAreaDelta);
                } else {
                    viewer.zoomAreaWidth = Math.max(viewer.zoomAreaDelta, viewer.zoomAreaWidth - viewer.zoomAreaDelta);
                    viewer.zoomAreaHeight = Math.max(viewer.zoomAreaDelta, viewer.zoomAreaHeight - viewer.zoomAreaDelta);
                }
            }
            e.preventDefault();
            handleZoomMode(e);
        }));

        const createMouseEmulator = (type, initDict = {}) => (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(type, {
                clientX: touch.clientX, clientY: touch.clientY, ...initDict
            });
            this.canvas.dispatchEvent(mouseEvent);
        }

        this.canvas.addEventListener('touchstart', createMouseEmulator('mousedown', {button: 0}));
        this.canvas.addEventListener('touchmove', createMouseEmulator('mousemove'));
        this.canvas.addEventListener('touchend', createMouseEmulator('mousemove'));
        this.canvas.addEventListener('touchcancel', createMouseEmulator('mouseup'));

        this.appendChild(this.canvas);

        this.image.onload = () => {
            if (!(this.image.width && this.image.height)) {
                return;
            }
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d', {willReadFrequently: true});
            tempCanvas.width = this.canvas.width;
            tempCanvas.height = this.canvas.height;
            tempCtx.drawImage(this.canvas, 0, 0);

            this.canvas.width = this.image.width;
            this.canvas.height = this.image.height;
            this.canvas.style.top = `${this.label.offsetHeight}px`;
            ctx.drawImage(tempCanvas, 0, 0, this.canvas.width, this.canvas.height);
        };

        if (this.image.complete) {
            this.image.onload();
        }

        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }

        this.resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                if (entry.target === this.image) {
                    this.image.onload();
                }
            }
        });

        this.resizeObserver.observe(this.image);
    }
}
customElements.define('image-container', ImageContainer, {extends: 'div'});
