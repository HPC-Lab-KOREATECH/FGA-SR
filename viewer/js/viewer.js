import ImageContainer from "./image-container.js"
import mappers from "./mappers.js"
import {
    beep,
    downloadURI,
    getContrastYIQ,
    getOrdinalSuffix,
    isMobile,
    naturalSort,
    stringToColor,
    waitFor
} from "./utils.js";
import {calculatePSNR, calculateSSIM, getDiffImage, getPSNRImage, waitImage, waitImages} from "./image-utils.js";

export default class Viewer {
    getDirectoryInfo(target) {
        if (this.isGitHubHosting) {
            return this.tree.filter(e => e.path.includes(target.searchPath) && e.path.replace(`${target.searchPath}/`).split('/').length === 1 && e.type === 'blob').map(e => e.path.split('/').pop());
        }
        return fetch(target.path, {cache: "no-store"})
            .then(response => response.ok ? response.text() : '[]')
            .then(response => {
                try {
                    response = JSON.parse(response);
                    // nginx autoindex json
                } catch (e) {
                    const anchorArray = Array.from(new DOMParser().parseFromString(response, 'text/html').querySelectorAll('a'));
                    response = anchorArray.map(a => {
                        const name = a.getAttribute('href');
                        const type = name.endsWith('/') ? 'directory' : 'file';
                        return {name: type === 'directory' ? name.substring(0, name.length - 1) : name, type};
                    });
                    // python -m http.server
                }
                return response.filter(f => f.type === 'file').map(f => f.name);
            });
    }

    async init() {
        this.params = Object.fromEntries(new URL(document.location).searchParams);
        this.isGitHubHosting = location.host.endsWith('github.io');
        this.user = location.host.split('.').shift();
        this.repo = location.pathname.split('/').filter(p => p).shift();
        this.root = location.pathname.replace(`/${this.repo}/`, '');

        try {
            if (this.params.configPath) {
                this.configPath = this.params.configPath;
                if (this.isGitHubHosting && this.configPath.startsWith('/')) {
                    this.configPath = `/${this.repo}${this.configPath}`;
                }
            } else {
                this.configPath = `configs/${this.params.config}`;
            }
            this.configRaw = await fetch(this.configPath, {cache: "no-store"}).then(response => response.text());
            const config = JSON.parse(this.configRaw);
            for (const key in config) {
                this[key] = this[key] ? this[key] : config[key];
            }
        } catch (e) {
            if (!this.baseTarget) {
                const h1 = document.createElement('h1');
                h1.textContent = `[ERROR] config file could not be found, or config file is not valid.`;
                h1.style.color = 'red';
                document.body.appendChild(h1);
            }
        }
        this.type = this.params.type || this.type || 'default';
        this.mappers = mappers[this.type];

        const filteredParamKeys = Object.keys(this.params);
        for (const target of this.targets) {
            target.path = filteredParamKeys
                .reduce((str, key) => str.replace(new RegExp(`(?<!\\\\){${key}(?<!\\\\)}`, 'g'), this.params[key]), target.path)
                .replace(/\\([{}])/g, '$1');
        }
        this.targets = this.targets.map((t, i, a) => this.mappers['targetBefore'](t, i, a, this));

        if (this.isGitHubHosting) {
            this.branch = (await fetch(`js/github.io.json`).then(r => r.json())).branch;
            try {
                const response = await fetch(`https://api.github.com/repos/${this.user}/${this.repo}/git/trees/${this.branch}?recursive=true`).then(r => r.json());
                this.tree = response.tree;
                if (!this.tree) {
                    throw new Error(JSON.stringify(response));
                }
                localStorage.tree = JSON.stringify(this.tree);
            } catch (e) {
                console.error(e);
                this.tree = JSON.parse(localStorage.tree);
            }
            this.targets.forEach(target => {
                target.originalPath = target.path;
                target.searchPath = target.path.startsWith('/') ? target.path.substring(1) : this.root + target.path;
                target.path = `/${this.repo}/${target.searchPath}`;
            });
        }
        const targetResponses = await Promise.all(this.targets.map(t => this.getDirectoryInfo(t)));
        this.targets = this.targets.map((t, i) => ({
            ...t, label: t.label || t.originalPath || t.path, files: t.files || targetResponses[i]
        }));
        this.targets = this.targets.filter(t => !t.ignore && t.files.length);
        const maxLabelLength = this.targets.reduce((length, target) => Math.max(length, target.label.length), -1);
        for (const target of this.targets) {
            target.labelBackgroundColor = target.labelBackgroundColor || stringToColor(target.label);
            target.labelBackgroundColor = /^([0-9A-F]{3}){1,2}$/i.test(target.labelBackgroundColor) ? `#${target.labelBackgroundColor}` : target.labelBackgroundColor;
            target.labelColor = target.labelColor || getContrastYIQ(target.labelBackgroundColor);
            target.labelColor = /^([0-9A-F]{3}){1,2}$/i.test(target.labelColor) ? `#${target.labelColor}` : target.labelColor;
            const paddingNeeded = maxLabelLength - target.label.length;
            const padLeft = Math.floor(paddingNeeded / 2);
            target.labelTextContent = target.label.padStart(padLeft + target.label.length, ' ');
            target.labelTextContent = target.labelTextContent.padEnd(maxLabelLength, ' ');
        }

        this.baseTarget = this.targets.find(t => t.groundTruth) || this.targets[0];
        if (!this.baseTarget) {
            const h1 = document.createElement('h1');
            h1.textContent = `[ERROR] Failed to load baseTarget`;
            h1.style.color = 'red';
            document.body.appendChild(h1);

            if (this.configHelp) {
                const span = document.createElement('span');
                span.style.whiteSpace = 'pre';
                span.textContent = this.configHelp;
                document.body.appendChild(span);
            }
        }
        this.baseTarget.files.sort(naturalSort);
        this.targets = this.targets.map((t, i, a) => this.mappers['targetAfter'](t, i, a, this));

        this.title = this.params.title || this.title || 'Super-Resolution Viewer';
        this.indexes = this.params?.indexes?.split(/[.,*]/gi) || this.indexes || Array.from(this.baseTarget.files.keys());
        this.maxFileNameLength = this.indexes.map(i => this.baseTarget.files[i].length).reduce((maxLength, length) => Math.max(maxLength, length));

        this.index = parseInt(this.params.index) || this.index;
        this.index = this.indexes.findIndex(i => i == this.index);
        this.index = this.index < 0 ? 0 : this.index;

        this.imageCaches = {};
        this.preloadSize = parseInt(this.params.preloadSize) || this.preloadSize || 3;

        this.container = document.createElement('article');
        this.container.classList.add('container');

        this.header = document.createElement('header');
        this.header.classList.add('header');

        this.hides = this.params?.hides?.split(/[.,*]/gi)?.map(s => parseInt(s)) || this.hides;
        this.targets.forEach((target, index) => target.hide = this.hides?.includes(index) || target.hide);

        this.imageContainers = this.targets.map(t => new ImageContainer(t));
        this.imageContainers.forEach(c => this.container.appendChild(c));

        this.canvasLeftColor = this.params.canvasLeftColor || this.canvasLeftColor || 'limegreen';
        this.canvasLeftColor = /^([0-9A-F]{3}){1,2}$/i.test(this.canvasLeftColor) ? `#${this.canvasLeftColor}` : this.canvasLeftColor;
        this.canvasWheelColor = this.params.canvasWheelColor || this.canvasWheelColor || 'red';
        this.canvasWheelColor = /^([0-9A-F]{3}){1,2}$/i.test(this.canvasWheelColor) ? `#${this.canvasWheelColor}` : this.canvasWheelColor;
        this.canvasThickness = parseInt(this.params.canvasThickness) || this.canvasThickness || 3;

        this.SSIMWindowSize = parseInt(this.params.SSIMWindowSize) || this.SSIMWindowSize || 11;

        this.PSNRGridWidth = parseInt(this.params.PSNRGridWidth) || this.PSNRGridWidth || 5;
        this.PSNRGridHeight = parseInt(this.params.PSNRGridHeight) || this.PSNRGridHeight || 5;
        this.PSNRGridSize = parseInt(this.params.PSNRGridSize) || this.PSNRGridSize;
        if (this.PSNRGridSize) {
            this.PSNRGridWidth = this.PSNRGridHeight = this.PSNRGridSize;
        }

        this.pageZoom = parseFloat(this.params.pageZoom) || this.pageZoom || localStorage['zoomLevel'] || 1;
        this.pageZoomDelta = parseFloat(this.params.pageZoomDelta) || this.pageZoomDelta || 0.01;
        this.showingPSNRVisualizer = this.params.showingPSNRVisualizer === 'true' || this.showingPSNRVisualizer || false;
        this.showingFavorites = false;
        this.diffIndex = parseInt(this.params.diffIndex) || this.diffIndex || -1;

        this.zoomMode = this.params.zoomMode === 'true' || this.zoomMode || false;
        this.zoomAreaWidthRatio = parseFloat(this.params.zoomAreaWidthRatio) || this.zoomAreaWidthRatio || 0.8;
        this.zoomAreaHeightRatio = parseFloat(this.params.zoomAreaHeightRatio) || this.zoomAreaHeightRatio || 0.8;
        this.zoomAreaWidth = parseInt(this.params.zoomAreaWidth) || this.zoomAreaWidth || 100;
        this.zoomAreaHeight = parseInt(this.params.zoomAreaHeight) || this.zoomAreaHeight || 100;
        this.zoomAreaColor = this.params.zoomAreaColor || this.zoomAreaColor || 'label';
        this.zoomAreaColor = /^([0-9A-F]{3}){1,2}$/i.test(this.zoomAreaColor) ? `#${this.zoomAreaColor}` : this.zoomAreaColor;
        this.zoomAlpha = this.params.zoomAlpha || this.zoomAlpha || 0.5;
        this.zoomAreaThickness = this.params.zoomAreaThickness || this.zoomAreaThickness || 5;
        this.zoomAreaDelta = this.params.zoomAreaDelta || this.zoomAreaDelta || 5;
        this.zoomMouseDown = false;
        this.zoomWidthOnly = false;
        this.zoomHeightOnly = false;

        if (this.params.mobile === 'true' || this.params.mobile === 'false') {
            this.mobile = this.params.mobile === 'true';
        } else {
            this.mobile = this.mobile || false;
        }

        this.crop = this.params.crop || this.crop || null;
        this.crop = this.crop ? this.parseCropString(this.crop) : null;
        if (this.crop) {
            this.showingPSNRVisualizer = this.crop.p === 1;
            this.diffIndex = !isNaN(this.crop.d) ? this.crop.d : -1;
        }

        this.configHelp = this.params.configHelp || this.configHelp;
    }

    parseCropString(input) {
        const result = {};
        input.replace(/([xywhdp])(\d+)/g, (match, key, value) => {
            result[key] = parseInt(value);
        });
        return result;
    }

    getCropString({x, y, w, h}) {
        return `x${x}y${y}w${w}h${h}` + (this.diffIndex > -1 ? `d${this.diffIndex}` : '') + (this.showingPSNRVisualizer ? `p1` : '');
    }

    async downloadCropImages({x, y, w, h}, includeOriginal = false) {
        await waitFor(_ => this.updateStatus === 'done');
        const file = this.getIndexFile();
        const containers = this.imageContainers.filter(c => !c.target.hide);
        const targets = containers.map(c => c.target);
        const images = containers.map(c => c.image);
        await waitImages(images);

        const croppedImages = await Promise.all(images.map(async (image) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d', {willReadFrequently: true});
            canvas.width = w;
            canvas.height = h;
            ctx.drawImage(image, x, y, w, h, 0, 0, w, h);

            return new Promise((resolve) => {
                canvas.toBlob(resolve, 'image/png');
            });
        }));

        const zip = new JSZip();
        croppedImages.forEach((blob, index) => {
            zip.file(`[${index}]${includeOriginal ? ' [CROPPED]' : ''} ${targets[index].label.replaceAll('/', '／')}.png`, blob);
        });

        if (includeOriginal) {
            const originalImages = await Promise.all(images.map(i => fetch(i.src).then(r => r.blob())));
            originalImages.forEach((image, index) => {
                zip.file(`[${index}] [ORIGINAL] ${targets[index].label.replaceAll('/', '／')}.png`, image);
            });
        }

        const content = await zip.generateAsync({type: "blob"});
        const blobURL = URL.createObjectURL(content);
        downloadURI(blobURL, `[${this.title}] ${file.substring(0, file.lastIndexOf('.'))}_${this.getCropString({
            x, y, w, h
        })}.zip`);
    }

    getFavorites() {
        return JSON.parse(localStorage.getItem(`favorites-${this.params.config}`) || '[]');
    }

    setFavorites(favorites) {
        localStorage.setItem(`favorites-${this.params.config}`, JSON.stringify(favorites));
    }

    showFavorites() {
        if (this.favoritesOverlay) {
            document.body.removeChild(this.favoritesOverlay);
        }
        this.favoritesOverlay = document.createElement('div');
        this.favoritesOverlay.classList.add('favorites-overlay');

        const favoritesList = document.createElement('ul');
        favoritesList.classList.add('favorites-list');

        const favorites = this.getFavorites();
        favorites.forEach(url => {
            const listItem = document.createElement('li');
            listItem.classList.add('favorite-item');
            const link = document.createElement('a');
            link.href = url;
            link.textContent = url;
            listItem.appendChild(link);
            favoritesList.appendChild(listItem);
        });

        const favoritesOnlyURL = this.createFavoritesIndexesURL();
        const favoritesOnlyButton = document.createElement('a');
        favoritesOnlyButton.classList.add('favorites-only-button');
        favoritesOnlyButton.textContent = `View Favorites Only: ${favoritesOnlyURL}`;
        favoritesOnlyButton.href = favoritesOnlyURL;
        this.favoritesOverlay.appendChild(favoritesOnlyButton);

        this.favoritesOverlay.appendChild(favoritesList);
        document.body.appendChild(this.favoritesOverlay);
    }

    toggleFavorites() {
        if (this.showingFavorites) {
            document.body.removeChild(this.favoritesOverlay);
            delete this.favoritesOverlay;
        } else {
            this.showFavorites();
        }
        this.showingFavorites = !this.showingFavorites;
    }

    saveFavorite() {
        const favorites = this.getFavorites();
        if (!favorites.includes(location.href)) {
            favorites.push(location.href);
            this.setFavorites(favorites);
        }
        if (this.showingFavorites) {
            this.showFavorites();
        }
        beep();
    }

    createFavoritesIndexesURL() {
        const favorites = this.getFavorites();
        const indexes = Array.from(new Set(favorites.map(f => {
            const urlParams = new URLSearchParams(f.split('?').pop());
            return urlParams.get('index');
        }))).join('.');

        const params = new URLSearchParams(window.location.search);
        params.set('indexes', indexes);

        const retrieveURL = `${location.origin}${location.pathname}?${params.toString()}`;
        return retrieveURL;
    }

    deleteFavorite() {
        this.setFavorites(this.getFavorites().filter(url => url !== location.href));
        if (this.showingFavorites) {
            this.showFavorites();
        }
        beep();
        setTimeout(beep, 100);
    }

    clearFavorites() {
        this.setFavorites([]);
        if (this.showingFavorites) {
            this.showFavorites();
        }
        beep();
    }

    getSafeIndex(index) {
        return Math.max(0, Math.min(index, this.indexes.length - 1));
    }

    updateURLIndex() {
        const currentURL = new URL(location);
        const currentIndex = parseInt(currentURL.searchParams.get('index'));
        if (currentIndex !== this.indexes[this.index]) {
            currentURL.searchParams.set('index', this.indexes[this.index]);
            history.pushState({}, '', currentURL);
        }
    }

    updateHeader() {
        const file = this.getIndexFile();
        const caches = Object.values(this.imageCaches);
        const loadedCaches = caches.filter(i => i.complete);
        const indexDigit = Math.ceil(Math.log10(this.baseTarget.files.length) || 0);
        const cacheDigit = Math.ceil(Math.log10(caches.length) || 0);
        this.header.textContent = `${file.padStart(this.maxFileNameLength, ' ')} (I:${String(this.indexes[this.index]).padEnd(indexDigit, ' ')}) / Caches: ${String(loadedCaches.length).padEnd(cacheDigit, ' ')} of ${String(caches.length).padEnd(cacheDigit, ' ')}`;
    }

    updateInfoLabel() {
        const file = this.getIndexFile();
        for (const container of this.imageContainers) {
            const image = this.getImage(container.target, file);
            if (container.image && this.zoomMode) {
                if (this?.zoomDrawParams?.crop) {
                    const {naturalWidth, naturalHeight} = container.image;
                    const {x, y, w, h} = this?.zoomDrawParams?.crop;
                    let {zoomRate} = this?.zoomDrawParams;
                    zoomRate = Math.round(zoomRate * 100) / 100;
                    container.infoLabel.textContent = `X: ${x}, Y:${y}, Z:${zoomRate}x, S: ${w}x${h}, I: ${naturalWidth}x${naturalHeight}`;
                    container.infoLabel.style.display = '';
                } else {
                    container.infoLabel.style.display = 'none';
                }
            } else {
                if (image) {
                    if (image.psnr || image.ssim) {
                        let labelTextContent = [];
                        if (image.psnr) {
                            labelTextContent.push(`PSNR: ${Math.round(image.psnr * 1e5) / 1e5} (${getOrdinalSuffix(image.psnrRanking)})`);
                        }
                        if (image.ssim) {
                            labelTextContent.push(`SSIM: ${Math.round(image.ssim * 1e5) / 1e5} (${getOrdinalSuffix(image.ssimRanking)})`);
                        }
                        labelTextContent = labelTextContent.join(', ');
                        container.infoLabel.textContent = labelTextContent;
                        container.infoLabel.style.display = '';
                    } else {
                        container.infoLabel.style.display = 'none';
                    }
                }
            }
        }
    }

    setIndex(index) {
        const newIndex = this.getSafeIndex(index);
        const isChanged = this.index !== newIndex;
        this.index = newIndex;
        return isChanged;
    }

    getImagePath(target, file) {
        return this.mappers['file'](target, file, this);
    }

    getImage(target, file) {
        return this.imageCaches[this.getImagePath(target, file)];
    }

    getIndexFile(index = this.index) {
        return this.baseTarget.files[this.indexes[index]];
    }

    generateImageCache(index) {
        const file = this.getIndexFile(index);
        const targets = this.targets.filter(t => !(t.hide && t !== this.baseTarget));
        for (const target of targets) {
            const path = this.getImagePath(target, file);
            if (!this.imageCaches[path]) {
                const newImage = new Image();
                newImage.src = path;
                this.imageCaches[path] = newImage;
            }
        }

        if (this.baseTarget.groundTruth) {
            const baseImage = this.getImage(this.baseTarget, file);
            targets.forEach(async target => {
                if (target !== this.baseTarget) {
                    const rawImage = this.getImage(target, file);
                    await waitImages([baseImage, rawImage]);
                    if (!rawImage.psnr) {
                        rawImage.psnr = 'calculating';
                        rawImage.psnr = calculatePSNR(baseImage, rawImage)[0][0];
                        const images = targets.map(t => this.getImage(t, file)).filter(i => i.psnr);
                        images.sort((a, b) => b.psnr - a.psnr);
                        images.forEach((image, index) => image.psnrRanking = index + 1);
                    }
                    if (!rawImage.ssim) {
                        rawImage.ssim = 'calculating';
                        rawImage.ssim = calculateSSIM(baseImage, rawImage, this.SSIMWindowSize);
                        const images = targets.map(t => this.getImage(t, file)).filter(i => i.ssim);
                        images.sort((a, b) => b.ssim - a.ssim);
                        images.forEach((image, index) => image.ssimRanking = index + 1);
                    }
                    if (!rawImage.psnrs) {
                        rawImage.psnrs = [];
                        rawImage.psnrs = calculatePSNR(baseImage, rawImage, this.PSNRGridWidth, this.PSNRGridHeight);
                    }
                }
            });
        }
    }

    update(delay = 300) {
        this.updateStatus = 'start';
        this.updateHeader();
        this.updateURLIndex();
        const file = this.baseTarget.files[this.indexes[this.index]];
        this.generateImageCache(this.index);
        for (const container of this.imageContainers) {
            const target = container.target;
            const path = this.getImagePath(target, file);
            container.setImage(this.imageCaches[path], this);
        }
        this.updateStatus = 'image';
        const currentIndex = this.index;
        setTimeout(async _ => {
            if (this.index === currentIndex) {
                await this.applyImageEffects();
                this.updateStatus = 'done';
            }
        }, delay);
        for (let i = 1; i <= this.preloadSize; i++) {
            this.generateImageCache(this.getSafeIndex(this.index + i));
            this.generateImageCache(this.getSafeIndex(this.index - i));
        }
    }

    addZoomHandler() {
        document.body.style.zoom = localStorage['zoomLevel'] = this.pageZoom;
        document.addEventListener('wheel', e => {
            if (e.shiftKey) {
                e.preventDefault();
                const currentZoom = parseFloat(document.body.style.zoom) || 1;
                let newZoomLevel = currentZoom + (e.deltaY < 0 ? this.pageZoomDelta : -this.pageZoomDelta);
                newZoomLevel = Math.max(this.pageZoomDelta, newZoomLevel);
                document.body.style.zoom = newZoomLevel;
                localStorage['zoomLevel'] = newZoomLevel;
            }
        }, {passive: false});
    }

    async applyImageEffects() {
        const imageContainers = this.imageContainers.filter(c => !c.target.hide);
        const file = this.getIndexFile();
        const diffBaseImage = this.diffIndex > -1 ? this.getImage(imageContainers[this.diffIndex].target, file) : null;
        await Promise.all(imageContainers.map(async (c, i) => {
            let rawImage = this.getImage(c.target, file);
            let currentImage = rawImage;
            await waitImage(rawImage);
            if (i !== this.diffIndex && diffBaseImage) {
                await waitImage(diffBaseImage);
                currentImage = getDiffImage(diffBaseImage, currentImage);
            }
            if (this.baseTarget.groundTruth && !c.target.groundTruth && this.showingPSNRVisualizer) {
                await waitFor(_ => rawImage.psnr && rawImage.psnrs && rawImage.psnrs.length > 0);
                currentImage = getPSNRImage({
                    totalPSNR: rawImage.psnr,
                    minPSNR: rawImage.psnrs.min,
                    maxPSNR: rawImage.psnrs.realMax,
                    psnrs: rawImage.psnrs,
                    gridWidth: this.PSNRGridWidth,
                    gridHeight: this.PSNRGridHeight
                }, currentImage);
            }
            currentImage.rawImage = rawImage;
            c.setImage(currentImage, this);
        }));
    }

    addMobileKeyboardHandler() {
        if (isMobile() || this.mobile) {
            this.mobileInput = document.createElement('input');
            this.mobileInput.type = 'text';
            this.mobileInput.style.position = 'fixed';
            this.mobileInput.style.zIndex = '0';
            this.mobileInput.style.width = '100vw';
            this.mobileInput.style.left = '0';
            this.mobileInput.style.bottom = '50vh';
            this.mobileInput.style.fontSize = '5rem';
            this.mobileInput.style.height = '0';
            this.mobileInput.style.opacity = '0';

            const redirectKeyEvent = (keyChar, eventType) => {
                const simulatedEvent = new KeyboardEvent(eventType, {
                    key: keyChar, bubbles: true, cancelable: true
                });
                document.dispatchEvent(simulatedEvent);
            }

            this.mobileInput.addEventListener('beforeinput', (e) => {
                if (e.data) {
                    redirectKeyEvent(e.data, "keydown");
                    redirectKeyEvent(e.data, "keypress");
                    redirectKeyEvent(e.data, "keyup");
                }
                e.target.value = "";
            });

            document.body.appendChild(this.mobileInput);
            document.body.addEventListener('click', () => {
                this.mobileInput.focus();
            });
        }
    }

    addKeyboardHandler() {
        document.addEventListener('keydown', e => {
            if (e.ctrlKey) {
                return;
            }
            if (e.key === "ArrowRight" || e.key === "Right" || e.key === "m") {
                if (this.setIndex(this.index + 1)) {
                    this.update();
                }
            } else if (e.key === "ArrowLeft" || e.key === "Left" || e.key === "n") {
                if (this.setIndex(this.index - 1)) {
                    this.update();
                }
            } else if (e.key === 'response') {
                this.update(0);
            } else if (e.key === 's') {
                this.saveFavorite();
            } else if (e.key === 'l') {
                this.toggleFavorites();
            } else if (e.key === 'd') {
                this.deleteFavorite();
            } else if (e.key === 'c') {
                (async () => {
                    downloadURI(await domtoimage.toPng(document.body), `[${this.title}] ${this.getIndexFile()}`);
                })();
            } else if (e.key === 'F3') {
                e.preventDefault();
                this.clearFavorites();
            } else if ('1' <= e.key && e.key <= '9') {
                const pressedIndex = parseInt(e.key) - 1;
                this.diffIndex = this.diffIndex !== pressedIndex ? pressedIndex : -1;
                this.update(0);
            } else if (e.key === 'w') {
                this.zoomWidthOnly = true;
            } else if (e.key === 'h') {
                this.zoomHeightOnly = true;
            } else if (e.key === 'p') {
                this.showingPSNRVisualizer = !this.showingPSNRVisualizer;
                this.update(0);
            } else if (e.key === 'i') {
                const newIndex = prompt(`index? (${this.baseTarget.files.length} files)`);
                if (newIndex && !isNaN(newIndex)) {
                    this.setIndex(this.getSafeIndex(parseInt(newIndex)));
                    this.update(0);
                }
            } else if (e.key === 'F1' || e.key === 'q') {
                alert(`
                        F1, q: help
                        F2, f: config help
                        ← , →: previous, next image
                        n , m: previous, next image
                        (i)ndex: move to index
                        (r)eset
                        shift + wheel: page zoom

                        (s)dd to favorites
                        (l)ist favorites (toggle)
                        (d)elete favorites
                        f3: delete all favorites

                        (p)snr visualizer (toggle)
                        1 ~ 9: show image diff (toggle)
                        (c)apture page

                        (z)oom mode (toggle)
                        wheel: resize zoom area (in zoom mode)
                        (w) + wheel: resize zoom area width (in zoom mode)
                        (h) + wheel: resize zoom area height (in zoom mode)
                        space: download cropped images (in zoom mode)
                        shift + space: download cropped images with original images (in zoom mode)
                        (u)rl: copy crop url (in zoom mode)
                    `.split('\n').map(l => l.trim()).join('\n').trim());
                e.preventDefault();
            } else if (e.key === 'F2' || e.key === 'f') {
                alert(this.configHelp);
            } else if (e.key === ' ') {
                if (this.zoomMode) {
                    if (e.shiftKey) {
                        this.downloadCropImages(this.zoomDrawParams.crop, true);
                    } else {
                        this.downloadCropImages(this.zoomDrawParams.crop);
                    }
                    e.preventDefault();
                }
            } else if (e.key === 'u') {
                if (this.zoomMode) {
                    const url = new URL(location.href);
                    url.searchParams.set('crop', this.getCropString(this.zoomDrawParams.crop));
                    navigator.clipboard.writeText(url.href);
                    alert("Copied to clipboard.");
                }
            } else if (e.key === 'z') {
                this.zoomMode = !this.zoomMode;
                this.update(0);
            }
        });
        document.addEventListener('keyup', e => {
            if (e.key === 'w') {
                this.zoomWidthOnly = false;
            } else if (e.key === 'h') {
                this.zoomHeightOnly = false;
            }
        });
    }

    runTimeBasedUpdater() {
        setInterval(() => {
            this.updateHeader();
            this.updateURLIndex();
            this.updateInfoLabel();
        }, 100);
    }

    async start() {
        this.addZoomHandler();
        this.addKeyboardHandler();
        this.addMobileKeyboardHandler();
        this.runTimeBasedUpdater();
        this.update();
        document.title = this.title;
        document.body.appendChild(this.header);
        document.body.appendChild(this.container);

        if (this.crop) {
            this.downloadCropImages(this.crop);
        }
    }
}
