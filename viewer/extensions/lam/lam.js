import ImageContainer from "../../js/image-container.js";
import { waitImage } from "../../js/image-utils.js";
import Viewer from "../../js/viewer.js";

export default class LAM {
    static patch(entrypoint = `http://localhost:5000/lam`) {
        const zipImages = ['image_abs.png', 'blend_abs.png', 'blend_kde.png', 'tensor.png'];
        let index = 1;
        const getContainers = (viewer) => {
            const containers = viewer.imageContainers.filter(c => !c.target.groundTruth && !c.target.hide);
            const gtContainer = viewer.imageContainers.find(c => c.target.groundTruth);
            return { containers, gtContainer };
        }

        let containers, gtContainer;
        const blobToImage = (blob) => {
            const image = new Image();
            image.src = URL.createObjectURL(blob);
            return image;
        }

        const exitLAMMode = (viewer) => {
            for (const container of viewer.imageContainers) {
                container.lam = null;
                container.infoLabel.style.backgroundColor = 'blue';
            }
            viewer.lamMode = false;
            viewer.diffIndex = -1;
            containers = gtContainer = null;
        }

        let results = {};
        Viewer.prototype._updateInfoLabel = Viewer.prototype.updateInfoLabel;
        Viewer.prototype.updateInfoLabel = function () {
            this._updateInfoLabel();
            for (const container of this.imageContainers) {
                if (this.lamMode && container?.lam?.label) {
                    container.infoLabel.textContent += container?.lam?.label;
                }
            }
        }
        Viewer.prototype._setIndex = Viewer.prototype.setIndex;
        Viewer.prototype.setIndex = function (index) {
            if (this.lamMode) {
                exitLAMMode(this);
            }
            return this._setIndex(index);
        }

        Viewer.prototype._update = Viewer.prototype.update;
        Viewer.prototype.update = function (t) {
            if (!this.lamMode) {
                this._update(t);
            } else {
                this.updateStatus = 'start';
                this.updateHeader();
                this.updateURLIndex();
                for (const container of this.imageContainers) {
                    const { lam, target } = container;
                    if (lam) {
                        if (lam.image) {
                            container.setImage(lam.image, this);
                        } else if (lam.images) {
                            container.setImage(lam.images[index], this);
                        } else {
                            const file = this.baseTarget.files[this.indexes[this.index]];
                            const path = this.getImagePath(target, file);
                            container.setImage(this.imageCaches[path], this);
                        }
                    }
                }
                this.updateStatus = 'done';
            }
        }

        Viewer.prototype._addKeyboardHandler = Viewer.prototype.addKeyboardHandler;
        Viewer.prototype.addKeyboardHandler = function () {
            this._addKeyboardHandler();
            document.addEventListener('keydown', async e => {
                const containerMap = getContainers(this);
                containers = containerMap.containers;
                gtContainer = containerMap.gtContainer;

                if ('1' <= e.key && e.key <= '4') {
                    index = parseInt(e.key) - 1;
                    this.update();
                    if (this.zoomMode) {
                        gtContainer.canvas.drawZoomInterface(this.zoomDrawParams);
                        for (let i = 0; i < containers.length; i++) {
                            containers[i].canvas.drawZoomInterface(this.zoomDrawParams);
                        }
                    }
                }
                if (e.key === 'a' && this.baseTarget.groundTruth) {
                    if (this.lamMode) {
                        exitLAMMode(this);
                        this.update();
                        return;
                    }

                    if (this.zoomMode) {
                        this.lamMode = true;
                        const minSize = Math.min(this.zoomAreaWidth, this.zoomAreaHeight);
                        this.zoomAreaWidth = this.zoomAreaHeight = minSize;
                        for (let i = 0; i < containers.length; i++) {
                            containers[i].canvas.updateZoomInterfaceData(this.zoomDrawParams.event);
                            containers[i].canvas.drawZoomInterface(this.zoomDrawParams);
                        }

                        const file = this.getIndexFile();
                        const filePath = this.getImagePath(this.baseTarget, file);
                        const location = JSON.parse(JSON.stringify(this.zoomDrawParams.crop));

                        for (let i = 0; i < containers.length; i++) {
                            containers[i].lam = { label: ` (waiting)` };
                        }
                        gtContainer.lam = { label: ` (calculating)` };
                        const body = JSON.stringify({ type: 'get_position_image', file: filePath, ...location });
                        const positionImage = await fetch(entrypoint, {
                            method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
                        }).then(r => r.blob()).then(blobToImage);
                        positionImage.rawImage = this.getImage(gtContainer.target, file);
                        gtContainer.lam = { image: positionImage, label: ` [GT]` };
                        await waitImage(positionImage);
                        await gtContainer.setImage(positionImage, this);
                        if (this.zoomMode) {
                            gtContainer.canvas.updateZoomInterfaceData(this.zoomDrawParams.event);
                            gtContainer.canvas.drawZoomInterface(this.zoomDrawParams);
                        }

                        for (let i = 0; i < containers.length; i++) {
                            try {
                                containers[i].lam = { label: ` (calculating)` };
                                const { target } = containers[i];
                                const path = target.model || target.path.split('/visualization').shift();
                                const body = JSON.stringify({ type: 'lam', path, file: filePath, ...location });
                                const response = await fetch(entrypoint, {
                                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
                                });
                                containers[i].lam = { label: ` (loading)` };
                                if (response.status === 200) {
                                    const zip = await JSZip.loadAsync(await response.blob());
                                    const images = await Promise.all(zipImages.map(name => zip.file(name).async('blob').then(blobToImage)));
                                    const { diffusionIndex } = await zip.file('data.json').async('string').then(data => JSON.parse(data));
                                    containers[i].lam = { images, diffusionIndex, label: `, DI: ${diffusionIndex}` };
                                    await waitImage(images[index]);
                                    await containers[i].setImage(images[index], this);
                                    images[index].rawImage = this.getImage(containers[i].target, file);
                                    if (this.zoomMode) {
                                        containers[i].canvas.drawZoomInterface(this.zoomDrawParams);
                                    }
                                } else {
                                    const { error } = await response.json();
                                    console.error(error);
                                    containers[i].lam = { error, label: ` (error)` };
                                    containers[i].infoLabel.style.backgroundColor = 'red';
                                }
                            } catch (e) {
                                containers[i].lam = { label: ` (error)` };
                                containers[i].infoLabel.style.backgroundColor = 'red';
                                console.error(e);
                            }
                        }
                    }
                }
            });
        };
    }
}
