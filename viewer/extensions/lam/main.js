import Viewer from './viewer.js'
import LAM from '../extensions/lam/lam.js';

LAM.patch();

const viewer = new Viewer();
await viewer.init();
await viewer.start();
