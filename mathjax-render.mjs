import fs from 'node:fs';
import {mathjax} from 'mathjax-full/js/mathjax.js';
import {TeX} from 'mathjax-full/js/input/tex.js';
import {SVG} from 'mathjax-full/js/output/svg.js';
import {liteAdaptor} from 'mathjax-full/js/adaptors/liteAdaptor.js';
import {RegisterHTMLHandler} from 'mathjax-full/js/handlers/html.js';
import {AllPackages} from 'mathjax-full/js/input/tex/AllPackages.js';

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: AllPackages});
const svg = new SVG({fontCache: 'none'});
const html = mathjax.document('', {InputJax: tex, OutputJax: svg});
const node = html.convert(input.latex, {display: true});
fs.writeFileSync(input.output, adaptor.innerHTML(node), 'utf8');

