'use strict';
const samples = [
  {id:'glass', title:'Double-walled glass', type:'In the wild', ratio:'1000 / 744', description:'Transparent surfaces and fine rims, seen by different normal estimators.', methods:['moge-2','lotus-2','transnormal','e2e-ft','marigold','geowizard']},
  {id:'prism', title:'Prism under extreme lighting', type:'In the wild', ratio:'1400 / 950', description:'Compare recovered geometry under strong refraction and lighting changes.', methods:['moge-2','lotus-2','transnormal','e2e-ft','marigold','geowizard']},
  {id:'watch', title:'Mechanical watch', type:'In the wild', ratio:'1024 / 768', description:'Compare fine structures among gears, edges, and textured surfaces.', methods:['moge-2','lotus-2','transnormal','e2e-ft','marigold','geowizard']},
  {id:'porcelain', title:'Porcelain details', type:'In the wild', ratio:'1024 / 768', description:'Compare surface orientation and shape beyond the appearance of the texture.', methods:['moge-2','lotus-2','transnormal','e2e-ft','marigold','geowizard']},
  {id:'cleargrasp', title:'ClearGrasp', type:'Synthetic benchmark', ratio:'1024 / 576', gt:true, description:'Transparent objects in ClearGrasp. Ground truth is also available in the comparison menu.', methods:['moge-2','lotus-2','transnormal','e2e-ft','genpercept']},
  {id:'tn-syn', title:'TN-Syn · Labware', type:'Synthetic benchmark', ratio:'800 / 800', gt:true, description:'Glass labware viewed from the side. Compare baselines or select ground truth to inspect the predicted shape.', methods:['moge-2','lotus-2','transnormal','e2e-ft','genpercept']}
];
const methodNames = {'moge-2':'MoGe-2','lotus-2':'Lotus-2',transnormal:'TransNormal','e2e-ft':'E2E-FT',marigold:'Marigold',geowizard:'GeoWizard',genpercept:'GenPercept',rgb:'Input RGB',gt:'Ground truth'};
const byId = id => document.getElementById(id);
const viewer = byId('compare'), slider = byId('comparison-slider'), picker = byId('comparison-method');
let currentIndex = 0, comparator = 'moge-2', loadRequest = 0;
function updatePosition() {
  const value = Number(slider.value);
  viewer.style.setProperty('--split', `${value}%`);
  slider.setAttribute('aria-valuetext', `${value} percent ${methodNames[comparator]}, ${100-value} percent TransNormal-2`);
}
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(src);
    img.onerror = () => reject(new Error('Image unavailable'));
    img.src = src;
  });
}
function updateOptions(sample) {
  const choices = [...sample.methods, 'rgb', ...(sample.gt ? ['gt'] : [])];
  if (!choices.includes(comparator)) comparator = 'moge-2';
  picker.replaceChildren();
  for (const [label, values] of [['Baseline methods', sample.methods], ['References', ['rgb', ...(sample.gt ? ['gt'] : [])]]]) {
    const group = document.createElement('optgroup'); group.label = label;
    for (const value of values) {
      const option = document.createElement('option');
      option.value = value; option.textContent = methodNames[value]; option.selected = value === comparator;
      group.append(option);
    }
    picker.append(group);
  }
}
async function showComparison() {
  const request = ++loadRequest, sample = samples[currentIndex], method = comparator;
  const status = byId('comparison-status');
  status.textContent = 'Loading comparison…'; status.hidden = false; viewer.setAttribute('aria-busy', 'true');
  try {
    const [normal, reference, rgb] = await Promise.all([
      loadImage(`assets/${sample.id}-normal.png`), loadImage(`assets/${sample.id}-${method}.png`), loadImage(`assets/${sample.id}-rgb.png`)
    ]);
    if (request !== loadRequest) return;
    viewer.style.setProperty('--ratio', sample.ratio);
    byId('comparison-base').src = normal; byId('comparison-base').alt = `TransNormal-2 predicted surface normals: ${sample.title}`;
    byId('comparison-overlay').src = reference; byId('comparison-overlay').alt = `${methodNames[method]}: ${sample.title}`;
    byId('input-preview').src = rgb; byId('input-preview').alt = `Input RGB: ${sample.title}`;
    byId('left-label').textContent = methodNames[method];
    byId('sample-number').textContent = `${String(currentIndex+1).padStart(2,'0')} / ${String(samples.length).padStart(2,'0')}`;
    byId('sample-title').textContent = sample.title; byId('sample-type').textContent = sample.type; byId('sample-description').textContent = sample.description;
    updatePosition(); status.hidden = true; viewer.setAttribute('aria-busy', 'false');
    byId('viewer-announcement').textContent = `${sample.title}. ${methodNames[method]} on the left, TransNormal-2 on the right.`;
  } catch {
    if (request !== loadRequest) return;
    status.textContent = 'This comparison could not load. Please choose another method or sample.'; viewer.setAttribute('aria-busy', 'false');
  }
}
function setSample(index) {
  currentIndex = (index + samples.length) % samples.length;
  const sample = samples[currentIndex]; updateOptions(sample);
  document.querySelectorAll('[data-sample]').forEach(button => {
    const active = button.dataset.sample === sample.id; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
  showComparison();
}
slider.addEventListener('input', updatePosition);
picker.addEventListener('change', () => { comparator = picker.value; showComparison(); });
document.querySelectorAll('[data-sample]').forEach(button => button.addEventListener('click', () => setSample(samples.findIndex(sample => sample.id === button.dataset.sample))));
byId('previous-sample').addEventListener('click', () => setSample(currentIndex-1)); byId('next-sample').addEventListener('click', () => setSample(currentIndex+1));
const tabs = ['transparent','general'];
function chooseTab(name, focus=false) {
  for (const id of tabs) {
    const active = id === name, tab = byId(`${id}-tab`);
    tab.classList.toggle('active', active); tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
    byId(`${id}-panel`).hidden = !active; if (active && focus) tab.focus();
  }
}
for (const name of tabs) {
  byId(`${name}-tab`).addEventListener('click', () => chooseTab(name));
  byId(`${name}-tab`).addEventListener('keydown', event => {
    if (['ArrowRight','ArrowLeft','Home','End'].includes(event.key)) {
      event.preventDefault(); chooseTab(event.key==='Home' ? 'transparent' : event.key==='End' ? 'general' : name==='general' ? 'transparent' : 'general', true);
    }
  });
}
byId('copy-citation').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(byId('bibtex').textContent);
    byId('copy-citation').textContent = 'Copied ✓'; byId('copy-status').textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => byId('copy-citation').textContent='Copy BibTeX ⧉', 2200);
  } catch {
    const range = document.createRange(); range.selectNodeContents(byId('bibtex'));
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    byId('copy-status').textContent = 'Citation selected. Use your browser copy command.'; byId('copy-citation').textContent = 'Select and copy';
  }
});
setSample(0);
