'use strict';
const samples=[
  {id:'glass',title:'Double-walled glass',type:'In the wild',ratio:'1000 / 744',description:'Transparent surfaces, recovered from a single RGB image.'},
  {id:'prism',title:'Prism under extreme lighting',type:'In the wild',ratio:'1400 / 950',description:'Surface geometry in a scene with strong refraction and lighting changes.'},
  {id:'watch',title:'Mechanical watch',type:'In the wild',ratio:'1024 / 768',description:'Fine geometric structures among gears, edges, and textured surfaces.'},
  {id:'porcelain',title:'Porcelain details',type:'In the wild',ratio:'1024 / 768',description:'Surface orientation and shape beyond the appearance of the texture.'},
  {id:'cleargrasp',title:'ClearGrasp',type:'Synthetic benchmark',ratio:'1024 / 576',gt:true,description:'A synthetic ClearGrasp example. Switch to GT / Normals to compare with ground truth.'},
  {id:'tn-syn',title:'TN-Syn',type:'Synthetic benchmark',ratio:'800 / 800',gt:true,description:'A synthetic transparent-object scene. Ground truth is available for direct comparison.'}
];
const byId=id=>document.getElementById(id);
const viewer=byId('compare'),slider=byId('comparison-slider');
let currentIndex=0,mode='rgb';
function updatePosition(){const value=Number(slider.value);viewer.style.setProperty('--split',`${value}%`);slider.setAttribute('aria-valuetext',`${value} percent ${mode==='gt'?'ground truth':'input RGB'}, ${100-value} percent predicted normals`);}
function setSample(index){
  currentIndex=(index+samples.length)%samples.length;const s=samples[currentIndex];if(!s.gt)mode='rgb';
  viewer.style.setProperty('--ratio',s.ratio);
  byId('comparison-base').src=`assets/${s.id}-normal.png`;byId('comparison-base').alt=`TransNormal-2 predicted surface normals: ${s.title}`;
  byId('comparison-overlay').src=`assets/${s.id}-${mode==='gt'?'gt':'rgb'}.png`;byId('comparison-overlay').alt=`${mode==='gt'?'Ground-truth surface normals':'Input RGB image'}: ${s.title}`;
  byId('sample-number').textContent=`${String(currentIndex+1).padStart(2,'0')} / ${String(samples.length).padStart(2,'0')}`;
  byId('sample-title').textContent=s.title;byId('sample-type').textContent=s.type;byId('sample-description').textContent=s.description;
  byId('left-label').textContent=mode==='gt'?'GROUND TRUTH':'INPUT RGB';byId('gt-mode').disabled=!s.gt;
  byId('gt-mode').title=s.gt?'Compare ground truth with TransNormal-2':'Ground truth is available for benchmark samples';
  for(const key of ['rgb','gt']){byId(`${key}-mode`).classList.toggle('active',mode===key);byId(`${key}-mode`).setAttribute('aria-pressed',String(mode===key));}
  document.querySelectorAll('[data-sample]').forEach(button=>{const active=button.dataset.sample===s.id;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  updatePosition();byId('viewer-announcement').textContent=`${s.title}. ${mode==='gt'?'Ground truth':'RGB'} compared with predicted normals.`;
}
slider.addEventListener('input',updatePosition);
document.querySelectorAll('[data-sample]').forEach(button=>button.addEventListener('click',()=>setSample(samples.findIndex(s=>s.id===button.dataset.sample))));
byId('previous-sample').addEventListener('click',()=>setSample(currentIndex-1));byId('next-sample').addEventListener('click',()=>setSample(currentIndex+1));
for(const key of ['rgb','gt'])byId(`${key}-mode`).addEventListener('click',()=>{mode=key;setSample(currentIndex);});
const tabs=['transparent','general'];
function chooseTab(name,focus=false){for(const id of tabs){const active=id===name;const tab=byId(`${id}-tab`);tab.classList.toggle('active',active);tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;byId(`${id}-panel`).hidden=!active;if(active&&focus)tab.focus();}}
for(const name of tabs){byId(`${name}-tab`).addEventListener('click',()=>chooseTab(name));byId(`${name}-tab`).addEventListener('keydown',event=>{if(['ArrowRight','ArrowLeft','Home','End'].includes(event.key)){event.preventDefault();chooseTab(event.key==='Home'?'transparent':event.key==='End'?'general':name==='general'?'transparent':'general',true);}});}
byId('copy-citation').addEventListener('click',async()=>{
  const text=byId('bibtex').textContent;
  try{await navigator.clipboard.writeText(text);byId('copy-citation').textContent='Copied ✓';byId('copy-status').textContent='BibTeX copied to clipboard.';setTimeout(()=>byId('copy-citation').textContent='Copy BibTeX ⧉',2200);}
  catch{const range=document.createRange();range.selectNodeContents(byId('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);byId('copy-status').textContent='Citation selected. Use your browser copy command.';byId('copy-citation').textContent='Select and copy';}
});
