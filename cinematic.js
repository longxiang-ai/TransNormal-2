(() => {
  const hero=document.querySelector('.cinema-hero'),video=document.getElementById('hero-film');
  const toggle=document.getElementById('motion-toggle'),icon=document.getElementById('motion-icon');
  const sceneName=document.getElementById('scene-name'),sceneDetail=document.getElementById('scene-detail');
  const progress=document.querySelector('.hero-progress span');
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  let paused=reduced.matches, visible=true;
  const params=new URLSearchParams(location.search);
  function sync(){
    hero.dataset.paused=String(paused||!visible||document.hidden);
    toggle.setAttribute('aria-pressed',String(paused));
    toggle.setAttribute('aria-label',paused?'Play background motion':'Pause background motion');
    icon.textContent=paused?'▷':'Ⅱ';
    if(paused||!visible||document.hidden||hero.dataset.scene!=='film')video.pause();
    else video.play().catch(()=>{paused=true;hero.dataset.paused='true';toggle.setAttribute('aria-pressed','true');toggle.setAttribute('aria-label','Play background motion');icon.textContent='▷';});
  }
  function selectScene(scene){
    hero.dataset.scene=scene;
    document.querySelectorAll('[data-background]').forEach(b=>{const active=b.dataset.background===scene;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',String(active));});
    sceneName.textContent=scene==='film'?'LABORATORY FILM':'TN-SYN / TRANSNORMAL-2';
    sceneDetail.textContent=scene==='film'?'Rendered visual study':'RGB → predicted surface normals';
    progress.style.width=scene==='film'?'0%':'100%';sync();
  }
  document.querySelectorAll('[data-background]').forEach(b=>b.addEventListener('click',()=>selectScene(b.dataset.background)));
  toggle.addEventListener('click',()=>{paused=!paused;sync();});
  reduced.addEventListener('change',e=>{paused=e.matches;sync();});
  document.addEventListener('visibilitychange',sync);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;sync();},{threshold:.02}).observe(hero);
  video.addEventListener('timeupdate',()=>{if(hero.dataset.scene==='film'&&video.duration)progress.style.width=`${video.currentTime/video.duration*100}%`;});
  video.addEventListener('error',()=>selectScene('dataset'));
  video.querySelector('source').addEventListener('error',()=>selectScene('dataset'));
  selectScene(params.get('scene')==='dataset'?'dataset':'film');
})();
