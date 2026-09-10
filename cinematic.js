(() => {
  const hero = document.querySelector('.cinema-hero');
  const video = document.getElementById('hero-film');
  const toggle = document.getElementById('motion-toggle');
  const icon = document.getElementById('motion-icon');
  const sceneName = document.getElementById('scene-name');
  const sceneDetail = document.getElementById('scene-detail');
  const progress = document.querySelector('.hero-progress span');
  const connection = navigator.connection;
  const constrained = () => Boolean(connection && (connection.saveData || /(^|-)2g$/.test(connection.effectiveType) || connection.effectiveType === '3g' || (connection.downlink > 0 && connection.downlink < 3)));
  let paused = false;
  let visible = true;
  let playPending = false;
  let retryLightweight = false;
  let playAttempt = 0;
  const params = new URLSearchParams(location.search);

  function updateControls() {
    hero.dataset.paused = String(paused || !visible || document.hidden);
    toggle.setAttribute('aria-pressed', String(paused));
    toggle.setAttribute('aria-label', paused ? 'Play background motion' : 'Pause background motion');
    icon.textContent = paused ? '▷' : 'Ⅱ';
  }
  function posterOnly() {
    paused = true;
    playPending = false;
    playAttempt++;
    retryLightweight = true;
    video.pause();
    hero.dataset.videoState = 'poster';
    video.removeAttribute('src');
    video.load();
    updateControls();
  }
  function watchLoading() {
    if (hero.dataset.videoState !== 'playing') hero.dataset.videoState = 'loading';
  }
  function sync() {
    updateControls();
    if (paused || !visible || document.hidden || hero.dataset.scene !== 'film') {
      playPending = false;
      playAttempt++;
      video.pause();
      return;
    }
    if (!video.hasAttribute('src')) {
      const lightweight = retryLightweight || constrained() || window.matchMedia('(max-width: 850px)').matches || (connection && connection.downlink > 0 && connection.downlink < 5);
      video.src = lightweight ? video.dataset.mobileSrc : video.dataset.desktopSrc;
      video.load();
    }
    if (playPending || !video.paused) return;
    playPending = true;
    const attempt = ++playAttempt;
    watchLoading();
    video.play().then(() => {
      if (attempt === playAttempt) playPending = false;
    }).catch(error => {
      if (attempt !== playAttempt) return;
      playPending = false;
      if (error.name !== 'AbortError') posterOnly();
    });
  }
  function selectScene(scene) {
    hero.dataset.scene = scene;
    if (scene === 'dataset') {
      document.querySelectorAll('.dataset-camera img[data-src]').forEach(img => {
        img.src = img.dataset.src;
        delete img.dataset.src;
      });
    }
    document.querySelectorAll('[data-background]').forEach(button => {
      const active = button.dataset.background === scene;
      button.classList.toggle('selected', active);
      button.setAttribute('aria-pressed', String(active));
    });
    sceneName.textContent = scene === 'film' ? 'LABORATORY FILM' : 'TN-SYN / TRANSNORMAL-2';
    sceneDetail.textContent = scene === 'film' ? 'Rendered visual study' : 'RGB → predicted surface normals';
    progress.style.width = scene === 'film' ? '0%' : '100%';
    sync();
  }
  document.querySelectorAll('[data-background]').forEach(button => button.addEventListener('click', () => selectScene(button.dataset.background)));
  toggle.addEventListener('click', () => { paused = !paused; sync(); });
  document.addEventListener('visibilitychange', sync);
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; sync(); }, { threshold: .02 }).observe(hero);
  video.addEventListener('playing', () => {
    if (paused || !visible || document.hidden || hero.dataset.scene !== 'film') { video.pause(); return; }
    hero.dataset.videoState = 'playing';
  });
  video.addEventListener('waiting', () => { if (!paused && visible && !document.hidden && hero.dataset.scene === 'film') watchLoading(); });
  video.addEventListener('timeupdate', () => {
    if (hero.dataset.scene === 'film' && video.duration) progress.style.width = `${video.currentTime / video.duration * 100}%`;
  });
  video.addEventListener('error', () => { if (video.hasAttribute('src')) posterOnly(); });
  selectScene(params.get('scene') === 'dataset' ? 'dataset' : 'film');
})();
