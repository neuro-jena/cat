// CAT26 website – small progressive enhancements; the page works without JavaScript.

// Mobile navigation toggle
const navToggle = document.querySelector('.nav-toggle');
const nav = document.getElementById('site-nav');
if (navToggle && nav) {
  const setOpen = (open) => {
    navToggle.setAttribute('aria-expanded', String(open));
    nav.classList.toggle('is-open', open);
  };
  navToggle.addEventListener('click', () => setOpen(navToggle.getAttribute('aria-expanded') !== 'true'));
  nav.addEventListener('click', (e) => { if (e.target.closest('a')) setOpen(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });
}

// "Copy citation" / "Copy BibTeX" buttons
document.querySelectorAll('[data-copy]').forEach((button) => {
  const label = button.textContent;
  button.addEventListener('click', async () => {
    const source = document.querySelector(button.dataset.copy);
    if (!source) return;
    try {
      await navigator.clipboard.writeText(source.textContent.replace(/\s+\n/g, '\n').trim());
      button.textContent = 'Copied';
    } catch {
      // Clipboard API unavailable (e.g. file:// or old browser): select the text instead
      source.closest('details')?.setAttribute('open', '');
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Press Ctrl+C to copy';
    }
    setTimeout(() => { button.textContent = label; }, 2000);
  });
});

// Highlight the visitor's operating system in the standalone download table
(() => {
  const ua = navigator.userAgent;
  const os = /Windows/.test(ua) ? 'windows'
    : /Macintosh|Mac OS X/.test(ua) ? 'mac'
    : /Linux/.test(ua) && !/Android/.test(ua) ? 'linux'
    : null;
  if (!os) return;
  document.querySelectorAll(`.dl-table tr[data-os^="${os}"]`).forEach((row) => row.classList.add('is-current'));
})();
