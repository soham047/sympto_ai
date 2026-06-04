// ======== Shared helpers ========
const API = "/api";
const fetchJSON = (url, opts = {}) => fetch(url, opts).then(r => r.json());
const postJSON  = (url, body) => fetchJSON(url, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
const getJSON   = (url) => fetchJSON(url);

// ======== Inject nav + footer + scroll-aware navbar + reveal animations ========
async function injectShell() {
  const [nav, footer] = await Promise.all([
    fetch("/nav.html").then(r => r.text()),
    fetch("/footer.html").then(r => r.text()),
  ]);
  document.body.insertAdjacentHTML("afterbegin", nav);
  document.body.insertAdjacentHTML("beforeend", footer);

  // Highlight active route
  const path = location.pathname === "/" ? "/" : location.pathname;
  document.querySelectorAll("[data-route]").forEach(a => {
    if (a.getAttribute("data-route") === path) a.classList.add("active-link");
  });
  if (["/blood.html","/symptom.html","/image.html","/chat.html","/lifestyle.html"].includes(path)) {
    document.querySelector('[data-route="/services.html"]')?.classList.add("active-link");
  }

  // Shadow on scroll
  const navbar = document.querySelector(".navbar");
  const onScroll = () => navbar.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();

  // Reveal on scroll
  initReveal();
}
document.addEventListener("DOMContentLoaded", injectShell);

function initReveal() {
  if (!("IntersectionObserver" in window)) {
    document.querySelectorAll(".reveal").forEach(el => el.classList.add("in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });
  document.querySelectorAll(".reveal").forEach(el => io.observe(el));
}

// ======== UI helpers ========
const showLoading = (el, msg = "Processing") =>
  el.classList.remove("d-none", "error") ||
  (el.innerHTML = `<div class="d-flex align-items-center gap-2 text-muted"><div class="spinner-border spinner-border-sm" style="color:var(--accent)"></div><span>${msg}…</span></div>`);

const showError = (el, msg) => {
  el.classList.remove("d-none"); el.classList.add("error");
  el.innerHTML = `<div class="d-flex align-items-start gap-2"><i class="bi bi-exclamation-circle" style="color:var(--danger);font-size:1.25rem"></i><div><strong style="color:var(--danger)">Error</strong><div class="small">${msg}</div></div></div>`;
};

const renderResult = (el, html) => {
  el.classList.remove("d-none", "error");
  el.innerHTML = html;
};

const probabilityBars = (probs) =>
  Object.entries(probs).sort((a,b) => b[1]-a[1]).map(([k,v]) => `
    <div class="mb-3">
      <div class="d-flex justify-content-between small mb-1"><span class="text-ink">${k}</span><span class="text-muted">${(v*100).toFixed(1)}%</span></div>
      <div class="prob"><div class="fill" style="width:${(v*100).toFixed(1)}%"></div></div>
    </div>`).join("");

const listClean = (items) =>
  `<ul class="list-clean">${items.map(i => `<li><i class="bi bi-check2"></i><span>${i}</span></li>`).join("")}</ul>`;
