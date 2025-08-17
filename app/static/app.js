async function jsonFetch(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "class") e.className = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.substring(2), v);
    else e.setAttribute(k, v);
  });
  (Array.isArray(children) ? children : [children]).forEach(c => {
    if (c == null) return;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return e;
}

function card(item, extras = {}) {
  const title = `${item.title || "?"}${item.year ? " (" + item.year + ")" : ""}`;
  const meta = [];
  if (item.genres && item.genres.length) meta.push(item.genres.join(" • "));
  if (item.score && typeof item.score === "number") meta.push(`score ${item.score.toFixed(3)}`);
  if (item.est != null) meta.push(`est ${item.est.toFixed(2)}`);
  if (item.score && typeof item.score === "object" && item.score.hybrid != null) {
    meta.push(`hyb ${item.score.hybrid.toFixed(3)}`);
    meta.push(`cont ${item.score.content_norm.toFixed(3)}`);
    if (item.score.cf_est != null) meta.push(`cf ${item.score.cf_est.toFixed(2)}`);
  }

  const rate = el("div", { class: "rate" }, [
    ...[1,2,3,4,5].map(n => {
      const b = el("button", { type: "button", class: "star", "data-stars": n }, "★");
      b.addEventListener("click", () => {
        // nur Demo – später Tag 12 speichern wir richtig
        console.log("rate", { movieId: item.movieId, stars: n, source: extras.source || "ui" });
        b.parentElement.querySelectorAll("button").forEach(x => x.classList.remove("on"));
        for (let i = 0; i < n; i++) rate.querySelectorAll("button")[i].classList.add("on");
      });
      return b;
    })
  ]);

  return el("article", { class: "card" }, [
    el("header", {}, el("strong", {}, title)),
    el("small", {}, meta.join(" • ")),
    rate
  ]);
}

function renderList(containerId, items, sourceLabel) {
  const root = document.getElementById(containerId);
  root.innerHTML = "";
  if (!items || !items.length) {
    root.appendChild(el("p", {}, "Keine Ergebnisse."));
    return;
  }
  items.forEach(it => root.appendChild(card(it, { source: sourceLabel })));
}

function toParams(obj) {
  return Object.entries(obj)
    .filter(([, v]) => v !== "" && v !== null && v !== undefined)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
}

// ---- Handlers ----

document.addEventListener("DOMContentLoaded", () => {

  // 1) Similar
  document.getElementById("form-similar").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = document.getElementById("q").value.trim();
    const k = document.getElementById("k1").value;
    const url = `/recommend/content?${toParams({ q, k })}`;
    try {
      const data = await jsonFetch(url);
      renderList("similar-results", data.results, "content-similar");
    } catch (err) {
      console.error(err);
      renderList("similar-results", [], "content-similar");
    }
  });

  // 2) Content personal
  document.getElementById("form-content-personal").addEventListener("submit", async (e) => {
    e.preventDefault();
    const userId = document.getElementById("userContent").value;
    const k      = document.getElementById("k2").value;
    const mode   = document.getElementById("mode").value;
    const minRating = document.getElementById("minRating").value;
    const url = `/recommend/content/personal?${toParams({ userId, k, mode, minRating })}`;
    try {
      const data = await jsonFetch(url);
      renderList("content-personal-results", data.results, "content-personal");
    } catch (err) {
      console.error(err);
      renderList("content-personal-results", [], "content-personal");
    }
  });

  // 3) CF personal
  document.getElementById("form-cf-personal").addEventListener("submit", async (e) => {
    e.preventDefault();
    const userId = document.getElementById("userCF").value;
    const k      = document.getElementById("k3").value;
    const url = `/recommend/cf/personal?${toParams({ userId, k })}`;
    try {
      const data = await jsonFetch(url);
      renderList("cf-personal-results", data.results, "cf-personal");
    } catch (err) {
      console.error(err);
      renderList("cf-personal-results", [], "cf-personal");
    }
  });

  // 4) Hybrid
  document.getElementById("form-hybrid").addEventListener("submit", async (e) => {
    e.preventDefault();
    const params = {
      userId: document.getElementById("userHybrid").value,
      k: document.getElementById("k4").value,
      alpha: document.getElementById("alpha").value,
      mode: document.getElementById("modeHybrid").value,
      minRating: document.getElementById("minRatingHybrid").value,
      include: document.getElementById("include").value,
      exclude: document.getElementById("exclude").value,
      yearMin: document.getElementById("yearMin").value,
      yearMax: document.getElementById("yearMax").value,
      minCount: document.getElementById("minCount").value,
      mmr: "true",
      mmrLambda: document.getElementById("mmrLambda").value,
      novelty: document.getElementById("novelty").value,
      cand: document.getElementById("cand").value
    };
    const url = `/recommend/hybrid/personal?${toParams(params)}`;
    try {
      const data = await jsonFetch(url);
      // data.results ist bereits fein strukturiert
      renderList("hybrid-results", data.results, "hybrid");
    } catch (err) {
      console.error(err);
      renderList("hybrid-results", [], "hybrid");
    }
  });

});
