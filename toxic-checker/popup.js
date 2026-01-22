// Default to new model
let useNewModel = true;

const API_BASE = "https://slip-off-tongue.onrender.com";

function fillExample(type) {
  const input = document.getElementById("commentInput");
  if (type === "clean") {
    input.value = "I disagree with your point, but I respect your opinion. Let's discuss calmly.";
  } else {
    input.value = "Stop talking, you're embarrassing yourself. This is pathetic.";
  }
}

function clearAll() {
  document.getElementById("commentInput").value = "";
  setOverall(0, "—");

  document.querySelectorAll("[data-badge]").forEach(el => {
    el.textContent = "0%";
    el.classList.remove("badge-green", "badge-red");
    el.classList.add("badge-neutral");
  });

  document.querySelectorAll("[data-percent]").forEach(el => {
    el.textContent = "Low";
    el.className = "percent percent-green";
  });
}

function setLoading(isLoading) {
  const btn = document.getElementById("analyzeBtn");
  const loader = document.getElementById("loader");
  if (isLoading) {
    btn.disabled = true;
    btn.textContent = "Analyzing...";
    loader.style.display = "flex";
  } else {
    btn.disabled = false;
    btn.textContent = "Analyze";
    loader.style.display = "none";
  }
}

function setOverall(overallPct, risk) {
  const pctEl = document.getElementById("overallPct");
  const riskEl = document.getElementById("riskLabel");

  pctEl.textContent = Number(overallPct).toFixed(2) + "%";
  riskEl.textContent = risk || "—";

  pctEl.classList.remove("low", "med", "high", "crit", "neutral");
  riskEl.classList.remove("low", "med", "high", "crit", "neutral");

  const p = Number(overallPct);
  if (!isFinite(p)) {
    pctEl.classList.add("neutral");
    riskEl.classList.add("neutral");
    return;
  }

  if (p < 20) { pctEl.classList.add("low"); riskEl.classList.add("low"); }
  else if (p < 50) { pctEl.classList.add("med"); riskEl.classList.add("med"); }
  else if (p < 80) { pctEl.classList.add("high"); riskEl.classList.add("high"); }
  else { pctEl.classList.add("crit"); riskEl.classList.add("crit"); }
}

async function analyze() {
  const input = document.getElementById("commentInput");
  const comment = (input.value || "").trim();

  if (!comment) {
    alert("Please enter or select a comment.");
    return;
  }

  setLoading(true);

  try {
    const url = useNewModel 
      ? `${API_BASE}/predict_new`
      : `${API_BASE}/predict`;

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comment })
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.error || "Prediction failed");
      return;
    }

    const percent = data.percent || {};
    const threshold = data.threshold ?? 50.0;

    setOverall(data.overall ?? 0, data.risk ?? "—");

    document.querySelectorAll("[data-badge]").forEach(el => {
      const key = el.getAttribute("data-badge");
      const pct = Number(percent[key] ?? 0);

      el.textContent = pct.toFixed(2) + "%";
      el.classList.remove("badge-neutral", "badge-green", "badge-red");
      el.classList.add(pct >= threshold ? "badge-red" : "badge-green");
    });

    document.querySelectorAll("[data-percent]").forEach(el => {
      const key = el.getAttribute("data-percent");
      const pct = Number(percent[key] ?? 0);

      if (pct >= threshold) {
        el.textContent = "High";
        el.className = "percent percent-red";
      } else {
        el.textContent = "Low";
        el.className = "percent percent-green";
      }
    });

  } catch (e) {
    alert("Could not connect.\n\n• Render may be sleeping (wait 10–30 sec)\n• Check internet\n• Console for details (right-click popup → Inspect)");
    console.error("Fetch error:", e);
  } finally {
    setLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("loader").style.display = "none";

  document.getElementById("analyzeBtn").addEventListener("click", analyze);
  document.getElementById("cleanBtn").addEventListener("click", () => fillExample("clean"));
  document.getElementById("toxicBtn").addEventListener("click", () => fillExample("toxic"));
  document.getElementById("clearBtn").addEventListener("click", clearAll);

  // Model toggle
  const switchInput = document.getElementById("modelSwitch");
  const statusText = document.getElementById("modelStatus");

  switchInput.checked = true;  // default new model
  statusText.innerHTML = 'Using <strong>New Model</strong> (better)';

  switchInput.addEventListener("change", () => {
    useNewModel = switchInput.checked;
    statusText.innerHTML = useNewModel 
      ? 'Using <strong>New Model</strong> (better)'
      : 'Using <strong>Classic Model</strong>';
  });

  // Auto-fill selected text and analyze
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      func: () => window.getSelection().toString().trim()
    }, (results) => {
      const selected = results[0].result;
      if (selected) {
        document.getElementById("commentInput").value = selected;
        analyze();
      }
    });
  });
});