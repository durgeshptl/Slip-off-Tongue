function fillExample(type){
  const input = document.getElementById("commentInput");
  if(type === "clean"){
    input.value = "I disagree with your point, but I respect your opinion. Let's discuss calmly.";
  } else {
    input.value = "Stop talking, you're embarrassing yourself. This is pathetic.";
  }
}

function clearAll(){
  document.getElementById("commentInput").value = "";

  // reset overall
  setOverall(0, "—");

  // reset badges
  document.querySelectorAll("[data-badge]").forEach(el=>{
    el.textContent = "0%";
    el.classList.remove("badge-green","badge-red");
    el.classList.add("badge-neutral");
  });

  // reset low/high
  document.querySelectorAll("[data-percent]").forEach(el=>{
    el.textContent = "Low";
    el.className = "percent percent-green";
  });
}

function setLoading(isLoading){
  const btn = document.getElementById("analyzeBtn");
  const loader = document.getElementById("loader");
  if(isLoading){
    btn.disabled = true;
    btn.textContent = "Analyzing...";
    loader.style.display = "flex";
  }else{
    btn.disabled = false;
    btn.textContent = "Analyze";
    loader.style.display = "none";
  }
}

function setOverall(overallPct, risk){
  const pctEl = document.getElementById("overallPct");
  const riskEl = document.getElementById("riskLabel");

  pctEl.textContent = Number(overallPct).toFixed(2) + "%";
  riskEl.textContent = risk || "—";

  pctEl.classList.remove("low","med","high","crit","neutral");
  riskEl.classList.remove("low","med","high","crit","neutral");

  const p = Number(overallPct);
  if(!isFinite(p)){
    pctEl.classList.add("neutral");
    riskEl.classList.add("neutral");
    return;
  }

  if(p < 20){ pctEl.classList.add("low"); riskEl.classList.add("low"); }
  else if(p < 50){ pctEl.classList.add("med"); riskEl.classList.add("med"); }
  else if(p < 80){ pctEl.classList.add("high"); riskEl.classList.add("high"); }
  else { pctEl.classList.add("crit"); riskEl.classList.add("crit"); }
}

async function analyze() {
  const input = document.getElementById("commentInput");
  const comment = (input.value || "").trim();

  if (!comment) {
    alert("Please enter a comment.");
    return;
  }

  setLoading(true);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ comment })
    });

    const data = await res.json();
    if (!res.ok) {
      alert(data.error || "Prediction failed");
      return;
    }

    const percent = data.percent || {};
    const threshold = typeof data.threshold === "number" ? data.threshold : 50.0;

    // overall + risk
    setOverall(data.overall ?? 0, data.risk ?? "—");

    // Update badges
    document.querySelectorAll("[data-badge]").forEach(el => {
      const key = el.getAttribute("data-badge");
      const pct = clampNumber(percent[key] ?? 0, 0, 100);

      el.textContent = pct.toFixed(2) + "%";

      el.classList.remove("badge-neutral","badge-green","badge-red");
      if (pct >= threshold) el.classList.add("badge-red");
      else el.classList.add("badge-green");
    });

    // Update Low/High text under badge
    document.querySelectorAll("[data-percent]").forEach(el => {
      const key = el.getAttribute("data-percent");
      const pct = clampNumber(percent[key] ?? 0, 0, 100);

      if (pct >= threshold) {
        el.textContent = "High";
        el.className = "percent percent-red";
      } else {
        el.textContent = "Low";
        el.className = "percent percent-green";
      }
    });

  } catch (e) {
    alert("Server error. Is Flask running?");
  } finally {
    setLoading(false);
  }
}

function clampNumber(x, lo, hi) {
  const n = Number(x);
  if (Number.isNaN(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

// hide loader on first load
document.addEventListener("DOMContentLoaded", () => setLoading(false));
