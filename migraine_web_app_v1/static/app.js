// Shared front-end helpers: slider fill, symptom chips, donut chart, prediction call.

function paintSlider(input) {
  const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--fill", pct + "%");
  const out = document.querySelector(`output[for="${input.id}"]`);
  if (out) out.textContent = input.value;
}

function initForm(form) {
  form.querySelectorAll("input[type=range]").forEach((input) => {
    paintSlider(input);
    input.addEventListener("input", () => paintSlider(input));
  });
  form.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const on = chip.getAttribute("aria-pressed") === "true";
      chip.setAttribute("aria-pressed", String(!on));
    });
  });
}

function readForm(form) {
  const data = {};
  form.querySelectorAll("input[type=range]").forEach((i) => (data[i.name] = Number(i.value)));
  form.querySelectorAll(".chip").forEach((c) => (data[c.dataset.name] = c.getAttribute("aria-pressed") === "true" ? 1 : 0));
  form.querySelectorAll("input[type=hidden]").forEach((i) => (data[i.name] = Number(i.value)));
  return data;
}

// items: [{label, value (0-100), color}]
function renderDonut(svg, items) {
  const r = 150, c = 2 * Math.PI * r;
  svg.innerHTML = "";
  let offset = 0;
  items.forEach((item) => {
    const len = (item.value / 100) * c;
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", 180); circle.setAttribute("cy", 180); circle.setAttribute("r", r);
    circle.setAttribute("fill", "none"); circle.setAttribute("stroke", item.color);
    circle.setAttribute("stroke-width", 44);
    circle.setAttribute("stroke-dasharray", `0 ${c}`);
    circle.setAttribute("stroke-dashoffset", -offset);
    svg.appendChild(circle);
    requestAnimationFrame(() => requestAnimationFrame(() =>
      circle.setAttribute("stroke-dasharray", `${len} ${c - len}`)));
    offset += len;
  });
}

function renderBars(el, items) {
  el.innerHTML = items.map((item) => `
    <div>
      <div class="bar-top"><span>${item.label}</span><b>${item.value.toFixed(1)}%</b></div>
      <div class="bar"><i style="background:${item.color}" data-w="${item.value}"></i></div>
    </div>`).join("");
  requestAnimationFrame(() => requestAnimationFrame(() =>
    el.querySelectorAll(".bar i").forEach((i) => (i.style.width = i.dataset.w + "%"))));
}

async function predict(task, payload) {
  const res = await fetch(`/api/predict/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}
