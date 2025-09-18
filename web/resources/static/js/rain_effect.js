const rainCanvas = document.createElement("canvas");
rainCanvas.style.position = "fixed";
rainCanvas.style.top = 0;
rainCanvas.style.left = 0;
rainCanvas.style.width = "100%";
rainCanvas.style.height = "100%";
rainCanvas.style.pointerEvents = "none";
rainCanvas.style.zIndex = 999;
document.body.appendChild(rainCanvas);

const ctx = rainCanvas.getContext("2d");
let width = (rainCanvas.width = window.innerWidth);
let height = (rainCanvas.height = window.innerHeight);

window.addEventListener("resize", () => {
  width = rainCanvas.width = window.innerWidth;
  height = rainCanvas.height = window.innerHeight;
});

const raindrops = Array.from({ length: 100 }).map(() => ({
  x: Math.random() * width,
  y: Math.random() * height,
  length: Math.random() * 20 + 10,
  speed: Math.random() * 5 + 5,
  opacity: Math.random() * 0.5 + 0.3
}));

function drawRain() {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(173,216,230,0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const drop of raindrops) {
    ctx.moveTo(drop.x, drop.y);
    ctx.lineTo(drop.x, drop.y + drop.length);
  }
  ctx.stroke();
}

function updateRain() {
  for (const drop of raindrops) {
    drop.y += drop.speed;
    if (drop.y > height) {
      drop.y = -drop.length;
      drop.x = Math.random() * width;
    }
  }
}

function animateRain() {
  drawRain();
  updateRain();
  requestAnimationFrame(animateRain);
}

animateRain();
