const $ = (id) => document.getElementById(id);
const drop = $("drop"), file = $("file"), go = $("go"), err = $("error");
let current = null;

function pick(f) {
  if (!f || !f.type.startsWith("image/")) return;
  current = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").hidden = false;
  $("hint").hidden = true;
  go.disabled = false;
  err.hidden = true;
  $("result").hidden = true;
}

file.onchange = () => pick(file.files[0]);
["dragover", "dragenter"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("over")));
drop.addEventListener("drop", (ev) => { ev.preventDefault(); pick(ev.dataTransfer.files[0]); });

go.onclick = async () => {
  go.disabled = true;
  go.textContent = "Detecting…";
  err.hidden = true;
  const fd = new FormData();
  fd.append("image", current);
  try {
    const res = await fetch("/api/detect", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Detection failed.");
    $("label").textContent = data.label;
    $("conf").textContent = `${data.confidence}% match`;
    $("kp").textContent = `${data.keypoints} keypoints`;
    $("annotated").src = data.annotated;
    $("features").src = data.features;
    $("result").hidden = false;
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    go.disabled = false;
    go.textContent = "Detect";
  }
};
