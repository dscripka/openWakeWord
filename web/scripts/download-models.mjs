#!/usr/bin/env node
// Downloads the ONNX versions of the openWakeWord models into ./models so the
// browser port can run fully client-side. These are the exact same model files
// used by the Python package (the .onnx assets from the v0.5.1 GitHub release).

import { mkdir, writeFile, access } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BASE =
  "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1";

// Feature models (always required) + the pre-trained wake word models.
const MODELS = [
  "melspectrogram.onnx",
  "embedding_model.onnx",
  "silero_vad.onnx",
  "alexa_v0.1.onnx",
  "hey_mycroft_v0.1.onnx",
  "hey_jarvis_v0.1.onnx",
  "hey_rhasspy_v0.1.onnx",
  "timer_v0.1.onnx",
  "weather_v0.1.onnx",
];

// Models live in web/models; this script lives in web/scripts.
const outDir = join(dirname(fileURLToPath(import.meta.url)), "..", "models");

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function download(name) {
  const dest = join(outDir, name);
  if (await exists(dest)) {
    console.log(`✓ ${name} (already present)`);
    return;
  }
  const url = `${BASE}/${name}`;
  process.stdout.write(`↓ ${name} ... `);
  const res = await fetch(url, { redirect: "follow" });
  if (!res.ok) {
    throw new Error(`failed to download ${url} (HTTP ${res.status})`);
  }
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(dest, buf);
  console.log(`done (${(buf.length / 1024).toFixed(0)} KB)`);
}

await mkdir(outDir, { recursive: true });
for (const m of MODELS) {
  await download(m);
}
console.log(`\nModels saved to ${outDir}`);
