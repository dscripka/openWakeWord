// End-to-end verification of the browser port, run under Node with the same
// onnxruntime-web package. Streams real test clips through the model exactly
// like Python's `Model.predict_clip` and checks that the right wake word fires.

import { readFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { OpenWakeWord, configureOrt } from "../src/openwakeword.js";

// This script lives in web/test. The web package root is one level up, and the
// repository root (which holds tests/data/*.wav) is two levels up.
const here = dirname(fileURLToPath(import.meta.url));
const webRoot = join(here, "..");
const repo = join(here, "..", "..");
const modelsDir = join(webRoot, "models") + "/";

configureOrt({ numThreads: 1 });

// Minimal 16-bit PCM mono WAV reader -> Int16Array of samples.
function readWavInt16(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let off = 12; // skip RIFF header
  let dataOffset = -1;
  let dataLen = 0;
  while (off + 8 <= dv.byteLength) {
    const id = String.fromCharCode(
      dv.getUint8(off), dv.getUint8(off + 1), dv.getUint8(off + 2), dv.getUint8(off + 3)
    );
    const size = dv.getUint32(off + 4, true);
    if (id === "data") {
      dataOffset = off + 8;
      dataLen = size;
      break;
    }
    off += 8 + size + (size % 2);
  }
  if (dataOffset < 0) throw new Error("no data chunk");
  return new Int16Array(buf.buffer, buf.byteOffset + dataOffset, dataLen / 2);
}

async function predictClip(oww, samples, { padding = 1, chunk = 1280 } = {}) {
  const pad = 16000 * padding;
  const data = new Int16Array(pad + samples.length + pad);
  data.set(samples, pad);

  const maxScores = {};
  for (let i = 0; i + chunk < data.length; i += chunk) {
    const frame = data.slice(i, i + chunk);
    const scores = await oww.predict(frame);
    for (const [k, v] of Object.entries(scores)) {
      maxScores[k] = Math.max(maxScores[k] ?? 0, v);
    }
  }
  return maxScores;
}

const CASES = [
  { file: "tests/data/alexa_test.wav", expect: "alexa" },
  { file: "tests/data/hey_mycroft_test.wav", expect: "hey_mycroft" },
];

const oww = await OpenWakeWord.create({
  baseUrl: modelsDir,
  wakewordModels: ["alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy"],
});
console.log("Loaded models:", oww.modelNames.join(", "));
console.log(
  "Input frames per model:",
  Object.fromEntries(Object.entries(oww.models).map(([k, v]) => [k, v.inputFrames]))
);

let allPass = true;
for (const { file, expect } of CASES) {
  const buf = await readFile(join(repo, file));
  const samples = readWavInt16(buf);
  await oww.reset();
  const scores = await predictClip(oww, samples);
  const top = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const pass = scores[expect] >= 0.5 && top[0][0] === expect;
  allPass = allPass && pass;
  console.log(`\n${file}`);
  console.log("  max scores:", Object.fromEntries(top.map(([k, v]) => [k, +v.toFixed(3)])));
  console.log(`  expected "${expect}" -> ${pass ? "PASS" : "FAIL"} (score ${(scores[expect] ?? 0).toFixed(3)})`);
}

console.log(`\n${allPass ? "ALL PASS ✅" : "FAILURES ❌"}`);
process.exit(allPass ? 0 : 1);
