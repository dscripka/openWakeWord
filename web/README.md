# openWakeWord — native browser port

A native **JavaScript / browser** port of [openWakeWord](../README.md). Wake word
detection runs **fully client-side** in the browser using
[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) —
no Python server, no websocket streaming.

It loads the exact same ONNX models as the Python package (melspectrogram,
speech embedding, and the pre-trained wake word models) and faithfully
reimplements the streaming feature pipeline (`AudioFeatures`) and prediction
logic (`Model.predict`) in JavaScript.

> Looking for the server-based approach instead? See
> `[examples/web](../examples/web)` for streaming microphone audio over a
> websocket into openWakeWord running in a Python backend.



## Pipeline

```
mic (16 kHz, 16-bit PCM)
  └─ melspectrogram.onnx   → mel features (32 bins), transformed x/10 + 2
       └─ embedding_model.onnx → 96-dim speech embeddings (76-frame windows, step 8)
            └─ <wakeword>.onnx  → detection score 0..1
```



## Quick start

```bash
cd web
npm install                 # installs onnxruntime-web
npm run download-models     # fetches the ONNX models into ./models
npm run serve               # or any static file server
# open http://localhost:8080/demo/
```

Then open the page, click **Start Listening**, and say *"hey jarvis"*,
*"alexa"*, *"hey mycroft"*, or *"hey rhasspy"*.

> The demo loads `onnxruntime-web` from a CDN via an
> [import map](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/script/type/importmap),
> so it works without a bundler. A static server is still required because ES
> modules and the AudioWorklet cannot be loaded from `file://`.



### Two hard requirements (browser rules)

- **Must be served from** `http://localhost` **or** `https://`**.** ES modules,
`AudioWorklet`, and microphone access all require a real, secure origin, so
opening the file directly via `file://` will not work.
- **Model files must be same-origin** (or served with CORS headers). The
upstream GitHub release assets do **not** send `Access-Control-Allow-Origin`,
so you cannot point `baseUrl` directly at the GitHub release — host the
`.onnx` files yourself (which is what `npm run download-models` sets up).



## Library usage

```bash
npm install openwakeword-web
```

```js
import { OpenWakeWord } from "openwakeword-web";
import { Microphone } from "openwakeword-web/microphone";

const oww = await OpenWakeWord.create({
  baseUrl: "./models/",
  wakewordModels: ["hey_jarvis", "alexa"], // or omit for all pre-trained models
  threshold: 0.5,
  onDetection: ({ label, score }) => {
    console.log(`Wake word detected: ${label} (score ${score.toFixed(2)})`);
  },
});

const mic = new Microphone(async (frame) => {
  // frame is an Int16Array of 1280 samples (80 ms @ 16 kHz)
  await oww.predict(frame); // onDetection fires automatically when threshold is met
});

await mic.start();
```

`predict()` accepts any `Int16Array` of 16 kHz PCM audio (ideally multiples of
1280 samples) and returns a `{ label: score }` map, matching the Python API.
The `onDetection` callback fires inside `predict()` for every label whose score
meets `threshold` — so you don't have to poll the returned scores yourself.

Both `threshold` and `onDetection` can be updated at runtime:

```js
oww.threshold = 0.75;
oww.onDetection = ({ label }) => triggerAssistant(label);
```

If you prefer the polling style, simply omit `onDetection` and inspect the
scores returned by `predict()` directly:

```js
const scores = await oww.predict(frame);
if (scores["hey_jarvis"] >= 0.5) console.log("detected hey jarvis!");
```

> **Bundlers:** `Microphone` loads its AudioWorklet via
> `new URL("./mic-worklet.js", import.meta.url)`, which Vite and webpack 5
> handle automatically. If your bundler does not emit the worklet asset, copy
> `mic-worklet.js` somewhere same-origin and pass it explicitly:
> `new Microphone(onFrame, { workletUrl: "/mic-worklet.js" })`.

> TypeScript type definitions (`.d.ts`) ship with the package, so the API is
> fully typed out of the box.



### Usage in React (Vite)

Place the `.onnx` model files in `public/assets/models/` so Vite serves them
as static assets, then use a `useEffect` hook to own the lifecycle:

```tsx
// src/hooks/useWakeWord.ts
import { useEffect, useRef } from "react";
import { OpenWakeWord, configureOrt } from "openwakeword-web";
import { Microphone } from "openwakeword-web/microphone";

export function useWakeWord(
  onDetection: (label: string, score: number) => void,
  wakewordModels = ["hey_jarvis", "alexa"],
  threshold = 0.5,
) {
  // Keep a stable ref so the callback never causes re-initialisation.
  const onDetectionRef = useRef(onDetection);
  onDetectionRef.current = onDetection;

  useEffect(() => {
    let oww: OpenWakeWord | null = null;
    let mic: Microphone | null = null;
    let cancelled = false;

    configureOrt({ numThreads: 1 }); // avoid COOP/COEP requirement

    OpenWakeWord.create({
      baseUrl: "/assets/models/",
      wakewordModels,
      threshold,
      onDetection: ({ label, score }) => onDetectionRef.current(label, score),
    }).then((instance) => {
      if (cancelled) return;
      oww = instance;
      mic = new Microphone(async (frame) => {
        await oww!.predict(frame);
      });
      mic.start();
    });

    return () => {
      cancelled = true;
      mic?.stop();
      oww?.reset();
    };
  }, [threshold, ...wakewordModels]); // re-init if config changes
}
```

```tsx
// src/App.tsx
import { useCallback } from "react";
import { useWakeWord } from "./hooks/useWakeWord";

export default function App() {
  const handleDetection = useCallback((label: string, score: number) => {
    console.log(`Detected: ${label} (${score.toFixed(2)})`);
  }, []);

  useWakeWord(handleDetection, ["hey_jarvis", "alexa"], 0.5);

  return <div>Listening for wake words…</div>;
}
```

**Model files** — copy the `.onnx` files into `public/assets/models/`:

```
public/
  assets/
    models/
      melspectrogram.onnx
      embedding_model.onnx
      hey_jarvis_v0.1.onnx
      alexa_v0.1.onnx
```

You can copy them from the `web/models/` directory after running
`npm run download-models` in this package, or point `baseUrl` at any
same-origin path that serves the files with correct CORS headers.



### Custom models

```js
const oww = await OpenWakeWord.create({
  wakewordModels: [
    "alexa",                                   // pre-trained, by name
    { name: "my_word", url: "/models/my_word.onnx" }, // your own trained model
  ],
});
```



### Using your own ONNX runtime / wasm hosting

```js
import { configureOrt } from "./src/openwakeword.js";
configureOrt({ wasmPaths: "/ort/", numThreads: 1 });
```

Multi-threaded wasm requires the page to be cross-origin isolated
(`COOP`/`COEP` headers). The demo uses `numThreads: 1` to avoid that
requirement.

## Layout


| Path                          | Purpose                                                 |
| ----------------------------- | ------------------------------------------------------- |
| `src/openwakeword.js`         | Main `OpenWakeWord` class — model loading + `predict()` |
| `src/audio-features.js`       | Streaming melspectrogram + embedding pipeline           |
| `src/models.js`               | Pre-trained model registry + class mappings             |
| `src/microphone.js`           | Mic capture helper (16 kHz `AudioContext` + worklet)    |
| `src/mic-worklet.js`          | AudioWorklet: float → 16-bit PCM framing                |
| `demo/index.html`             | Live detection demo UI (uses the `src/` modules)        |
| `scripts/download-models.mjs` | Downloads the ONNX models from the GitHub release       |
| `test/verify.mjs`             | Node script that runs the port on the repo's test clips |




## Verification

`test/verify.mjs` streams the repository's test clips
(`tests/data/alexa_test.wav`, `tests/data/hey_mycroft_test.wav`) through the
port under Node and checks the correct wake word fires:

```bash
cd web
npm test
# alexa_test.wav       → alexa = 1.000        PASS
# hey_mycroft_test.wav → hey_mycroft = 1.000  PASS
```



## Browser support

Requires `AudioWorklet`, ES modules, and `AudioContext({ sampleRate: 16000 })`
— supported by current Chrome, Edge, Firefox, and Safari. A secure context
(`https://` or `localhost`) is required for microphone access.

## License

Apache-2.0, same as openWakeWord. The bundled model files are downloaded from
the upstream openWakeWord releases and retain their original licenses.