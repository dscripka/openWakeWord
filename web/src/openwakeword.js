import * as ort from "onnxruntime-web";
import { AudioFeatures, CHUNK } from "./audio-features.js";
import {
  FEATURE_MODELS,
  PRETRAINED_MODELS,
  MODEL_CLASS_MAPPINGS,
} from "./models.js";

const DEFAULT_INPUT_FRAMES = 16; // openWakeWord models use 16 feature frames

/**
 * Configure the ONNX Runtime Web environment. Call once before creating a model
 * if you want to point at self-hosted wasm binaries or tweak threading.
 * @param {{wasmPaths?: string, numThreads?: number, simd?: boolean}} opts
 */
export function configureOrt(opts = {}) {
  if (opts.wasmPaths !== undefined) ort.env.wasm.wasmPaths = opts.wasmPaths;
  if (opts.numThreads !== undefined) ort.env.wasm.numThreads = opts.numThreads;
  if (opts.simd !== undefined) ort.env.wasm.simd = opts.simd;
}

function readShapeDim(session, which, idx) {
  // Best-effort read of an input/output dimension across ort-web versions.
  const meta =
    which === "input"
      ? session.inputMetadata?.[0]
      : session.outputMetadata?.[0];
  const shape = meta?.shape ?? meta?.dimensions;
  const v = shape?.[idx];
  return typeof v === "number" && v > 0 ? v : null;
}

/**
 * Native browser port of `openwakeword.Model`.
 *
 * Runs the full melspectrogram -> embedding -> wake word pipeline client-side
 * using ONNX Runtime Web. No server required.
 */
export class OpenWakeWord {
  constructor() {
    this.models = {}; // name -> { session, inputName, inputFrames, outputClasses, classMapping }
    this.features = null;
    this.predictionBuffer = {}; // label -> number[] (max 30)
    this.threshold = 0.5;
    this.onDetection = null;
  }

  /**
   * Create and initialise a model.
   *
   * @param {object} opts
   * @param {string} [opts.baseUrl="./models/"] Base URL/path for model files.
   * @param {Array<string|{name:string,url:string,inputFrames?:number,classMapping?:object}>} [opts.wakewordModels]
   *        Wake word models to load. Strings are looked up in the pre-trained
   *        registry (e.g. "hey_jarvis"); objects allow custom models by URL.
   *        Defaults to all pre-trained models.
   * @param {string} [opts.melspectrogramUrl] Override the melspectrogram model URL.
   * @param {string} [opts.embeddingUrl] Override the embedding model URL.
   * @param {string[]} [opts.executionProviders=["wasm"]] ORT execution providers.
   * @param {object} [opts.ort] Options forwarded to {@link configureOrt}.
   * @returns {Promise<OpenWakeWord>}
   */
  static async create(opts = {}) {
    const {
      baseUrl = "./models/",
      wakewordModels = Object.keys(PRETRAINED_MODELS),
      executionProviders = ["wasm"],
      ort: ortOpts,
      threshold = 0.5,
      onDetection = null,
    } = opts;

    if (ortOpts) configureOrt(ortOpts);

    const join = (file) =>
      /^https?:|^\.|^\//.test(file) ? file : baseUrl + file;
    const sessOpts = { executionProviders };

    const melspectrogramUrl = opts.melspectrogramUrl
      ? opts.melspectrogramUrl
      : join(FEATURE_MODELS.melspectrogram);
    const embeddingUrl = opts.embeddingUrl
      ? opts.embeddingUrl
      : join(FEATURE_MODELS.embedding);

    const self = new OpenWakeWord();

    // Load feature models + create the streaming feature extractor.
    const [melspecSession, embeddingSession] = await Promise.all([
      ort.InferenceSession.create(melspectrogramUrl, sessOpts),
      ort.InferenceSession.create(embeddingUrl, sessOpts),
    ]);
    self.features = new AudioFeatures(melspecSession, embeddingSession);

    // Load wake word models.
    for (const entry of wakewordModels) {
      let name, url, inputFrames, classMapping;
      if (typeof entry === "string") {
        name = entry;
        const file = PRETRAINED_MODELS[entry] || entry;
        url = join(file);
        classMapping = MODEL_CLASS_MAPPINGS[entry];
      } else {
        name = entry.name;
        url = /^https?:|^\.|^\//.test(entry.url) ? entry.url : join(entry.url);
        inputFrames = entry.inputFrames;
        classMapping = entry.classMapping || MODEL_CLASS_MAPPINGS[name];
      }

      const session = await ort.InferenceSession.create(url, sessOpts);
      const detectedFrames = readShapeDim(session, "input", 1);
      const outputClasses = readShapeDim(session, "output", 1) ?? 1;
      self.models[name] = {
        session,
        inputName: session.inputNames[0],
        inputFrames: inputFrames ?? detectedFrames ?? DEFAULT_INPUT_FRAMES,
        outputClasses,
        classMapping: classMapping || null,
      };
    }

    self.threshold = threshold;
    self.onDetection = onDetection;

    await self.features.warmup();
    return self;
  }

  /** Names of the loaded wake word models. */
  get modelNames() {
    return Object.keys(this.models);
  }

  /** Reset all streaming/prediction state. */
  async reset() {
    this.features.reset(true);
    await this.features.warmup();
    this.predictionBuffer = {};
  }

  async _runModel(m, feat) {
    const tensor = new ort.Tensor("float32", feat.data, feat.dims);
    const out = await m.session.run({ [m.inputName]: tensor });
    const data = out[m.session.outputNames[0]].data;
    return Array.from(data); // length = outputClasses (the [0] row)
  }

  _pushPrediction(label, value) {
    if (!this.predictionBuffer[label]) this.predictionBuffer[label] = [];
    this.predictionBuffer[label].push(value);
    if (this.predictionBuffer[label].length > 30) {
      this.predictionBuffer[label].shift();
    }
  }

  /**
   * Predict wake word scores for a frame of 16-bit PCM @ 16 kHz audio.
   * Ideally pass multiples of 1280 samples (80 ms).
   *
   * @param {Int16Array} x
   * @returns {Promise<Record<string, number>>} label -> score (0..1)
   */
  async predict(x) {
    if (!(x instanceof Int16Array)) {
      throw new TypeError("Input audio (x) must be an Int16Array of 16 kHz PCM.");
    }

    const nPrepared = await this.features.streamingFeatures(x);
    const predictions = {};

    for (const [name, m] of Object.entries(this.models)) {
      let prediction; // array of length outputClasses

      if (nPrepared > CHUNK) {
        const group = [];
        for (let i = Math.floor(nPrepared / CHUNK) - 1; i >= 0; i--) {
          const feat = this.features.getFeatures(
            m.inputFrames,
            -m.inputFrames - i
          );
          group.push(await this._runModel(m, feat));
        }
        prediction = group.reduce((acc, row) =>
          acc.map((v, idx) => Math.max(v, row[idx]))
        );
      } else if (nPrepared === CHUNK) {
        const feat = this.features.getFeatures(m.inputFrames);
        prediction = await this._runModel(m, feat);
      } else {
        // Not enough new samples yet: reuse the previous prediction.
        if (m.outputClasses === 1) {
          const buf = this.predictionBuffer[name];
          prediction = [buf && buf.length > 0 ? buf[buf.length - 1] : 0];
        } else {
          prediction = new Array(m.outputClasses).fill(0);
        }
      }

      if (m.outputClasses === 1) {
        predictions[name] = prediction[0];
      } else if (m.classMapping) {
        for (const [intLabel, cls] of Object.entries(m.classMapping)) {
          predictions[cls] = prediction[Number.parseInt(intLabel, 10)];
        }
      } else {
        for (let c = 0; c < m.outputClasses; c++) {
          predictions[`${name}_${c}`] = prediction[c];
        }
      }
    }

    // Zero out predictions for the first 5 frames during model warm-up.
    for (const label of Object.keys(predictions)) {
      if (!this.predictionBuffer[label] || this.predictionBuffer[label].length < 5) {
        predictions[label] = 0;
      }
    }

    // Update the prediction history buffers.
    for (const label of Object.keys(predictions)) {
      this._pushPrediction(label, predictions[label]);
    }

    // Fire detection callback for any label that meets the threshold.
    if (this.onDetection) {
      for (const [label, score] of Object.entries(predictions)) {
        if (score >= this.threshold) {
          this.onDetection({ label, score });
        }
      }
    }

    return predictions;
  }
}

export { AudioFeatures } from "./audio-features.js";
export * from "./models.js";
export default OpenWakeWord;
