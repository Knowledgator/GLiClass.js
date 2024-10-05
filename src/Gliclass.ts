import { AutoTokenizer, env } from "@xenova/transformers";
import { UniEncoderModel } from "./model";
import { UniEncoderProcessor } from "./processor";
import { UniEncoderDecoder } from "./decoder";
import { IONNXSettings, ONNXWrapper } from "./ONNXWrapper";

export interface ITransformersSettings {
  allowLocalModels: boolean;
  useBrowserCache: boolean;
}

export interface InitConfig {
  tokenizerPath: string;
  onnxSettings: IONNXSettings;
  transformersSettings?: ITransformersSettings;
  promptFirst?: boolean;
}

export interface IInference {
  texts: string[];
  labels: string[]|string[][];
  batchSize?: number;
  threshold?: number;
  multiLabel?: boolean;

}

export type RawInferenceResult = [string, number][][];

export interface IClassificationResult {
  label: string;
  score: number;
}
export type InferenceResultSingle = IClassificationResult[];
export type InferenceResultMultiple = InferenceResultSingle[];

export class Gliclass {
  private model: UniEncoderModel | null = null;

  constructor(private config: InitConfig) {
    env.allowLocalModels = config.transformersSettings?.allowLocalModels ?? false;
    env.useBrowserCache = config.transformersSettings?.useBrowserCache ?? false;

    this.config = { ...config, promptFirst: config.promptFirst || true };
  }

  async initialize(): Promise<void> {
    const { tokenizerPath, onnxSettings, promptFirst } = this.config;

    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
    console.log("Tokenizer loaded.");
    const onnxWrapper = new ONNXWrapper(onnxSettings);

    const processor = new UniEncoderProcessor({ promptFirst: promptFirst }, tokenizer);
    const decoder = new UniEncoderDecoder({ promptFirst: promptFirst });

    this.model = new UniEncoderModel({ promptFirst: promptFirst }, processor, decoder, onnxWrapper);

    await this.model.initialize();
  }

  async inference({
    texts,
    labels,
    batchSize = 8,
    threshold = 0.5,
    multiLabel = false,
  }: IInference): Promise<InferenceResultMultiple> {
    if (!this.model) {
      throw new Error("Model is not initialized. Call initialize() first.");
    }

    const result = await this.model.inference(texts, labels, batchSize, threshold, multiLabel);
    return this.mapRawResultToResponse(result);
  }

  mapRawResultToResponse(rawResult: RawInferenceResult): InferenceResultMultiple {
    const response: InferenceResultMultiple = [];
    for (const individualResult of rawResult) {
      const classificationResult: IClassificationResult[] = individualResult.map(
        ([label, score]) => ({
          label,
          score,
        }),
      );
      response.push(classificationResult);
    }

    return response;
  }
}
