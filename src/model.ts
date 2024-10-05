import ort from "onnxruntime-web";
import { ONNXWrapper } from "./ONNXWrapper";
import { RawInferenceResult } from "./Gliclass";

abstract class Model {
  constructor(
    public config: any,
    public processor: any,
    public decoder: any,
    public onnxWrapper: ONNXWrapper,
  ) {}

  async initialize(): Promise<void> {
    await this.onnxWrapper.init();
  }

  abstract prepareInputs(batch: any): Record<string, ort.Tensor>;

};


export class UniEncoderModel extends Model {
  prepareInputs(batch: any): Record<string, ort.Tensor> {
    const batch_size: number = batch.inputsIds.length;
    const num_tokens: number = batch.inputsIds[0].length;

    const createTensor = (data: any[], shape: number[], tensorType: any = "int64"): ort.Tensor => {
      // @ts-ignore // NOTE: node types not working
      return new this.onnxWrapper.ort.Tensor(tensorType, data.flat(Infinity), shape);
    };

    let input_ids: ort.Tensor = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask: ort.Tensor = createTensor(batch.attentionMasks, [batch_size, num_tokens]); 

    const feeds: Record<string, ort.Tensor> = {
      input_ids: input_ids,
      attention_mask: attention_mask
    };

    return feeds;
  }

  async inference(
    texts: string[],
    labels: string[]|string[][],
    batchSize: number = 8,
    threshold: number = 0.5,
    multiLabel: boolean = false,
  ): Promise<RawInferenceResult> {
    const sameLabels: boolean = typeof labels[0] === 'string';
    let output: RawInferenceResult = [];

    for (let idx = 0; idx < texts.length; idx += batchSize) {
      const batchTexts = texts.slice(idx, idx + batchSize);
      const currLabels = sameLabels ? labels : labels.slice(idx, idx + batchTexts.length);
      const maxLabelsCount = sameLabels 
        ? labels.length // If labels are the same for all texts, just get the length
        : Math.max(...labels.map(labelSet => labelSet.length)); 

      const batch = this.processor.prepareBatch(batchTexts, currLabels);
      const feeds = this.prepareInputs(batch);

      const results: Record<string, ort.Tensor> = await this.onnxWrapper.run(feeds);
      const modelOutput: any = results["logits"].data;

      const currBatchSize: number = batchTexts.length;

      const decodedLabels: RawInferenceResult = this.decoder.decode(
        currBatchSize,
        maxLabelsCount,
        currLabels,
        batch.idToClass,
        modelOutput,
        sameLabels,
        multiLabel,
        threshold,
      );
      output = output.concat(decodedLabels);
    }
  return output;
  }
}