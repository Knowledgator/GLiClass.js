import { RawInferenceResult } from "./Gliclass";

const sigmoid = (x: number): number => {
  return 1 / (1 + Math.exp(-x));
};

const softmax = (logits: number[]): number[] => {
  const maxLogit = Math.max(...logits);

  const exps = logits.map(logit => Math.exp(logit - maxLogit));

  const sumExps = exps.reduce((a, b) => a + b, 0);

  return exps.map(exp => exp / sumExps);
};

// BaseDecoder class using abstract methods
abstract class BaseDecoder {
  config: any;

  constructor(config: any) {
    if (new.target === BaseDecoder) {
      throw new TypeError("Cannot instantiate an abstract class.");
    }
    this.config = config;
  }

  abstract decode(...args: any[]): any;
}

export class UniEncoderDecoder extends BaseDecoder {
  getLabel(labels:string[]|string[][], idToClass: Record<number, string>, i:number, j: number, sameLabels: boolean): string  {
    let label: string = '[Unknown]';
    if (sameLabels) {
      const labelsArray = labels as string[];
      label = labelsArray[j];
    } else {
      const labelsArray = labels as string[][];
      if (labelsArray[i] && labelsArray[i][j]) {
        label = labelsArray[i][j];
      }
    }
    if (!label) {
      label = idToClass[j] || '[Unknown]';
    }
    return label;
  }

  decode(
    batchSize: number,
    numLabels: number,
    labels: string[] | string[][],
    idToClass: Record<number, string>,
    logits: number[],
    sameLabels: boolean,
    multiLabel: boolean = false,
    threshold: number = 0.5,
  ): RawInferenceResult {
    const logits2D: number[][] = [];
    for (let i = 0; i < batchSize; i++) {
      const start = i * numLabels;
      const end = start + numLabels;
      logits2D.push(logits.slice(start, end));
    }

    const results: RawInferenceResult = [];

    for (let i = 0; i < batchSize; i++) {
      const logitsForText = logits2D[i];

      if (multiLabel) {
        // For multi-label classification
        const resultForText: [string, number][] = [];
        for (let j = 0; j < numLabels; j++) {
          const logit = logitsForText[j];
          const prob = sigmoid(logit);

          if (prob > threshold) {
            let label: string = this.getLabel(labels, idToClass, i, j, sameLabels);

            resultForText.push([label, prob]);
          }
        }
        results.push(resultForText);
      } else {
        // For single-label classification, use softmax
        const probs = softmax(logitsForText);
        let maxProb = -Infinity;
        let maxIdx = -1;

        for (let j = 0; j < numLabels; j++) {
          const prob = probs[j];
          if (prob > maxProb) {
            maxProb = prob;
            maxIdx = j;
          }
        }
        
        let label: string = this.getLabel(labels, idToClass, i, maxIdx, sameLabels);
        
        const resultForText: [string, number][] = [];
        resultForText.push([label, maxProb]);
        results.push(resultForText);
    }

    return results;
    }
  }
}