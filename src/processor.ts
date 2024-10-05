abstract class Processor {
  config: any;
  tokenizer: any;

  constructor(config: any, tokenizer: any) {
    this.config = config;
    this.tokenizer = tokenizer;
  }

  createMappings(classes: string[]): {
    classToId: Record<string, number>;
    idToClass: Record<number, string>;
  } {
    const classToId: Record<string, number> = {};
    const idToClass: Record<number, string> = {};

    classes.forEach((className, index) => {
      const id = index + 1; // Start numbering from 1
      classToId[className] = id;
      idToClass[id] = className;
    });

    return { classToId, idToClass };
  }

  abstract prepareTextInputs(texts: string[], labels: string[]): string[];

  abstract prepareBatch(texts: string[], labels: string[]): Record<string, any>;

  encodeInputs(
    texts: string[],
  ): [number[][], number[][]] {
    let inputsIds: number[][] = [];
    let attentionMasks: number[][] = [];
  
    for (let id = 0; id < texts.length; id++) {
      let textInput = texts[id];
  
      let inputIds: number[] = this.tokenizer.encode(textInput);
  
      let attentionMask: number[] = new Array(inputIds.length).fill(1);
  
      inputsIds.push(inputIds);
      attentionMasks.push(attentionMask);
    }
  
    return [inputsIds, attentionMasks];
  }

  padArray(arr: any[], dimensions: number = 2): any[] {
    if (dimensions < 2 || dimensions > 3) {
      throw new Error("Only 2D and 3D arrays are supported");
    }

    const maxLength = Math.max(...arr.map((subArr: any[]) => subArr.length));
    const finalDim = dimensions === 3 ? arr[0][0].length : 0;

    return arr.map((subArr: any[]) => {
      const padCount = maxLength - subArr.length;
      const padding = Array(padCount).fill(dimensions === 3 ? Array(finalDim).fill(0) : 0);
      return [...subArr, ...padding];
    });
  }
}

export class UniEncoderProcessor extends Processor {
  constructor(config: any, tokenizer: any) {
    super(config, tokenizer);
  }

  prepareTextInputs(texts: string[], labels: string[] | string[][]): string[] {
    const inputTexts: string[] = [];
    let sameLabels: boolean;
  
    if (typeof labels[0] === 'string') {
      sameLabels = true;
    } else {
      sameLabels = false;
    }
  
    texts.forEach((text, id) => {
      let promptTokens: string[] = [];
      let currLabels: string[];
  
      if (sameLabels === true) {
        currLabels = labels as string[];
      } else {
        currLabels = (labels as string[][])[id];
      }
  
      for (let label of currLabels) {
        promptTokens.push("<<LABEL>>");
        promptTokens.push(label);
      }
      promptTokens.push("<<SEP>>");
      let inputText: string = promptTokens.join("");
  
      if (this.config.promptFirst) {
        inputText = inputText.concat(text);
      } else {
        inputText = text.concat(inputText);
      }
      
      inputTexts.push(inputText);
    });
  
    return inputTexts;
  }

  prepareBatch(texts: string[], labels: string[]): Record<string, any> {
    const inputTexts: string[] = this.prepareTextInputs(texts, labels);
    
    const { idToClass } = this.createMappings(labels);

    let [inputsIds, attentionMasks] = this.encodeInputs(inputTexts);

    inputsIds = this.padArray(inputsIds);
    attentionMasks = this.padArray(attentionMasks);

    return {
      inputsIds: inputsIds,
      attentionMasks: attentionMasks,
      idToClass: idToClass,
    };
  }
}
