
# ⭐GLiClass.c: Generalist and Lightweight Model for Sequence Classification in C

GLiClass.c is a C - based inference engine for running GLiClass(Generalist and Lightweight Model for Sequence Classification) models. This is an efficient zero-shot classifier inspired by [GLiNER](https://github.com/urchade/GLiNER) work. It demonstrates the same performance as a cross-encoder while being more compute-efficient because classification is done at a single forward path.  

It can be used for topic classification, sentiment analysis and as a reranker in RAG pipelines.

<p align="center">
    <img src="kg.png" style="position: relative; top: 5px;">
    <a href="https://www.knowledgator.com/"> Knowledgator</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.linkedin.com/company/knowledgator/">✔️ LinkedIn</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/NNwdHEKX">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/spaces/knowledgator/GLiClass_SandBox">🤗 Space</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://huggingface.co/collections/knowledgator/gliclass-6661838823756265f2ac3848">🤗 GliClass Collection</a>
</p>

## 🌟 Key Features

- Flexible entity recognition without predefined categories
- Lightweight and fast inference
- Easy integration with web applications
- TypeScript support for better developer experience

## 🚀 Getting Started

### Installation

```bash
npm install gliclass
```

### Basic Usage

```javascript
const gliclass = new Gliclass({
    tokenizerPath: "knowledgator/gliclass-small-v1.0",
    onnxSettings: {
      modelPath: "public/model.onnx",
      executionProvider: "cpu",
      multiThread: true,
    },
    promptFirst: false,
  });

  await gliclass.initialize();

const input_text = "Your input text here";
const texts = [input_text];
const labels = ["business", "science", "tech"];
const threshold = 0.5;

const decoded = await gliclass.inference({ texts, labels, threshold });
console.log(decoded);
```

### Advanced Usage

#### ONNX settings API

- modelPath: can be either a URL to a local model as in the basic example, or it can also be the Model itself as an array of binary data.
- executionProvider: these are the same providers that ONNX web supports, currently we allow `webgpu` (recommended), `cpu`, `wasm`, `webgl` but more can be added
- wasmPaths: Path to the wasm binaries, this can be either a URL to the binaries like a CDN url, or a local path to a folder with the binaries.
- multiThread: wether to multithread at all, only relevent for wasm and cpu exeuction providers.
- multiThread: When choosing the wasm or cpu provider, multiThread will allow you to specify the number of cores you want to use.
- fetchBinary: will prefetch the binary from the default or provided wasm paths

## 🛠 Setup & Model Preparation

To use GLiNER models in a web environment, you need an ONNX format model. You can:

1. Search for pre-converted models on [HuggingFace](https://huggingface.co/onnx-community?search_models=gliclass)
2. Convert a model yourself using the [official Python script](https://github.com/Knowledgator/GLiClass.c/blob/main/ONNX_CONVERTING/convert_to_onnx.py)

### Converting to ONNX Format

Use the `convert_to_onnx.py` script with the following arguments:

- `model_path`: Location of the GLiNER model
- `save_path`: Where to save the ONNX file
- `quantize`: Set to True for IntU8 quantization (optional)

Example:

```bash
python convert_to_onnx.py --model_path /path/to/your/model --save_path /path/to/save/onnx --quantize True
```

## 🌟 Use Cases

GLiClass.js offers versatile text classification capabilities across various domains:

1. **Documents Classification**
2. **Sentiment Analysis**
3. **Reranking of Search Results**
   ...

## 🔧 Areas for Improvement

- [ ] Further optimize inference speed
- [ ] Add support for more architectures
- [ ] Enable model training capabilities
- [ ] Provide more usage examples

## Creating a PR

- for any changes, remember to run `pnpm changeset`, otherwise there will not be a version bump and the PR Github Action will fail.

## 🙏 Acknowledgements

- [GLiNER original authors](https://github.com/urchade/GLiNER)
- [ONNX Runtime Web](https://github.com/microsoft/onnxruntime)
- [Transformers.js](https://github.com/xenova/transformers.js)

## 📞 Support

For questions and support, please join our [Discord community](https://discord.gg/ApZvyNZU) or open an issue on GitHub.
