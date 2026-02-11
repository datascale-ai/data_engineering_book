## Chapter 7: Data Recaptioning

### Chapter Summary

Original Alt-text (alternative text) on the internet is essentially auxiliary content designed by web developers for Search Engine Optimization (SEO). Its core goal is to improve webpage ranking in search results, not to accurately and comprehensively describe the visual content of the image itself—this leads to a large amount of raw Alt-text that cannot meet the core requirement of "visual-text precise alignment" for Visual Language Model (VLM) training. This chapter systematically introduces how to leverage mainstream Visual Language Models (VLMs) to build an efficient, scalable "synthetic caption factory" for automatic large-scale image data recaptioning. We will explore in depth the key role of Prompt Engineering in precisely controlling description granularity (from brief to detailed), crack the differentiated description precision requirements for different downstream tasks, and introduce Optical Character Recognition (OCR) technology as a supplement to address VLM's weak recognition of text in rich-text images (such as documents, posters, charts), further enhancing model understanding of complex images.

**Learning Objectives**:
* Deeply understand the essential causes of Alt-text's "three sins" (irrelevant, too short, visual omission), and the specific harms such low-quality descriptions cause to visual language model and generative visual model training (e.g., model hallucination, visual-text alignment failure, poor generalization).
* Master the complete workflow of deploying mainstream VLMs like LLaVA and CogVLM using vLLM (efficient large model inference engine), understand core principles of high-throughput inference, and achieve rapid large-scale image recaptioning.
* Be able to design layered Prompt strategies based on different downstream tasks (e.g., CLIP-style dual-tower model pre-training, Sora-style generative model training), flexibly generating brief or detailed image descriptions to precisely match task requirements.
* Master core OCR application methods, implement dynamic fusion of OCR recognition results with VLM Prompts, solve low description quality for document and poster-class rich-text images, and significantly improve description accuracy and richness for such images.

**Scenario Introduction**:
> "Imagine you are training a Sora-like model. You feed the model an image of a golden retriever running in the sunset with the Eiffel Tower in the background. Yet the raw data label is 'IMG_20240501.jpg' or 'Best dog food 50% off.' With such data, the model will never learn the visual correspondence of 'golden retriever' and 'Eiffel Tower,' let alone understand 'sunset lighting.' We need 'data recaptioning'—letting AI act as annotator—to accurately write the dog and tower into the text."

### 7.1 Limitations of Alt-text: Why Are Raw Web Descriptions Unusable?

In the training of visual language models and generative visual models, data quality directly determines model ceiling—and raw Alt-text from web pages is precisely one of the main sources of low-quality visual-text data. According to internal research reports from DeepMind ("Scaling Language-Image Pre-training with Weakly Supervised Image-Text Data") and OpenAI ("Training language models to follow instructions with human feedback"), directly using raw Alt-text crawled from the internet as training data causes model performance to "cap" prematurely (i.e., regardless of data volume increase, model visual understanding and text generation precision cannot improve further), or even degrade. The core problems can be summarized as "three sins":

* **Extremely Noisy**: Large amounts of Alt-text contain only filenames (e.g., "IMG_20240501.jpg"), dates, irrelevant SEO keyword stacking (e.g., "buy cheap shoes nike adidas" "best coffee shop near me"). Such descriptions are completely unrelated to image visual content. Using them for training not only cannot help the model establish visual-text correspondence, but also pollutes model language ability, causing the model to generate irrelevant, redundant text, or even severe hallucinations.
* **Visual Omission**: Alt-text design intent is mostly to support webpage functionality, not to describe visual content—it often only describes image function (e.g., "click purchase button," "view more details") or commercial attributes (e.g., "red XL size," "limited-time discount"), while completely ignoring image visual details (e.g., object shape, color, texture, spatial relations, lighting effects). For example, an image showing "a red pure cotton T-shirt with white vintage logo on chest, laid flat on wooden table" may have Alt-text of merely "red T-shirt promotion"—such description cannot let the model learn any visual features.
* **Too Short**: According to Common Crawl (world's largest web crawl dataset) statistics, over 50% of Alt-text is less than 5 words, 30% even less than 3 words. Such extremely short descriptions cannot carry complex visual logic, spatial relations, and detail information—e.g., cannot describe "a golden retriever lying on grass, front paws on a red ball, background of hillside full of wildflowers" involving multiple objects, scenes, and interaction relations.

**Value of Recaptioning**: The core value of data recaptioning is to use AI to automatically generate high-quality "visual-text precise alignment" descriptions to replace low-quality raw Alt-text, breaking model performance ceiling. This has been confirmed by top industry research—OpenAI explicitly stated in the DALL-E 3 paper ("DALL·E 3: Scaling Autoregressive Image Generation with Improved Alignment") that using up to 95% synthetic long-form text (Synthetic Captions, i.e., recaption text generated by VLM) for training is one of the core reasons its instruction-following ability and visual restoration precision far surpass Stable Diffusion XL (SDXL). Synthetic long-form text can precisely capture image visual details, logical relations, and scene atmosphere, letting the model truly learn "describe what you see," thereby improving subsequent generation, recognition, and understanding capabilities.

### 7.2 Synthetic Caption Factory: Using VLM to Rebirth Data

To achieve large-scale image recaptioning, relying solely on manual annotation is not only extremely costly (minutes per image, datasets often millions or billions of images) but also suffers from inconsistent annotation standards and low efficiency. Therefore, we need to build a VLM-driven "synthetic caption factory"—taking raw images as input and high-quality, standardized text descriptions as output, completing data recaptioning through automation and batch processing to achieve data value "rebirth."

The core logic of this "factory" is: feed raw images into optimized VLM, control description granularity and style through carefully designed Prompts, then improve processing throughput through efficient inference engine, finally outputting high-quality descriptions that meet downstream task requirements. The entire flow can be divided into three core links: "Model Selection and Architecture Design," "Prompt Strategy Optimization," and "Engineering Deployment."

#### 7.2.1 Model Selection and Architecture

VLM architecture directly determines description quality, speed, and applicable scenarios. Current mainstream VLM architectures are mainly divided into three types. The table below shows representative models, advantages/disadvantages comparison, and recommended scenarios for each architecture. Selection can be flexible based on downstream task requirements (e.g., description precision, processing speed, data type):

| Model Architecture | Representative Models | Advantages | Disadvantages | Recommended Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| **Q-Former Connection** | BLIP-2, InstructBLIP | Small parameter count (typically billions, far below large language models), fast inference (single image inference can be tens of milliseconds), low training and deployment cost, less prone to text hallucination (description closely fits image) | Short description length, average detail capture, prone to "repetitive description" (repeating few core objects, lacking detail extension), limited understanding of complex scenes | Quick initial screening of massive images (e.g., rough recaption of billions of images to filter valuable data), or generating short Alt-text replacements (for scenarios with strict description length limits) |
| **MLP Projection + LLM** | LLaVA-1_6 / NeXT | Extremely detailed descriptions, captures subtle image details (e.g., lighting, texture, object interactions), strong instruction following (precisely responds to Prompt requirements like "describe in scene order," "highlight core objects"), supports multi-turn dialogue (can optimize description quality through multi-turn Prompts) | Heavy logic computation (requires 7B+ parameter LLMs like LLaMA 2 7B/13B), relatively slow inference, without Prompt constraints prone to verbose, redundant descriptions | Main model, for generating high-quality, long-form Dense Caption (e.g., training Sora-style generative models, SD3 image generation models, scenarios requiring precise, detailed visual-text alignment data) |
| **Vision-First Architecture** | CogVLM, Qwen-VL | High visual resolution (supports HD image input, some models support 4K), excels at fine-grained object recognition, especially high precision for text and small widgets (buttons, input fields) in rich-text images (documents, charts, UI screenshots), understands text-visual element associations | Higher VRAM usage (7B model deployment requires at least 24GB VRAM), non-standard architecture (different models have varying deployment approaches), slightly cumbersome deployment, medium inference speed | Specifically for documents, charts, UI screenshots, posters and other rich-text data (e.g., training models that generate document images, UI interfaces, or scenarios requiring precise text recognition in images) |

Supplementary note: Core difference among the three architectures lies in "how visual module connects to language module": Q-Former architecture uses dedicated Q-Former module to convert visual features to language-understandable vectors before input to lightweight language model; MLP projection architecture uses multi-layer perceptron (MLP) to project visual features to language model embedding space, deeply integrating with large language model; vision-first architecture strengthens visual module resolution and recognition capability, weakens language module redundant computation, prioritizing "vision understanding first."

#### 7.2.2 Prompt Strategy: Controlling Granularity

Prompt Engineering is the "core controller" of the "synthetic caption factory"—the same VLM, under different Prompt guidance, generates completely different data distributions (description length, detail richness, style). Therefore, we need to design layered Prompt strategies based on specific downstream task requirements, precisely controlling description granularity so generated descriptions perfectly match task needs.

Core Principle: Prompt design must clearly specify "task instruction," "description scope," and "granularity requirements," avoiding vague expressions (e.g., only using "describe this image" leads to unstable model output). Meanwhile, adding "constraints" (e.g., "no more than 20 words," "highlight core objects and background") can further optimize output quality.




![Figure 7-1: Brief vs Detailed Prompt Strategy](../../images/part3/图7_1_简略与详细的Prompt策略.png)
*Figure 7-1: Brief vs Detailed Prompt Strategy*


Figure 7-1 intuitively compares output differences of two core Prompt strategies—brief Prompt generates concise descriptions with only core objects and scenes; detailed Prompt generates descriptions with rich details including object form, lighting, color, spatial relations, etc. The two strategies adapt to different downstream tasks respectively.

Below are two most commonly used layered Prompt strategies. Adjust flexibly based on actual needs, or design medium-granularity Prompt strategies on this basis:

**Strategy One: Brief Description (Brief Caption)**
* **Prompt**: "Describe this image concisely in one sentence."
  Supplementary optimization Prompt (for stability): "Describe this image concisely in one sentence, focusing only on the main subject and key background, no redundant details."
* **Purpose**: Adapt to CLIP-style dual-tower models' (vision-text dual-tower architecture) Context Length limit—such models typically limit text input to 77 tokens or less; overly long descriptions get truncated, preventing normal learning. Also suitable for scenarios with strict description length requirements (e.g., image retrieval, quick annotation).
* **Expected Output**: "A golden retriever running on grass near the Eiffel Tower."
  Output characteristics: Length controlled to 10-20 words, only core objects (golden retriever), key action (running), core background (Eiffel Tower, grass), no extra details, concise and clear.

**Strategy Two: Detailed Description (Detailed Caption)**
* **Prompt**: "Describe this image in extreme detail. Start with the main subject, then describe the background, lighting, colors, and artistic style. Mention any specific interactions between objects."
  Supplementary optimization Prompt (for better detail capture): "Describe this image in extreme detail. First, describe the main subject's appearance (shape, color, texture), then the background scene, lighting effects (brightness, color temperature), color matching, and artistic style. Finally, mention the interactions between objects and the overall atmosphere of the image."
* **Purpose**: Adapt to GenAI model training (e.g., Sora, SD3, Ideogram)—such models need detailed descriptions to learn image detail features, logical relations, and scene atmosphere to generate high-precision, instruction-compliant images. Also suitable for scenarios requiring precise visual-text alignment (e.g., visual QA, image editing).
* **Expected Output**: "A dynamic wide-angle shot of a fluffy golden retriever running joyfully across a green lawn. The dog's fur is illuminated by the warm, golden light of a setting sun, with some light brown strands glinting in the sunlight. Its ears flop backward as it runs, and its tail is raised high, showing a happy mood. In the blurred background, the iconic iron lattice structure of the Eiffel Tower rises against a gradient sky of purple and orange, with a few wispy clouds floating nearby. The lawn is dotted with small white clover flowers, and the overall atmosphere of the image is warm and lively, with soft focus on the dog and a blurred background that highlights the main subject."
  Output characteristics: Length typically 50-200 words, covering core object details, background scene, lighting, color, artistic style, object interactions, and overall atmosphere—detail-rich, high visual-text alignment precision.

Supplementary tip: Besides the above two strategies, "task-oriented Prompts" can be designed, e.g., for e-commerce images ("Describe this product image in detail, focusing on the product's appearance, color, size, texture, and placement, suitable for e-commerce promotion"), for document images ("Describe this document image in detail, including the text content, layout, font style, and color of the text"), to further improve description relevance.

#### 7.2.3 Engineering Implementation: Building High-Throughput Inference Service with vLLM

For large-scale data recaptioning (e.g., processing billion-scale image datasets), ordinary HuggingFace `generate()` is far insufficient—slow inference, low throughput, unable to efficiently utilize GPU resources. A single GPU can only process thousands of images per day; large-scale processing would consume significant time and hardware cost. Therefore, we need dedicated large model inference engine—vLLM, which supports PagedAttention and Continuous Batching, two core optimization techniques that can improve VLM inference throughput 3-5x while reducing GPU VRAM usage, achieving efficient, large-scale image recaptioning.

vLLM is a high-efficiency large model inference engine developed by UC Berkeley research team. Core advantages are "high throughput, low latency, high GPU utilization," perfectly suited for deploying LLaVA, CogVLM and other mainstream VLMs, with API interface compatible with HuggingFace and minimal migration cost.

Below is the core code for deploying LLaVA-1_5-7b-hf with vLLM for high-throughput image recaptioning, including model initialization, Prompt template design, batch processing, and output extraction—complete workflow with key parameter interpretation and optimization tips:

```python
from vllm import LLM, SamplingParams
from PIL import Image
import os
from tqdm import tqdm  # For displaying batch processing progress

# Initialize vLLM inference engine
# tensor_parallel_size=4: Use 4 GPUs for tensor parallelism, for large models (7B/13B) deployment, adjust based on GPU count (1, 2, 4, 8)
# Note: Tensor parallelism requires multiple same-model GPUs with NVLink support for faster data transfer
# trust_remote_code=True: Allow loading LLaVA custom code (e.g., vision-language fusion module), as LLaVA architecture is non-standard HuggingFace
# model: Model name from HuggingFace Hub (e.g., llava-hf/llava-1_5-7b-hf, llava-hf/llava-1_5-13b-hf)
# gpu_memory_utilization=0_9: Set GPU VRAM utilization to 90%, balance throughput and stability, avoid OOM
llm = LLM(
    model="llava-hf/llava-1_5-7b-hf",
    tensor_parallel_size=4,
    trust_remote_code=True,
    gpu_memory_utilization=0_9
)

# Define Prompt template (LLaVA requires specific dialogue format, otherwise affects instruction following)
# Tip: Adding "Analyze the image" in Prompt often works better than "Describe the image"
# because "Analyze" guides model to observe image details more carefully, reducing perfunctory descriptions
# Here using detailed description template, can replace with brief description template as needed
prompt_template = "USER: <image>\nAnalyze this image and describe it in extreme detail. Start with the main subject, then describe the background, lighting, colors, and artistic style. Mention any specific interactions between objects. ASSISTANT:"

# Configure sampling parameters for description quality and stability
# temperature=0_2: Lower randomness (0-1 range), lower temperature = more stable, image-fitting descriptions, fewer hallucinations
# For more diverse descriptions, adjust to 0_5-0_7; if too high (>0_8), likely to produce image-unrelated hallucinations
# max_tokens=256: Limit output length, prevent overly verbose descriptions
# top_p=0_95: Nucleus sampling, keep only tokens with cumulative probability to 95%, further reduce hallucination risk
sampling_params = SamplingParams(
    temperature=0_2,
    max_tokens=256,
    top_p=0_95
)

def load_image_batch(image_dir, batch_size=32):
    """
    Batch load images for efficient processing
    image_dir: Image folder path, all images in this folder
    batch_size: Images per batch, adjust based on GPU VRAM (16, 32, 64), larger VRAM = larger batch_size
    return: Batch image list (PIL.Image format) and corresponding image path list
    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
    image_batches = []
    path_batches = []
    
    # Load images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            try:
                # Load image and convert to RGB (avoid model errors from grayscale/transparent images)
                img = Image.open(path).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
                continue
        if batch_images:  # Skip empty batches
            image_batches.append(batch_images)
            path_batches.append(batch_paths)
    return image_batches, path_batches

def process_batch(image_batch):
    """
    Process a batch of images, generate corresponding recaption text
    image_batch: List[PIL.Image], batch image list
    return: List[str], recaption text list for each image
    """
    # Generate corresponding Prompt for each image
    prompts = [prompt_template for _ in range(len(image_batch))]
    
    # vLLM supports direct multi_modal_data input, no manual image format conversion needed
    # This step is non-blocking; vLLM internally does Continuous Batching for efficient GPU utilization
    # When one batch completes partially, immediately load next batch partial data to avoid GPU idle
    outputs = llm.generate(
        prompts, 
        sampling_params, 
        multi_modal_data={"image": image_batch}
    )
    
    # Extract generated description text, remove Prompt part, keep only model response
    captions = []
    for output in outputs:
        # Extract content after ASSISTANT: as model-generated description
        caption = output.outputs[0].text.strip().replace("ASSISTANT:", "").strip()
        captions.append(caption)
    return captions

def save_captions(image_paths, captions, save_path):
    """
    Save recaption text corresponding to image paths for subsequent use (e.g., model training)
    image_paths: Image path list
    captions: Recaption text list
    save_path: Save file path (txt format)
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for path, cap in zip(image_paths, captions):
            # Format: image_path\trecaption_text, for easy reading and parsing
            f.write(f"{path}\t{cap}\n")

# Main function: batch process images and generate recaptions
if __name__ == "__main__":
    image_dir = "path/to/your/image/directory"  # Replace with your image folder path
    save_path = "recaption_results.txt"        # Recaption results save path
    batch_size = 32                            # Images per batch, adjust based on GPU VRAM
    
    # Load image batches
    image_batches, path_batches = load_image_batch(image_dir, batch_size)
    
    # Batch process and save results
    with open(save_path, 'w', encoding='utf-8') as f:
        for img_batch, path_batch in tqdm(zip(image_batches, path_batches), total=len(image_batches)):
            captions = process_batch(img_batch)
            # Write current batch results
            for path, cap in zip(path_batch, captions):
                f.write(f"{path}\t{cap}\n")
    print(f"Recaptioning complete, results saved to {save_path}")

```
Supplementary engineering optimization tips:

* **Image Preprocessing**: When batch loading, uniformly resize images (e.g., to 224×224 or 448×448) to avoid model inference speed fluctuation and VRAM instability from image size differences; normalize images to improve description precision.
* **Error Handling**: Add exception handling for image load failure and model inference failure to avoid batch processing interruption; for failed images, record path and process separately.
* **Hardware Optimization**: Prefer NVIDIA A100, A800 for deployment; VRAM at least 24GB (7B model); for very large scale, use GPU cluster and vLLM distributed inference for higher throughput.
* **Prompt Caching**: For same-type images (e.g., batch e-commerce posters), cache Prompt template to avoid repeated generation and improve processing speed.

### 7.3 OCR Enhancement: Extract and Fuse Text from Images

While ordinary VLMs have certain visual understanding and can recognize objects, scenes, and simple text in images, they face two core problems with dense-text images (documents, posters, charts, PDF screenshots): first, low text recognition precision, prone to misrecognition and omission (especially artistic fonts, blurry text); second, unable to effectively associate text with visual elements, causing descriptions to omit text meaning and role.

For example, an e-commerce poster with large text "Summer Sale 50% Off" may get only "A red promotional poster" from ordinary VLM—completely ignoring text. Even when text is recognized, errors like "Summer Sale 30% Off" may occur. Yet text is crucial for recaptioning such images—it directly determines core meaning and purpose.

Best practice is to introduce dedicated OCR engine (e.g., PaddleOCR, Tesseract) as VLM's "external brain." Use OCR to precisely extract image text, then dynamically fuse with VLM Prompt, letting VLM combine text to generate more accurate, richer descriptions—significantly improving recaption quality for document and poster-class rich-text images.

OCR (Optical Character Recognition) technology's core is converting printed and handwritten text in images to editable text. Its text recognition precision far exceeds ordinary VLM, especially for dense text and complex font scenarios. Currently, the most widely used industrial, open-source, free, and high-precision OCR engine is PaddleOCR (Baidu PaddlePaddle open-source OCR). It supports multilingual, multi-font, blurry text recognition, fast inference, simple deployment, GPU acceleration—very suitable for combination with VLM.




![Figure 7-2: OCR Enhancement Pipeline](../../images/part3/图7_2_OCR增强流水线.png)
*Figure 7-2: OCR Enhancement Pipeline*


**Chart Core Interpretation**: Figure 7-2 shows the complete OCR-enhanced VLM recaption flow. Core is "OCR extract text → context construction → Prompt fusion → VLM generate description," supplementing VLM's text recognition weakness with OCR to achieve "visual details + text information" dual-precise description.

#### 7.3.1 OCR Enhancement Pipeline

OCR enhancement core is organically fusing OCR-extracted text with VLM Prompt, not simple concatenation. The entire pipeline has three core steps, each with clear optimization direction to ensure text effectively improves recaption quality:

1.  **Detection and Recognition**: Use PaddleOCR to process raw image—first detect all text regions (Bounding Box for text position), then recognize each region, output recognized text and confidence (0-1, higher = more accurate). Core goal: "precise text extraction, filter wrong recognition"—filter low-confidence results to avoid misleading VLM.
2.  **Context Construction**: Concatenate all valid OCR text (after filtering low confidence) by actual image position (top-to-bottom, left-to-right, multi-column by column order) to build human-readable text context. Optionally classify text (e.g., title, body, button text) for VLM to understand hierarchy and role. E.g., poster text "Summer Sale" (title), "50% Off" (subtitle), "June 1 - June 10" (time) concatenates to "Summer Sale, 50% Off, June 1 - June 10" with title/body labels.
3.  **Prompt Fusion**: Naturally integrate constructed text context into VLM Prompt, explicitly telling VLM "the image contains these texts, please describe combining text and visual elements," guiding VLM to associate text with visuals (position, color, font style, text meaning vs. scene). Key is "natural fusion, no redundancy"—avoid awkwardly appending text to Prompt end, causing VLM to ignore visual details.

**Supplementary Note**: Pipeline optimization focuses on "confidence filtering" and "Prompt fusion"—without filtering low-confidence text, wrong text misleads VLM; if Prompt fusion is awkward, VLM separates text from visuals, failing to achieve true enhancement.

#### 7.3.2 Core Code: OCR Result Injection

Below is the core code for using PaddleOCR to extract image text and dynamically fuse into VLM Prompt. It can seamlessly integrate with Section 7.2.3 vLLM batch processing for OCR-enhanced large-scale image recaptioning. The code includes complete logic for text extraction, confidence filtering, context construction, Prompt fusion, plus key parameter interpretation and optimization tips:

```python
from paddleocr import PaddleOCR
import os
from PIL import Image

# Initialize OCR engine (recommend GPU for speed; set use_gpu=False if no GPU)
# use_angle_cls=True: Enable text direction detection, supports tilted text (e.g., tilted poster, rotated documents) to avoid recognition errors
# lang='en': English recognition; for Chinese set lang='ch'; supports Chinese-English mixed (lang='ch_en')
# det_model_dir, rec_model_dir: Can specify OCR detection/recognition model paths; auto-download if not specified
# gpu_mem=500: GPU VRAM limit (MB), adjust based on GPU
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    gpu_mem=500)

def generate_ocr_enhanced_prompt(image_path, base_prompt="Describe this image in detail."):
    """
    Generate OCR-enhanced VLM Prompt, integrating OCR-extracted text into Prompt
    image_path: Raw image path
    base_prompt: Base Prompt (e.g., brief/detailed description) as Prompt body
    return: Complete OCR-enhanced Prompt; if no valid text, return base Prompt
    """
    # Step 1: Run OCR, extract image text and confidence
    # result is nested list: [[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]], ...]
    # [x1,y1]~[x4,y4] are Bounding Box coordinates (top-left, top-right, bottom-right, bottom-left)
    # text is recognized content, confidence is recognition confidence
    result = ocr.ocr(image_path, cls=True)
    
    # Handle OCR result: if no text or empty, fall back to base Prompt
    if not result or not result[0]:
        return f"USER: <image>\n{base_prompt}\nASSISTANT:"
    
    # Step 2: Extract valid text (filter low confidence), build text context
    detected_texts = []
    for line in result[0]:
        text = line[1][0]  # Recognized text
        confidence = line[1][1]  # Recognition confidence
        # Filter results below 0_8 confidence (threshold adjustable, 0_7-0_9 based on image text clarity)
        # Also filter empty text and meaningless garbage (e.g., symbols only, spaces)
        if confidence > 0_8 and text.strip() and len(text.strip()) > 1:
            detected_texts.append(text.strip())
    
    # Build text context: concatenate by recognition order, comma-separated, human-readable
    ocr_context = ", ".join(detected_texts)
    
    # Step 3: Dynamically fuse OCR result with base Prompt, generate enhanced Prompt
    # Key technique: Tell model "I have detected these texts..." so model knows this is image text
    # and guide model to associate text with visuals (position, color, font, text meaning vs. scene)
    if len(ocr_context) > 10:  # Only enhance when text long enough (>10 chars) to avoid redundancy
        enhanced_prompt = (
            f"USER: <image>\n"
            f"I have detected these text segments in the image: '{ocr_context}'. "
            f"Using this text as a reference, describe the image in detail, "
            f"paying attention to how the text relates to the visual elements (such as the position, color, and font style of the text, "
            f"and the connection between the text content and the image scene). {base_prompt}\n"
            f"ASSISTANT:"
        )
        return enhanced_prompt
    else:
        # If text too short (1-2 words), don't enhance, avoid redundancy, fall back to base Prompt
        return f"USER: <image>\n{base_prompt}\nASSISTANT:"

# Test code: verify OCR-enhanced Prompt generation
if __name__ == "__main__":
    # Test image path (replace with your rich-text image, e.g., poster, document screenshot)
    test_image_path = "path/to/your/test/poster.jpg"
    # Base Prompt (detailed description template)
    base_prompt = "Describe this image in extreme detail. Start with the main subject, then describe the background, lighting, colors, and artistic style."
    # Generate enhanced Prompt
    enhanced_prompt = generate_ocr_enhanced_prompt(test_image_path, base_prompt)
    print("OCR-enhanced Prompt:")
    print(enhanced_prompt)
```

**Supplementary Optimization Tips:**

* **Confidence Threshold Adjustment**: For clear-text images (HD documents, formal posters), adjust to 0_8-0_9 to filter few errors; for blurry, complex-font images (old posters, handwriting), adjust to 0_7-0_8 to avoid missing valid text.
* **Text Context Optimization**: For multi-column, hierarchical text (title, body, footnote), use Bounding Box coordinates to classify and concatenate, e.g., "Title: Summer Sale; Body: 50% Off, June 1 - June 10; Footnote: Final interpretation right reserved," for clearer VLM text hierarchy understanding.
* **Prompt Fusion Optimization**: Adjust fusion wording by image type—document images add "Describe the layout of the text and the relationship between the text and the document structure," poster images add "Describe the font style of the text and the role of the text in the promotional scene" for more targeted descriptions.
* **Multi-OCR Engine Fusion**: For extremely high precision requirements, use both PaddleOCR and Tesseract, take intersection of results for higher text recognition precision.

**Practical Benefits**: OCR enhancement significantly improves recaption quality for rich-text images. Typical comparison:

* **Ordinary VLM (no OCR) on e-commerce poster**: "A red promotional poster with a white background, featuring some vague text and a button at the bottom."
* **OCR-enhanced VLM**: "A promotional red poster with a white background, featuring the text 'SUMMER SALE 50% OFF' in large white bold letters at the top center of the poster, and 'Shop Now' in a small blue button at the bottom right. The text 'SUMMER SALE' is in a decorative font, with a yellow shadow effect that makes it stand out. The overall layout is simple and eye-catching, focusing on highlighting the promotional information. The background is plain white, which makes the red poster and white text more prominent."

This difference is crucial for training models that generate accurate text (Ideogram, SD3, document generation models)—recaptions with precise text let models learn "visual presentation of text" and "text-scene association," generating more compliant images. For visual QA and image retrieval, OCR-enhanced descriptions also improve task precision and model understanding of image core meaning.
