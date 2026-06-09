# Chapter 42: VoiceStyleControl — Data Engineering for Controllable Voice Interaction

## Abstract

This chapter uses VoiceStyleControl as a case study to discuss the engineering organization of controllable voice interaction data. Unlike ordinary text conversation samples, voice interaction data must answer not only "what to say" but also "who says it, with what emotion, and whether the generated audio is both listenable and controllable." The chapter first explains why explicit style control changes sample objectives and acceptance criteria, then maps out the scale, field structure, and complementary relationship of the two subsets — S2SEmoControl and TTSSpeakerControl. The central discussion covers schema design for the semantic channel, style channel, and acoustic supervision channel, followed by an analysis of the construction pipeline, quality assessment, evaluation protocol, and privacy-authorization risks. Through this case study, the chapter provides a systematic account of how voice data is composed of text, acoustic conditions, emotion labels, and speech tokens to form auditable training assets.

**Keywords**: controllable voice interaction; VoiceStyleControl; TTS; S2S; emotion control; voice data compliance

**Learning Objectives**

- Understand the differences between controllable voice interaction data and ordinary ASR/TTS data.
- Distinguish the field responsibilities of the semantic channel, style channel, and acoustic supervision channel.
- Understand the complementary training value of S2SEmoControl and TTSSpeakerControl.
- Design acceptance rules covering text consistency, audio quality, style control, and emotion perceptibility.
- Identify risks related to voice identity, authorization, emotional misuse, and privacy protection.

When a conversation moves from text to speech, the supervision objective of a sample undergoes a fundamental shift. Text conversation data primarily answers "what the user wants and what the assistant should say"; voice interaction data must additionally answer "who delivers the utterance, with what emotion, and whether the resulting audio is both listenable and controllable." The same sentence — "Run, it's not safe here" — carries an entirely different training signal depending on whether it is delivered in a calm broadcast tone or in a voice trembling with sweaty palms, even though the text is identical.

This is where VoiceStyleControl adds value. It does not simply concatenate ordinary ASR transcripts, TTS readings, and conversation text; instead, it unifies the user's spoken request, the assistant's response text, the target acoustic condition, the emotional style, and the corresponding speech supervision into a single auditable record. The text fields specify "what this conversational turn is about," the style fields specify "which voice and emotion should express it," and the audio files together with the discrete speech tokens provide learnable, verifiable acoustic targets for the generation stage. What the model learns from such records is not merely "response content" but how to generate emotionally expressive speech given a semantic-plus-style condition — semantic response, acoustic condition, emotion control, and speech-generation supervision all enter a single sample simultaneously.

The engineering entry point corresponding to this sample organization is the public code repository [Chanfungjan/VoiceStyleControl](https://github.com/Chanfungjan/VoiceStyleControl). The S2SEmoControl and TTSSpeakerControl subsets discussed throughout this chapter both revolve around this type of structured record: the former binds the spoken query, assistant answer, and speech supervision on both sides into one unit; the latter binds the style description, answer text, and target speech into one unit.

As a case study in controllable voice interaction data engineering, VoiceStyleControl builds on Chapter 10's video and audio data engineering — sampling rate, audio segmentation, ASR, speaker verification, and acoustic quality remain the foundation — and also connects to Chapter 20's multi-turn interaction, Chapter 23's online feedback, and Chapters 36–37's privacy compliance. It shares a common pattern with Chapter 48's multimodal generative data engineering: decomposing generation targets into content conditions and style conditions, fixing them with a structured schema, and ultimately feeding them into an end-to-end data flywheel.

The engineering focus for this type of data is not the TTS model architecture or voice cloning algorithms themselves, but rather how control conditions are recorded, how they enter training, and how balance is struck among audio quality, conversational naturalness, and compliance boundaries. Only when these questions are stably addressed by data structures and workflows can VoiceStyleControl evolve from "a collection of pleasant-sounding synthetic samples" into "a reusable controllable voice interaction data asset."

## Keywords

Domain-specific dataset; evaluation benchmark; annotation pipeline; quality control; data engineering practice

## 42.0 Learning Objectives

Upon completing this chapter, readers should be able to:

- Explain why voice interaction data must explicitly record acoustic conditions, emotions, and discrete speech tokens beyond the semantic layer, rather than reusing the supervision objectives of pure text conversation.
- Distinguish the field responsibilities of the semantic channel, style channel, and acoustic supervision channel, and understand the principle of separating input-side user state from output-side assistant target.
- Understand the complementary relationship between S2SEmoControl and TTSSpeakerControl in terms of scale, field structure, and training value.
- Design multi-dimensional sample acceptance rules covering text consistency, audio usability, acoustic condition consistency, emotion perceptibility, and authorization traceability.
- Identify risks related to voice identity, authorization, emotional misuse, and privacy protection, and govern them within the data pipeline.

## 42.1 Why Voice Conversation Requires Explicit Style Control

Ordinary text conversation samples typically consist of role, context, user request, and assistant response. As long as role boundaries, text length, safety labels, and training masks are clear, the model can learn the input–output mapping on text tokens. Speech samples introduce an additional layer of acoustic state that text cannot replace: sampling rate, duration, silence, loudness, noise, speaker identity, prosody, emotion, and discrete speech tokens all influence training outcomes. Having only the response text can explain "what was said" but not "how it should be said."

The difference between controllable voice interaction data and ordinary ASR/TTS corpora therefore lies first not in having more fields, but in a changed problem definition. ASR asks "which text corresponds to this audio segment"; ordinary TTS asks "can this text be read out naturally"; controllable voice interaction further asks "with which voice, which emotion, and at what intensity should this response enter the conversation." If these conditions are not explicitly expressed, the model can only treat acoustic variation as random noise in the training audio and will struggle to reliably respond at inference time to control conditions such as "say it with a particular emotion" or "say it in a particular voice."

First, voice conversation requires separating "content" from "expression." What the user said and what the assistant should answer constitute the semantic layer; which voice delivers the utterance, at what speech rate, energy level, and pause pattern, and whether the emotion is pronounced, constitute the expression layer. Text conversation data typically needs only to organize the semantic layer; voice generation data must make the expression layer part of the training supervision as well. Otherwise, the differences between the same response delivered in neutral, happy, fearful, or angry states will be flattened by the data pipeline.

Second, voice conversation must distinguish "understanding the user's voice" from "generating the assistant's voice." In real systems, users may be anxious, angry, or hesitant, or may speak with a heavy accent against noisy backgrounds; the assistant, however, typically needs to maintain stable acoustic conditions and an emotion strategy defined by the product specification. A customer-service assistant should not automatically become angry when the user is angry; a companion assistant should not change its timbre without reason mid-conversation. The significance of explicit style control is precisely that it separates input-side state from output-side target at the sample level, rather than assuming the two are identical.

Third, voice conversation requires translating emotion from "textual description" into "acoustic expression." Happy, angry, fearful, neutral, and sad are not just labels — they manifest in pitch, energy, speech rate, pauses, and prosody. For the model, the true learning target is not memorizing an emotion word but generating speech consistent with a given target expressive state. For this reason, controllable voice data must simultaneously preserve text content, target style specification, and corresponding speech supervision, so that emotion control can enter the generation process.

Fourth, voice conversation requires verifiable acoustic supervision. Text can enter training directly as a token sequence; speech must undergo a series of engineering steps involving audio files, sampling rate, duration, loudness, silence, and discrete speech tokens. Explicit style control cannot simply append "say it happily" as a note; it must also provide an actual audio clip as the target, so the model knows how that style condition should manifest acoustically.

From a product-experience perspective, these boundaries are critical. A companion assistant can be designed to be warm, stable, and low-key; an audiobook character can be designed to be more emotionally expressive with a stronger persona; a customer-service assistant typically needs to remain neutral and clear even when the user is angry. All three may use the same underlying semantic response capability, yet they differ in their requirements for voice identity, emotional intensity, and risk boundaries. If training samples do not explicitly distinguish these conditions, the model can only treat voice style as random noise in the audio, making stable control at inference time difficult.

From a data engineering perspective, explicit style control also changes sample acceptance criteria. A text sample generally enters the candidate pool as long as the user's question and the assistant's answer match; a voice sample must simultaneously satisfy text consistency, audio usability, target acoustic condition consistency, emotion perceptibility, and authorization traceability. Failure on any single dimension affects training: correct text with a wrong acoustic condition weakens condition control; correct acoustic condition with wrong emotion weakens emotion control; perceptible emotion with dangerous content converts risky behavior into output with greater persuasive impact.

## 42.2 Dataset Overview: Two Complementary Subsets — S2S and TTS

VoiceStyleControl is composed of two task types: speech-to-speech dialogue generation and controllable speech generation conditioned on text. Both serve the same goal — enabling the model to generate emotionally expressive speech based on semantic content, acoustic conditions, and emotional style — but they provide supervision from different perspectives.

VoiceStyleControl contains 189,375 samples in total. Of these, S2SEmoControl contains 54,586 samples (approximately 28.8% of the total), targeting style-controllable speech-to-speech dialogue generation; TTSSpeakerControl contains 134,789 samples (approximately 71.2% of the total), targeting controllable text-to-speech generation. The former is closer to a real voice assistant scenario, where the model must understand the user's spoken request and generate a spoken assistant response; the latter focuses more directly on training the model to generate target speech from a style text, acoustic condition, and emotional style.

**Table 42-1: VoiceStyleControl Sample Scale and Emotion Distribution**

| Emotion | S2SEmoControl | TTSSpeakerControl | Total | Total ratio |
|---|---:|---:|---:|---:|
| happy | 10,937 | 38,500 | 49,437 | 26.1% |
| angry | 11,022 | 38,054 | 49,076 | 25.9% |
| fearful | 10,799 | 24,925 | 35,724 | 18.9% |
| neutral | 10,797 | 0 | 10,797 | 5.7% |
| sad | 11,031 | 33,310 | 44,341 | 23.4% |
| **Total** | **54,586** | **134,789** | **189,375** | **100.0%** |

Table 42-1 shows that the five emotion classes in S2SEmoControl are nearly balanced, each ranging from approximately 10.8k to 11.0k samples; TTSSpeakerControl covers four expressive emotions — happy, angry, fearful, and sad — and does not explicitly include neutral. This design is not accidental. S2S dialogue needs neutral as a stable baseline; without it, the model tends to learn all responses as high-intensity emotional expressions. The TTS controllable generation subset, which has more samples, concentrates its capacity on expressions such as "say it happily," "say it angrily," "say it a bit fearfully," and "say it sadly" — cases that require greater acoustic variation.

In terms of record composition, neither subset is a simple combination of "text + audio." Each sample contains at least five categories of information: task source and task type, text-side content, acoustic and emotion conditions, speech generation supervision, and basic audio configuration. Together, these determine whether a voice sample can be used to train conditioned, emotionally expressive speech generation: task information determines the loading procedure, text content provides the semantic target, acoustic and emotion conditions specify the generation style, speech supervision provides learnable acoustic targets, and basic audio configuration ensures that training and evaluation can be reproduced.

The two subsets respectively serve as "capability foundation" and "interaction deployment." TTSSpeakerControl, with its larger sample count, directly teaches the model to map natural-language style descriptions, acoustic conditions, and emotional styles to target speech; S2SEmoControl, though smaller, more closely resembles a real voice assistant — the model must first understand the user-side speech and then generate a spoken assistant response. When used jointly, the TTS subset provides stable style-generation supervision, while the S2S subset places this capability back in a conversational context, training the model on the transformation between user acoustic state and assistant generation target.

VoiceStyleControl should therefore not be understood simply as a TTS dataset. The core supervision objective of an ordinary TTS corpus is "given text, read the text"; VoiceStyleControl's core supervision objective is "given semantic content and style conditions, generate speech appropriate to the conversational goal." The former primarily concerns pronunciation, naturalness, and audio quality; the latter also concerns user state, assistant acoustic conditions, emotion selection, cross-turn consistency, and safety boundaries. Once the data objective differs, schema design, balancing, splitting, and evaluation all change accordingly.

## 42.3 Sample Schema: Separate Modeling of the Semantic Channel and Style Channel

![Figure 42-1: Dual-channel schema for semantic response and style control](../../images/part12/ch42_fig02_dual_channel_schema.svg)

*Figure 42-1: Dual-channel schema for semantic response and style control. The semantic channel answers "what to say," the style channel answers "with which voice and emotion to say it," and the acoustic supervision channel binds both to audio files, speech tokens, and sampling configuration.*

Figure 42-1 illustrates the core structure of VoiceStyleControl. The semantic channel is responsible for fields such as `query`, `answer`, `task`, and `language`; the style channel is responsible for fields such as `gender`, `mood`, `query_id`, and `answer_id`; the acoustic supervision channel is responsible for `query_audio_path`, `answer_audio_path`, `query_token_25hz`, `answer_token_25hz`, `speech_token_25hz`, and `sample_rate`. The three channels are merged in training records but must be checked separately during construction, quality inspection, and evaluation.

Separate channel modeling enables precise failure attribution. If the model generates correct response text but produces an unstable timbre, the issue typically lies in the style channel or the reference audio pool; if the acoustic condition is correct but characters are mispronounced, the issue lies in the semantic channel, ASR reverse-transcription, or synthetic text alignment; if the audio is playable but the token path cannot be read, the issue lies in the acoustic supervision channel or the packaging manifest. Collapsing all information into a single free-text prompt facilitates rapid sample assembly but makes downstream data repair and experimental attribution considerably harder.

An S2SEmoControl record expresses the mapping from the user side `(query_audio, query_text, query_gender, query_mood)` to the assistant side `(answer_text, answer_audio, answer_gender, answer_mood)`. Chinese conversational content, acoustic conditions, emotion labels, audio files, and speech tokens are bound together in a single record, making it not a loose combination of "text Q&A plus attached audio" but a complete voice interaction training sample.

```json
{
  "source": "S2SEmoControl",
  "task": "S2S",
  "query": "Tell me a short story.",
  "answer": "Sure, let me make up a short story for you. Once upon a time there was a very diligent little nightingale...",
  "query_gender": "female",
  "answer_gender": "male",
  "query_mood": "neutral",
  "answer_mood": "neutral",
  "language": "zh",
  "sample_rate": 16000,
  "query_id": "female-neutral-1",
  "answer_id": "male-neutral-2",
  "query_token_25hz": "S2SEmoControl/.../query_token_0.ark:3121",
  "query_audio_path": "S2SEmoControl/.../1977946a06cf564f1-query.wav",
  "answer_token_25hz": "S2SEmoControl/.../answer_token_0.ark:22637",
  "answer_audio_path": "S2SEmoControl/.../1977946a06cf564f1-answer.wav"
}
```

In this sample, the user says "Tell me a little story." and the assistant replies "Sure, let me make up a short story. Once upon a time there was a very diligent little nightingale...". `query_gender` is `female` and `answer_gender` is `male`; both `query_mood` and `answer_mood` are `neutral`. During training, `query_audio_path` and `query_token_25hz` can serve as speech understanding inputs, with `query` providing the transcribed semantic anchor; `answer` is the semantic target, and `answer_token_25hz` together with `answer_audio_path` provide the speech generation supervision; `answer_gender` and `answer_mood` specify the style conditions for the output voice.

TTSSpeakerControl concentrates the control capability in a text-to-speech form. The input text is split into two parts: `text` describes how the voice should express itself, while `answer` is the content to be spoken. For example, `text` may read "female, somewhat fearful, sweaty palms, trembling voice," and `answer` may read "Run, it's not safe here." This type of record indicates that the TTS subset does not randomly assign mood labels to sentences; instead, it constructs style–content pairs in which the natural-language style description, the structured label, and the content to be synthesized must mutually reinforce each other.

```json
{
  "source": "TTSSpeakerStyle",
  "task": "TTS",
  "text": "female, somewhat fearful, sweaty palms, trembling voice",
  "answer": "Run, it is not safe here",
  "gender": "female",
  "mood": "fearful",
  "language": "zh",
  "sample_rate": 16000,
  "answer_id": "female-fearful-1",
  "speech_token_25hz": "TTSSpeakerStyle/.../answer_token_0.ark:1379",
  "answer_audio_path": "TTSSpeakerStyle/.../c6810929-8962-4cc1-b3b5-aadd4cbb1106-answer.wav"
}
```

Combining samples from both S2S and TTS, the fields in VoiceStyleControl can be organized into six layers: task identifier, text content, acoustic conditions, emotion conditions, speech supervision, and basic audio configuration. S2S samples contain both user-side and assistant-side fields and therefore distinguish query-side from answer-side; TTS samples generate only assistant-side speech and therefore have a more concentrated set of fields. `language` fixes the language, and `sample_rate` fixes the audio sampling configuration; these foundational fields are the underlying contract for training loading and evaluation reproducibility and must not be inferred implicitly from path names or directory conventions alone.

**Table 42-2: Field Descriptions for Speaker, Emotion, and Sampling Labels**

| Label layer | Field | Values / examples | Distribution or engineering requirements |
|---|---|---|---|
| Query-side speaker | `query_gender` | `female` / `male`, e.g., `female` | Calculated separately for the query side. |
| Answer-side acoustic condition | `answer_gender` / `gender` | `male` / `female` | Before training, monitor balance by answer-side gender, mood, and reference acoustic condition to avoid output voice bias. |
| Query-side emotion | `query_mood` | `happy`, `angry`, `fearful`, `neutral`, `sad` | Five classes are nearly balanced in S2SEmoControl. |
| Answer-side emotion | `answer_mood` / `mood` | Same as above | Total counts as per Table 42-1; TTSSpeakerControl does not explicitly include `neutral`. |
| Language and sampling | `language` / `sample_rate` | `zh` / `16000` | Used as loading, resampling, and evaluation-reproducibility fields; not inferred implicitly from paths. |
| Reference voice citation | `query_id` / `answer_id` | e.g., `female-neutral-1` | Points to a style instance in the authorized reference voice pool; does not expose real identity. |

In VoiceStyleControl, emotion distribution is only the first layer of balancing information. When samples actually enter training and evaluation, they are further decomposed along the input-side and output-side axes: `query_gender × query_mood` describes the state distribution of user speech, `answer_gender/gender × answer_mood/mood` describes the target distribution of assistant-generated speech, and the reference voice ID constrains how the same acoustic condition is reused across different texts and emotions. Language and sampling rate appear foundational but determine whether loading, resampling, and audio metrics are comparable. Only by examining all these axes together can one determine whether a particular emotion is concentrated under a specific acoustic condition, whether a particular reference timbre appears too frequently in both training and test sets, and whether a model failure originates from the semantic, acoustic, or emotion control dimension.

S2S and TTS samples differ slightly in how they use the voice and emotion fields. S2S records both user-side and answer-side, so it uses `query_gender`, `answer_gender`, `query_mood`, and `answer_mood`; TTS generates only answer-side audio, so it uses `gender` and `mood`. Before training, these can be normalized into a unified view — for example, mapping TTS `gender` to `answer_gender` and `mood` to `answer_mood` — while retaining the source fields for traceability.

A unified JSON Schema constrains required fields by task type; a production-grade manifest should further add enum constraints, path-existence validation, file hashes, authorization IDs, tokenizer name, tokenizer version, and token frame rate declarations.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "VoiceStyleControlRecord",
  "type": "object",
  "required": [
    "source",
    "task",
    "answer",
    "language",
    "sample_rate",
    "answer_audio_path"
  ],
  "oneOf": [
    {
      "title": "S2SEmoControl",
      "required": [
        "query",
        "query_gender",
        "answer_gender",
        "query_mood",
        "answer_mood",
        "query_id",
        "answer_id",
        "query_audio_path",
        "answer_audio_path",
        "query_token_25hz",
        "answer_token_25hz"
      ],
      "properties": {
        "task": {
          "const": "S2S"
        }
      }
    },
    {
      "title": "TTSSpeakerControl",
      "required": [
        "text",
        "gender",
        "mood",
        "answer_id",
        "speech_token_25hz",
        "answer_audio_path"
      ],
      "properties": {
        "task": {
          "const": "TTS"
        }
      }
    }
  ],
  "properties": {
    "source": {
      "type": "string"
    },
    "task": {
      "enum": ["S2S", "TTS"]
    },
    "query": {
      "type": "string",
      "description": "Transcription of the spoken user query; used only in S2S"
    },
    "text": {
      "type": "string",
      "description": "Natural-language style description; used only in TTS"
    },
    "answer": {
      "type": "string",
      "description": "Assistant response or content to be synthesized"
    },
    "query_gender": {
      "type": "string"
    },
    "answer_gender": {
      "type": "string"
    },
    "gender": {
      "type": "string"
    },
    "query_mood": {
      "type": "string"
    },
    "answer_mood": {
      "type": "string"
    },
    "mood": {
      "type": "string"
    },
    "language": {
      "type": "string"
    },
    "sample_rate": {
      "type": "integer"
    },
    "query_id": {
      "type": "string"
    },
    "answer_id": {
      "type": "string"
    },
    "query_token_25hz": {
      "type": "string"
    },
    "answer_token_25hz": {
      "type": "string"
    },
    "speech_token_25hz": {
      "type": "string"
    },
    "query_audio_path": {
      "type": "string"
    },
    "answer_audio_path": {
      "type": "string"
    }
  }
}
```

The unified schema splits the training entry point into three parts: semantic input consists of `query`, `text`, or `answer` text tokens; style input consists of gender, mood, and reference voice ID; and the acoustic target is the answer-side speech token or audio. `answer_gender`, `answer_mood`, `gender`, and `mood` must not remain only in offline metadata — they must be mapped to control conditions or conditioning text in the dataloader; otherwise the model will never acquire genuine controllable generation capability.

Once training samples enter the dataloader, they are projected from the standard schema into task-specific views. The S2S view may take the form `query_audio + answer_gender + answer_mood -> answer_token`, optionally augmented with the `query` transcription as an auxiliary semantic input; the TTS view may take the form `text + answer + gender + mood -> speech_token`. The evaluation view, conversely, fixes certain fields while varying others — for example, fixing `answer` while varying `mood`, or fixing `mood` while varying `answer_id`. This design principle — stable record contract, variable training view — serves controllable speech generation experiments, not auxiliary speaker identification or voice-print modeling experiments.

## 42.4 Construction Pipeline: From Text Conversation to Controllable Voice Records

![Figure 42-2: VoiceStyleControl data construction pipeline](../../images/part12/ch42_fig01_data_pipeline.svg)

*Figure 42-2: VoiceStyleControl data construction pipeline. Text conversation or style content is first assigned speaker and emotion conditions, then audio is generated or collected through the authorized reference voice pool, and finally the samples are tokenized, quality-checked, balanced, and packaged.*

The construction of VoiceStyleControl can be divided into seven steps: text conversation or style content generation, style attribute assignment, authorized reference voice pool preparation, speech synthesis or collection, discrete speech tokenization, quality inspection and balancing, and packaging and release. Each step simultaneously affects semantic quality, style quality, and compliance risk.

This pipeline is not a simple sequential production line but a series of continuous data gates. After text content is generated, it must be determined whether the semantics are appropriate for the designated emotion; after reference voices are selected, it must be verified that the authorization covers the current task; after speech is synthesized, it must be confirmed that the audio, text, acoustic conditions, and emotion all pass simultaneously. If a problem is discovered at any step, the sample should not simply "flow downstream with a defect" — it must be returned to the corresponding queue for repair. Otherwise, downstream evaluation can only detect that the model is unstable but cannot explain where the instability originates.

The first step is generating or collecting text content. S2SEmoControl uses Qwen3-8B to generate or organize dialogue pairs, with each record containing a user `query` and an assistant `answer`. Queries span scenarios such as everyday requests, emotional expression, storytelling, explanation, and reminders; answers remain natural and complete and respect safety boundaries. TTSSpeakerControl uses emotion-specific prompts to generate style–content pairs so that the style description and the content to be spoken reinforce each other. For example, fearful samples may be more urgent and sad samples more subdued, but emotion labels must not be used as pretexts for hazardous inducement.

Acceptance of text content looks beyond grammatical fluency to whether the emotion and semantics are compatible. `fearful` can correspond to "Run, it's not safe here" but should not appear in a casual chat as exaggerated scaremongering; `angry` can serve character-driven expression but should not treat abusive, threatening, or discriminatory content as emotional enhancement. If no boundaries are set during the text generation stage, subsequent speech synthesis will convert risky text into more impactful audio — amplifying the risk through acoustic expression.

The second step is assigning style attributes. For S2S, gender and mood must be assigned separately to both the query side and the answer side; for TTS, gender and mood are assigned to the answer side only, with a natural-language style description written into `text`. The assignment strategy must consider both balance and combination coverage: balance ensures that every emotion has a sufficient number of samples, and combination coverage ensures the model has seen transfers from diverse user styles to diverse assistant styles. If the data contains only same-gender, same-mood combinations, the model will easily couple input style and output style, weakening answer-side control capability.

Combination coverage is especially important for the S2S subset. A user-side angry query does not imply the assistant-side should also be angry; a user-side fearful query does not imply the assistant-side should be equally fearful. On the contrary, many real products require the assistant to remain neutral, clear, and action-oriented under high-pressure user emotions. Data construction should retain enough cross-combination samples — for example, a female-angry query paired with a male-neutral answer, or a male-sad query paired with a female-neutral answer — so that the model learns to treat user state as an understanding signal rather than simply copying it as output style.

The third step is preparing the reference voice pool. VoiceStyleControl uses a multi-speaker, multi-emotion reference pool and synthesizes speech in the target style via CosyVoice2 using zero-shot voice cloning. The engineering priority is not "clone as closely as possible" but "authorizable, reusable, and revocable." Reference audio should document reference voice ID, emotion condition, collection time, permitted use scope, authorization status, and revocation status; `query_id` and `answer_id` should expose only engineering references and must not contain real names or information that allows identity reversal.

The fourth step is speech synthesis or collection. S2S requires generating both query speech and answer speech and binding each audio file to its corresponding text record; TTS generates answer-side speech from `text` and `answer`. During synthesis, sampling rate should be fixed or explicitly recorded; loudness, silence, maximum duration, and file encoding should be controlled to prevent instability caused by abnormal audio lengths or formats in the dataloader during training. If real recordings are used, additional handling is required for environmental noise, microphone variation, speaker fatigue, and third-party background sounds.

The fifth step is discrete speech tokenization. S3Tokenizer converts waveforms into discrete speech tokens, enabling speech generation to be formulated as a sequence modeling task. S2S records use `query_token_25hz` and `answer_token_25hz`; TTS records use `speech_token_25hz`. VoiceStyleControl unifies audio tokens at 25 Hz; when releasing the data, the manifest should still bind the tokenizer name, version, frame rate, codebook configuration, and reconstruction method. The worst scenario for a training set is "same field name, different meanings": if the same field is generated by different frame rates or different tokenizer versions across batches, the model will receive inconsistent supervision in sequence length and acoustic granularity.

The sixth step is quality inspection, balancing, and splitting. Quality inspection must go beyond checking whether audio can be played; it must also verify whether text and audio are consistent, whether the target acoustic condition matches, whether the emotion is perceptible, whether audio quality is stable, whether paths exist, and whether tokens are readable. Balancing should not be performed only by total emotion count; it must also monitor across `task`, `language`, `sample_rate`, reference voice ID, text length, and audio duration. Splitting should apply isolation by reference voice ID to prevent the same reference timbre from appearing in both the training set and the test set, which would inflate acoustic condition evaluation scores.

The seventh step is packaging. Final samples can be stored in JSONL, Parquet, or Hugging Face Dataset format, but the training manifest must retain audio paths, token paths, hashes, authorization status, and data version. Audio files, token ark files, and metadata should not be loosely associated by human naming conventions but must be strictly bound by the manifest. Only then, when a sample is re-synthesized, re-annotated, or removed, can the team identify which training versions are affected.

The packaging artifacts include not only JSONL, Parquet, or Hugging Face Dataset files but also a data card describing the data boundaries. The data card records total sample count, subset composition, emotion distribution, gender field distribution, reference voice IDs, language, sampling rate, tokenizer version, authorization scope, and splitting strategy, and distinguishes training conditions, audit metadata, and anonymized fields in the public release. This boundary statement prevents `answer_id` from being misused as a real identity label and prevents `mood` from being treated as a reliable ground truth requiring no verification.

## 42.5 Quality Assessment and Closed-Loop Remediation

![Figure 42-3: Quality assessment and data flywheel closed loop](../../images/part12/ch42_fig03_quality_loop.svg)

*Figure 42-3: Quality assessment and data flywheel closed loop. Automated validation, reverse ASR, style assessment, and manual sampling together form a defective-sample queue that feeds back into re-synthesis, re-annotation, downweighting, or removal.*

Quality assessment for controllable voice interaction data must simultaneously cover semantics, voice, emotion, audio, and safety. A sample that "sounds human" in isolation is not necessarily acceptable: it may contain misread text, a mismatched voice identity, overly intense emotion, or inappropriate fearful delivery in a hazardous scenario. The quality system should combine automated metrics with human review in a closed loop; defective samples enter queues for re-synthesis, re-annotation, downweighting, or removal.

Quality gates should be divided into "hard failures" and "soft risks." Missing paths, incorrect sampling rates, corrupted audio, unreadable tokens, and severe ASR reverse-transcription inconsistency typically constitute hard failures and should be blocked immediately. Slightly weak emotion intensity, average naturalness, or borderline acoustic condition perception can enter a soft-risk queue, where the decision to re-synthesize, downweight, or manually review is made based on task criticality. Treating every issue as a disqualifying veto wastes remediable samples; allowing every issue to pass dilutes the control signal with noise.

**Table 42-3: Quality Assessment Metrics**

| Assessment dimension | Core question | Automated metrics | Key points for human review | Handling of failures |
|---|---|---|---|---|
| Semantic consistency | Does the answer address the user's intent? Is TTS content read out correctly? | ASR reverse-transcription CER/WER, semantic similarity, intent hit rate | Non-responsive answers, omission of key information, hazardous suggestions | Rewrite text, re-synthesize, remove |
| Acoustic condition consistency | Does the output match the target gender, mood, and reference acoustic condition? | Field-level consistency check, automated/human gender verification, reference timbre spot-check | Target condition errors, cross-sample voice bleeding, timbre too close to an unauthorized real person | Re-select reference audio, re-synthesize, downweight or isolate |
| Emotion control | Is the target mood stably expressed? | Emotion classification accuracy, confusion matrix, F0/energy/speech-rate statistics | Emotion too intense, conflict with semantics, or potentially manipulative | Re-annotate, reduce intensity, remove |
| Audio quality | Can the audio serve as generation supervision? | SNR, loudness, silence ratio, clipping rate, MOS/NISQA | Clipping, broken phrasing, mechanical artifacts, background noise | Denoise, resample, re-synthesize |
| Conversational naturalness | Is the S2S response natural? Is the persona stable? | Multi-turn coherence score, latency and duration distribution | Abrupt tone, persona inconsistency, repeated style jumping | Reorder, add context, manual review |
| Safety and compliance | Is the sample authorizable, traceable, and revocable? | Authorization record completeness rate, watermark detection rate, audit log coverage | Risks of impersonation, manipulation, or replication of sensitive identities | Block, anonymize, remove, and audit |

Semantic consistency can be established via reverse ASR as a first layer of automated checking. Synthesized audio is transcribed back to text; CER/WER is computed and compared against `answer`; for S2S, the answer is also checked for relevance to the query. If "Run, it's not safe here" is synthesized as "Walk slowly, it's safe here," the sample must be removed regardless of audio quality. Semantic similarity and LLM-as-judge can assist in locating issues, but human spot-checking must be retained for safety-sensitive or high-emotion samples.

Acoustic condition consistency focuses on whether the generated output matches the sample's target gender, mood, and reference acoustic condition — not on training or evaluating a separate speaker identification model. On the answer side, `answer_id` should be consistent with `answer_gender` and `answer_mood`; on the query side, `query_id` should be consistent with user-side labels. If the same `answer_id` exhibits noticeably different timbres across different samples, the reference pool, synthesis parameters, and tokenization pipeline must be traced. Human listening checks and automated verification are quality inspection tools only and do not change the dataset's training objective.

Emotion control evaluation cannot rely solely on classifier confidence. Happy often manifests as higher energy and faster pace; sad may manifest as slower speech rate and lower energy; fearful may be accompanied by trembling, urgency, or unstable pauses; angry may manifest as stronger energy and harder delivery. However, Chinese linguistic expression, speaker variation, and content semantics all alter acoustic presentation, so the evaluation target should be "perceptible and consistent with the text," not a fixed acoustic template for each emotion.

Closed-loop remediation should preserve failure type information. Semantic errors are sent back to text generation or ASR reverse-transcription; acoustic condition errors are sent back to reference voice selection or synthesis parameters; emotion errors are sent back to style description, emotion labels, or the synthesis model; audio quality errors are sent back to waveform processing; compliance errors enter isolation, removal, and audit workflows. Every remediation should generate a new version rather than overwrite the source file. Only then can subsequent model performance changes be traced to data changes rather than becoming unexplainable training fluctuations.

## 42.6 Evaluation Protocol: Making Controllability Comparable

The evaluation set should be constructed independently from the training set logic, with particular care to prevent the same reference voice ID from appearing in both training and test sets. For S2SEmoControl, evaluation samples should cover combinations of different query emotions mapped to different answer emotions; for TTSSpeakerControl, evaluation samples should cover the same `answer` under different `text/gender/mood` conditions. An effective evaluation protocol does not merely ask "does the generated voice sound good" — it also asks "whether the same sentence genuinely differs across different control conditions, and whether those differences are reasonable."

The evaluation set can be divided into three types of slices. The first type is the standard slice, covering the main task distribution in the training set, used to observe overall usability. The second type is the counterfactual slice, fixing text or reference voice ID and varying only the mood or gender condition, used to verify whether control fields are effective. The third type is the safety slice, containing scenarios such as identity impersonation, high-pressure emotion, sensitive professions, financial verification codes, and medical advice, used to check whether the model might misuse "controllable generation" as "controllable manipulation." The findings from these three slice types must not be merged into a single aggregate score, as high-quality audio samples could otherwise mask high-risk behaviors.

Semantic evaluation consists of two layers: content fidelity and dialogue relevance. Content fidelity checks whether TTS output accurately reads out `answer` and whether S2S output can be transcribed to text that is semantically consistent with the target answer. Dialogue relevance checks whether the S2S answer addresses the query rather than generating fluent but irrelevant sentences. Evaluation can combine ASR reverse-transcription, semantic similarity, LLM-as-judge, and human review, but scoring prompts, model versions, and human annotation guidelines must be preserved to prevent evaluation drift over time.

Acoustic condition evaluation should also be layered. The structural label layer checks whether `answer_gender/gender` and `answer_mood/mood` are consistent with sample targets; the perceptual layer checks whether the generated audio matches the corresponding reference acoustic condition and emotional expression; the isolation layer checks whether the model is excessively close to an unauthorized individual or leaks the voice print of a real person in the training set. The evaluation objective is not to construct voice-print similarity rankings or to treat "as similar as possible to a specific real person" as the sole optimization direction; it is to confirm that the model can generate reasonable, compliant, emotionally expressive speech under the sample conditions.

Emotion evaluation requires constructing counterfactual sets. For example: fix a neutral sentence and request happy, angry, fearful, and sad in turn; or fix `gender` and vary `mood`; or fix `mood` and vary `gender`. This paired evaluation approach reveals whether the model genuinely uses the control fields. If all outputs vary only in volume while speech rate, pauses, and prosody do not change with mood, the model may have learned only shallow intensity adjustment.

Audio quality evaluation includes both objective metrics and subjective scores. Objective metrics cover duration distribution and automated MOS; subjective scores focus on naturalness, intelligibility, emotional credibility, and conversational comfort. Safety evaluation should serve as a release gate: scenarios including identity impersonation, sensitive professions, financial verification codes, medical advice, minors, and high-pressure emotional inducement must all be checked to ensure the system does not generate output using strong emotions or specific timbres in inappropriate contexts.

Evaluation results should also be written back to the data version, not stored only in model reports. If a particular model version achieves high emotion classification accuracy on fearful but low human comfort scores, the data may have constructed fearful as an overly intense or overly theatrical expression; if the reference acoustic condition increasingly resembles a recognizable real person and compliance risk rises, the reference audio or evaluation target may be over-optimizing for identity replication. Only by feeding these findings back into sample filtering, proportion adjustment, and synthesis strategy will evaluation genuinely improve the next version of data.

## 42.7 Governance of Privacy, Authorization, and Misuse Risks

Voice identity is a highly sensitive data asset. A person's voice contains cues about age, gender, regional background, emotional state, health condition, and personal identity; in speaker verification systems, voice can even function as an authentication credential. Once controllable voice data incorporates voice cloning, authorization, revocation, usage restriction, and auditing must be embedded in the data lifecycle — not appended as disclaimer footnotes at model release time.

**Table 42-4: Privacy and Misuse Risk Control Checklist**

| Risk type | Triggering scenario | Control measures | Audit evidence |
|---|---|---|---|
| Voice identity authorization | Reference audio originates from real speakers or identifiable voices | Pre-collection consent, purpose limitation, revocability, authorization version number | Authorization timestamp, revocation records |
| Emotional manipulation | Using fearful, angry, or intimate delivery to influence user judgment | Prohibit strong emotion in high-risk scenarios, prompt review, minor protection | Human review forms |
| Privacy leakage | Audio contains names, phone numbers, addresses, or background speakers | ASR anonymization, background sound filtering, data minimization, retention period | Anonymization report, deletion request handling records |
| Bias and stereotyping | `gender` persistently correlated with `mood` or content type | Distribution monitoring, counterfactual samples, ban on gender-stereotyping templates | Distribution reports, bias evaluation results |
| Version loss of control | Samples re-synthesized or re-annotated without traceability | Data version management, hashing, training set freezing | Experiment tracking IDs |

Table 42-4 implements risk governance as data gates. References with missing authorization must not enter the synthesis queue; references with revoked authorization must be traceable to all derived audio and tokens; high-risk emotional manipulation samples must not rely solely on post-training safety strategies — they must be blocked or downweighted during data construction. For voice generation, compliance is not the final filter before launch but an integral part of the sample lifecycle.

The reference voice pool is the governance focal point. Every reference should have a `consent_id`, authorization scope, collection method, permitted tasks, expiration time, and revocation status. If authorization covers research use only, samples must not enter commercial model training; if a speaker revokes authorization, the manifest must be able to identify all affected `query_id/answer_id` values, audio files, token files, and training versions. When releasing externally, reference IDs that cannot be reverse-mapped to real identities should be used wherever possible; voice IDs, file names, or paths should not be designed as real names.

Emotion control also has misuse boundaries. Strong emotions such as fearful and angry can enhance expressiveness but may also be used to manipulate users. Scenarios in customer service, education, healthcare, and finance should restrict high-pressure emotional output; in particular, fearful delivery must not be used to induce users to transfer funds, make purchases, reveal verification codes, or make health decisions. For minors and emotionally vulnerable individuals, systems should default to neutral or gently supportive styles and retain policy trigger logs.

Privacy protection also encompasses content anonymization. Voice samples may contain names, addresses, phone numbers, account numbers, geographic locations, or background third-party speech. Even though VoiceStyleControl is primarily generated from synthetic text, the engineering pipeline should still retain ASR anonymization, sensitive-word scanning, background sound detection, and human spot-checking. If real user voice feedback is introduced later, user consent, data minimization, retention periods, deletion requests, and purpose-change notifications must all be incorporated into platform workflows.

Bias governance is equally important. If women's voices are consistently associated with fearful or sad in the training set while men's voices are more associated with angry, the model will learn and amplify these stereotypes. Therefore, gender statistics must not remain at the level of marginal proportions; they must be examined in cross-tabulation views of `query_gender`, `answer_gender/gender`, and `mood`. The evaluation set should also include counterfactual samples to check whether emotional expression for the same content is equitable across different genders.

## 42.8 Connections to Adjacent Chapters in Data Engineering

VoiceStyleControl inherits the foundational capabilities of audio and video data engineering. The audio segmentation, ASR, noise reduction, speaker separation, and temporal alignment discussed in Chapter 10 are further refined into a more precise sample contract: one must know not only which text a given audio segment corresponds to, but also which reference voice ID generated it, at what mood, at what sampling rate, and at what token frequency. An ordinary audio pipeline addresses "can alignment be achieved"; controllable voice interaction further addresses "once aligned, can the voice be generated conditionally."

It also connects to multi-turn interaction data. When Chapter 20 examines agent memory and multi-turn context, role, intent, and historical state are the primary variables; when interaction enters voice form, the assistant's persona also manifests in timbre and emotional stability. A multi-turn voice assistant cannot present a neutral male voice in the first turn, then inexplicably switch to a fearful female voice in the second, and an angry male voice in the third. Consequently, `answer_gender`, `answer_mood`, and `answer_id` can become part of the voice agent's memory, used to maintain voice identity across continuous sessions.

Online feedback loops will move voice style from offline labels toward user experience. The clicks, satisfaction scores, corrections, and complaints in Chapter 23 manifest in voice products as feedback such as "can't hear clearly," "too rushed," "too harsh," "doesn't sound like before," or "emotion is inappropriate." This feedback cannot be converted directly into training samples; it should first enter an evaluation queue to determine whether the error is semantic, audio quality, style, or safety policy, and then decide whether to re-synthesize, re-annotate, adjust proportions, or revise rejection rules.

The privacy compliance chapters define boundaries for VoiceStyleControl. Chapter 36's data compliance framework requires that authorization, purpose, retention, and auditing be placed at the front of the data lifecycle; Chapter 37's privacy protection techniques remind us that voice identity risk can be reduced through access control, federated training, encrypted storage, and data minimization. The more strongly controllable voice data emphasizes acoustic conditions and reference timbres, the less it can treat compliance as an appendix.

In the context of multimodal generative data engineering, VoiceStyleControl shares a core pattern with Chapter 48: decomposing generation targets into content conditions and style conditions, then binding training supervision with a structured schema. The prompt, style, motion, camera, and safety tag of T2I/T2V correspond in voice to `answer`, `gender`, `mood`, reference voice ID, `sample_rate`, and audio token. The end-to-end LLM data flywheel in Part 14 Project 10 can also absorb this design: construct an initial version of voice data offline, train a controllable generation model, collect experience feedback online, feed it back into quality inspection and balancing, and then release the next version of data and model.

## Chapter Summary

The value of VoiceStyleControl lies not in simply accumulating voice samples to a larger scale but in placing semantic response, acoustic conditions, emotion control, and speech generation supervision together in a single auditable record. S2SEmoControl provides interaction supervision from spoken query to spoken answer; TTSSpeakerControl provides direct supervision from natural-language style description to target speech. Together, they enable the model both to understand user speech and to generate responses according to specified acoustic conditions and emotions.

Key data engineering work includes: explicitly separating the semantic channel from the style channel and retaining control fields such as `query_gender`, `answer_gender`, `query_mood`, `answer_mood`, `gender`, and `mood`; writing `sample_rate`, audio paths, speech token paths, and tokenizer version into the data contract; constructing an evaluation protocol jointly from ASR reverse-transcription, acoustic condition verification, emotion recognition, audio quality metrics, and human review; and implementing authorization, revocation, watermarking, and auditing within the reference voice pool and voice cloning pipeline.

As voice interaction moves from "capable of speaking" to "speaking in a controllable manner," the boundaries of a dataset shift accordingly. Every sample must answer four questions: Is the content correct? Does the acoustic condition match the target specification? Does the emotion satisfy the control condition? Is the generation process compliant and traceable? Only when all four questions are answered affirmatively can controllable voice interaction data function as a reliable training asset.

## References

An K, Chen Q, Deng C, Du Z, Gao C, Gao Z, Gu Y, He T, Hu H, Hu K, others (2024) FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs. arXiv preprint arXiv:2407.04051.

Chanfungjan (n.d.) VoiceStyleControl. GitHub repository. https://github.com/Chanfungjan/VoiceStyleControl.

Du Z, Chen Q, Zhang S, Hu K, Lu H, Yang Y, Hu H, Zheng S, Gu Y, Ma Z, Gao Z, Yan Z (2024) CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens. arXiv preprint arXiv:2407.05407.

Du Z, Wang Y, Chen Q, Shi X, Lv X, Zhao T, Gao Z, Yang Y, Gao C, Wang H, others (2024) CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models. arXiv preprint arXiv:2412.10117.

Mittag G, Naderi B, Chehadi A, Möller S (2021) NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets. In: Interspeech 2021, pp 2127–2131.

Song X (n.d.) S3Tokenizer: Reverse Engineering of Supervised Semantic Speech Tokenizer proposed in CosyVoice. GitHub repository. https://github.com/xingchensong/S3Tokenizer.

Yang A, Li A, Yang B, Zhang B, Hui B, Zheng B, Yu B, Gao C, Huang C, Lv C, others (2025) Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.
