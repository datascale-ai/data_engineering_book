<<<<<<< HEAD
# Chapter 42: VoiceStyleControl Controllable Voice Interaction Data Engineering

When dialogue moves from text into speech, the supervision target changes in a fundamental way. Text dialogue data mainly answers two questions: what does the user want, and what should the assistant say? Voice interaction data must also answer who should say the sentence, with what emotion, and whether the resulting voice is audible, controllable, and suitable for the setting. The sentence "Run now, this place is not safe" carries the same text whether it is read calmly or spoken with a trembling, fearful voice, but the training signal is completely different.

That is the value of VoiceStyleControl. It does not simply concatenate ASR transcripts, TTS readings, and dialogue text. It places the user request, assistant response text, target voice condition, emotional style, and corresponding speech supervision into one auditable record. Text fields describe what the turn is about. Style fields describe what voice and emotion should be used. Audio files and discrete speech tokens provide acoustic targets that can be learned and rechecked during generation. The model learns not only response content, but how to generate emotional speech under specified semantic and style conditions.

The engineering entry point for this sample organization is the public repository [Chanfungjan/VoiceStyleControl](https://github.com/Chanfungjan/VoiceStyleControl). The two subsets discussed below, S2SEmoControl and TTSSpeakerControl, are organized around this family of structured records. The former binds spoken queries, assistant answers, and speech supervision on both sides of a dialogue. The latter binds style description, response text, and target speech into a controllable TTS record.

As a controllable voice interaction case, VoiceStyleControl builds on the audio and video data engineering ideas from Chapter 10: sampling rate, audio segmentation, ASR, speaker characteristics, and acoustic quality remain foundational. It also connects to multi-turn interaction in Chapter 20, online feedback in Chapter 23, and privacy and compliance in Chapters 36 and 37. Looking ahead, it shares a pattern with the multimodal generation work in Chapter 48: separate the generation target into content conditions and style conditions, fix them in a structured schema, and feed the result into an end-to-end data flywheel.

The engineering focus is not the TTS architecture or voice-cloning algorithm itself. The focus is how control conditions are recorded, how they enter training, and how audio quality, dialogue naturalness, and compliance boundaries are balanced. Only when these questions are carried by stable data structures and processes can VoiceStyleControl become a reusable controllable voice interaction data asset instead of a collection of pleasant-sounding synthetic samples.

## 42.1 Why Voice Dialogue Needs Explicit Style Control

Ordinary text dialogue samples usually contain role boundaries, context, user request, and assistant answer. If the role boundary, text length, safety label, and training mask are clear, the model can learn an input-output mapping over text tokens. Speech samples add an acoustic state that text cannot replace: sampling rate, duration, silence, loudness, noise, speaker identity, prosody, emotion, and discrete speech tokens all affect training. The answer text alone says what was said, but not how it should be spoken.

The difference between controllable voice interaction data and ordinary ASR or TTS corpora is therefore not just that there are more fields. The task definition changes. ASR asks which text corresponds to a piece of audio. Ordinary TTS asks whether a piece of text can be read naturally. Controllable voice interaction also asks what voice, what emotion, and what intensity should be used inside a conversation. If these conditions are not expressed explicitly, the model can only treat voice differences as random variation in the training audio. It will be hard to respond reliably to conditions such as "say this sadly" or "use this kind of voice" at inference time.

First, voice dialogue must separate content from expression. What the user said and what the assistant should answer live at the semantic layer. Who speaks the sentence, how fast it is spoken, how much energy it carries, where it pauses, and whether the emotion is perceptible live at the expression layer. Text dialogue data can often organize only the semantic layer. Voice generation data must make the expression layer part of supervision.

Second, voice dialogue must distinguish understanding the user's voice from generating the assistant's voice. Real users may be anxious, angry, hesitant, accented, or speaking in a noisy environment. The assistant, however, usually needs to maintain a stable voice condition and an emotion policy set by the product. A customer-service assistant should not become angry simply because the user is angry. A companion assistant should not change timbre without reason on every turn. Explicit style control lets data distinguish input-side state from output-side target at the sample level.

Third, emotion must be grounded in sound rather than only described in text. Happy, angry, fearful, neutral, and sad are not merely labels. They are reflected in pitch, energy, speed, pauses, and rhythm. The learning target is not to memorize an emotion word, but to generate speech consistent with the target expression state. For this reason, controllable voice data must keep text content, target style description, and speech supervision together.

Fourth, speech needs recheckable acoustic supervision. Text can enter training directly as token sequences. Speech passes through audio files, sampling rate checks, duration constraints, loudness normalization, silence trimming, discrete speech-token extraction, and other engineering steps. Explicit style control cannot be a side note that says "speak happily." It needs a target audio signal that shows what that style condition means acoustically.

This boundary matters for product experience. A companion assistant may be designed as warm, stable, and not overly dramatic. An audiobook character may need stronger emotional expression and a clearer role voice. A customer-service assistant usually needs to remain neutral and clear when the user is angry. All three may share the same semantic response ability, but they have different voice-identity, emotion-intensity, and risk boundaries. If samples do not distinguish these conditions explicitly, the model treats style as noise and loses controllability at inference time.

From a data engineering perspective, explicit style control also changes sample acceptance. A text sample can often enter the candidate pool once the user question and assistant answer match. A voice sample must satisfy text consistency, usable audio, correct target voice condition, perceptible emotion, and traceable authorization. Any failed dimension affects training: correct text with a wrong voice condition weakens conditional control; correct voice with a wrong emotion weakens emotion control; strong emotion applied to unsafe content can make risky behavior more persuasive.

## 42.2 Dataset Overview: Two Complementary Subsets

VoiceStyleControl contains two task families: speech-to-speech dialogue generation and controllable speech generation from text. Both serve the same goal: enabling a model to generate emotional speech from semantic content, voice conditions, and emotion style. They provide supervision from different angles.

The full dataset contains **189,375 samples**. S2SEmoControl contains **54,586 samples**, about **28.8%** of the total, and targets style-controllable speech-to-speech dialogue generation. TTSSpeakerControl contains **134,789 samples**, about **71.2%** of the total, and targets controllable text-to-speech generation. S2SEmoControl is closer to a real voice assistant: the model must understand the user's spoken request and generate assistant-side speech. TTSSpeakerControl more directly trains the mapping from style text, voice condition, and emotion style to target speech.

**Table 42-1: VoiceStyleControl scale and emotion distribution**

| Emotion | S2SEmoControl | TTSSpeakerControl | Total | Total ratio |
| --- | ---: | ---: | ---: | ---: |
=======
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
>>>>>>> upstream/main
| happy | 10,937 | 38,500 | 49,437 | 26.1% |
| angry | 11,022 | 38,054 | 49,076 | 25.9% |
| fearful | 10,799 | 24,925 | 35,724 | 18.9% |
| neutral | 10,797 | 0 | 10,797 | 5.7% |
| sad | 11,031 | 33,310 | 44,341 | 23.4% |
| **Total** | **54,586** | **134,789** | **189,375** | **100.0%** |

<<<<<<< HEAD
S2SEmoControl is close to balanced across five emotions, with about 10.8k to 11.0k samples per class. TTSSpeakerControl covers four expressive emotions: happy, angry, fearful, and sad. It does not explicitly include neutral. This is not accidental. S2S dialogue needs neutral as a stable baseline; otherwise the model can learn to make every reply high intensity. The larger TTS subset concentrates capacity on styles where acoustic variation is more important, such as happy, angry, fearful, and sad speech.

Neither subset is just "text plus audio." Each record contains at least five kinds of information: task source and task type, text-side content, voice and emotion conditions, speech-generation supervision, and basic audio configuration. These fields jointly determine whether a sample can train conditional emotional speech generation. Task information determines how the sample is loaded; text content provides the semantic target; voice and emotion conditions define the generation style; speech supervision provides the learnable acoustic target; audio configuration makes training and evaluation reproducible.

The two subsets play complementary roles. TTSSpeakerControl provides the larger foundation for style generation: it teaches the model to map natural-language style descriptions, voice conditions, and emotion style into target speech. S2SEmoControl is smaller but closer to real assistant interaction: the model must understand user-side speech before generating assistant-side speech. Used together, the TTS subset gives stable style-generation supervision, while the S2S subset returns that ability to dialogue context.

VoiceStyleControl should therefore not be read as a plain TTS dataset. Ordinary TTS supervision says, given text, read the text. VoiceStyleControl supervision says, given semantic content and style conditions, generate speech that fits the dialogue goal. The former mainly cares about pronunciation, naturalness, and audio quality. The latter must also care about user state, assistant voice condition, emotion choice, cross-turn consistency, and safety boundary. Once the data objective changes, schema, balancing, splitting, and evaluation all change with it.

## 42.3 Sample Schema: Separate Semantic and Style Channels

![Figure 42-1: Dual-channel schema for semantic response and style control](../../images/part12/ch42_fig02_dual_channel_schema.svg)

*Figure 42-1: The semantic channel answers what to say; the style channel answers what voice and emotion to use; the acoustic supervision channel binds both to audio files, speech tokens, and sampling configuration.*

Figure 42-1 shows the core structure of VoiceStyleControl. The semantic channel contains fields such as `query`, `answer`, `task`, and `language`. The style channel contains `gender`, `mood`, `query_id`, and `answer_id`. The acoustic supervision channel contains `query_audio_path`, `answer_audio_path`, `query_token_25hz`, `answer_token_25hz`, `speech_token_25hz`, and `sample_rate`. The channels are merged into one training record, but they should be checked separately during construction, quality control, and evaluation.

Channel separation makes failures easier to diagnose. If the generated answer text is correct but the timbre is unstable, the likely issue is in the style channel or reference voice pool. If the voice condition is correct but the words are wrong, the issue lies in the semantic channel, reverse ASR, or synthesis-text alignment. If the audio plays but the token path cannot be read, the issue is in the acoustic supervision channel or packaged manifest. Collapsing all information into one free-text prompt may be convenient for quick sample assembly, but it makes later repair and experiment attribution much harder.

S2SEmoControl records a mapping from the user side, `(query_audio, query_text, query_gender, query_mood)`, to the assistant side, `(answer_text, answer_audio, answer_gender, answer_mood)`. Dialogue text, voice conditions, emotion labels, audio files, and speech tokens are bound in one record. It is therefore not a loose combination of a text QA pair and attached audio. It is a complete voice-interaction training sample.
=======
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
>>>>>>> upstream/main

```json
{
  "source": "S2SEmoControl",
  "task": "S2S",
  "query": "Tell me a short story.",
<<<<<<< HEAD
  "answer": "Sure. Let me make up a short story for you. Once there was a very diligent nightingale...",
=======
  "answer": "Sure, let me make up a short story for you. Once upon a time there was a very diligent little nightingale...",
>>>>>>> upstream/main
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

<<<<<<< HEAD
In this sample, the user asks for a short story and the assistant replies with the beginning of one. `query_gender` is `female`; `answer_gender` is `male`; both `query_mood` and `answer_mood` are `neutral`. During training, `query_audio_path` and `query_token_25hz` can serve as speech-understanding input, while `query` provides a transcript anchor. `answer` is the semantic target. `answer_token_25hz` and `answer_audio_path` are speech-generation supervision. `answer_gender` and `answer_mood` specify the output style condition.

TTSSpeakerControl concentrates control ability into a text-to-speech form. The input text is split into two parts: `text` describes how the voice should express itself, and `answer` is the content to read. For example, `text` may describe a female voice that sounds afraid, with trembling delivery, while `answer` says "Run now, this place is not safe." Such records build style-content pairs: natural-language style description, structured labels, and target content should support one another.
=======
In this sample, the user says "Tell me a little story." and the assistant replies "Sure, let me make up a short story. Once upon a time there was a very diligent little nightingale...". `query_gender` is `female` and `answer_gender` is `male`; both `query_mood` and `answer_mood` are `neutral`. During training, `query_audio_path` and `query_token_25hz` can serve as speech understanding inputs, with `query` providing the transcribed semantic anchor; `answer` is the semantic target, and `answer_token_25hz` together with `answer_audio_path` provide the speech generation supervision; `answer_gender` and `answer_mood` specify the style conditions for the output voice.

TTSSpeakerControl concentrates the control capability in a text-to-speech form. The input text is split into two parts: `text` describes how the voice should express itself, while `answer` is the content to be spoken. For example, `text` may read "female, somewhat fearful, sweaty palms, trembling voice," and `answer` may read "Run, it's not safe here." This type of record indicates that the TTS subset does not randomly assign mood labels to sentences; instead, it constructs style–content pairs in which the natural-language style description, the structured label, and the content to be synthesized must mutually reinforce each other.
>>>>>>> upstream/main

```json
{
  "source": "TTSSpeakerStyle",
  "task": "TTS",
<<<<<<< HEAD
  "text": "female, slightly fearful, tense, voice trembling",
  "answer": "Run now, this place is not safe",
=======
  "text": "female, somewhat fearful, sweaty palms, trembling voice",
  "answer": "Run, it is not safe here",
>>>>>>> upstream/main
  "gender": "female",
  "mood": "fearful",
  "language": "zh",
  "sample_rate": 16000,
  "answer_id": "female-fearful-1",
  "speech_token_25hz": "TTSSpeakerStyle/.../answer_token_0.ark:1379",
  "answer_audio_path": "TTSSpeakerStyle/.../c6810929-8962-4cc1-b3b5-aadd4cbb1106-answer.wav"
}
```

<<<<<<< HEAD
Across S2S and TTS samples, fields can be grouped into six layers: task identity, text content, voice condition, emotion condition, speech supervision, and audio configuration. S2S records include both user-side and assistant-side fields; TTS records focus on assistant-side speech. `language` and `sample_rate` are basic contractual fields for loading, resampling, and evaluation. They should not be inferred only from path names or directory conventions.

**Table 42-2: Speaker, emotion, and sampling fields**

| Label layer | Fields | Values / examples | Distribution or engineering requirement |
| --- | --- | --- | --- |
| Query-side speaker | `query_gender` | `female` / `male` | Count separately on the query side. |
| Answer-side voice condition | `answer_gender` / `gender` | `male` / `female` | Monitor balance by answer-side gender, mood, and reference voice condition before training. |
| Query-side emotion | `query_mood` | `happy`, `angry`, `fearful`, `neutral`, `sad` | S2SEmoControl is close to balanced across five classes. |
| Answer-side emotion | `answer_mood` / `mood` | same set | Overall counts follow Table 42-1; TTSSpeakerControl does not explicitly include `neutral`. |
| Language and sampling | `language` / `sample_rate` | `zh` / `16000` | Used for loading, resampling, and reproducible evaluation, not only for path inference. |
| Reference voice handle | `query_id` / `answer_id` | `female-neutral-1` | Points to a style instance in the authorized reference pool without exposing real identity. |

Emotion distribution is only the first layer of balancing. During training and evaluation, samples must also be separated by input side and output side. `query_gender x query_mood` describes the state of the user's speech. `answer_gender/gender x answer_mood/mood` describes the target distribution for generated assistant speech. Reference voice ID constrains how the same voice condition is reused across texts and emotions. Looking across these axes helps identify whether an emotion is concentrated in a single voice condition, whether a reference timbre appears too often in both train and test, and whether a model failure comes from semantics, voice condition, or emotion control.

S2S and TTS use voice and emotion fields slightly differently. S2S records both sides and therefore uses `query_gender`, `answer_gender`, `query_mood`, and `answer_mood`. TTS generates only answer-side audio, so it uses `gender` and `mood`. Before training, a normalized view can map TTS `gender` to `answer_gender` and `mood` to `answer_mood`, while retaining the original fields for traceability.

A joint JSON Schema should constrain required fields by task type. A production manifest should also add enum constraints, path-existence checks, file hashes, authorization IDs, tokenizer name, tokenizer version, and token frame-rate declarations.
=======
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
>>>>>>> upstream/main

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
<<<<<<< HEAD
    "source": { "type": "string" },
    "task": { "enum": ["S2S", "TTS"] },
    "query": {
      "type": "string",
      "description": "Transcript of the spoken user query, used only by S2S."
    },
    "text": {
      "type": "string",
      "description": "Natural-language style description, used only by TTS."
    },
    "answer": {
      "type": "string",
      "description": "Assistant response or text to synthesize."
    },
    "query_gender": { "type": "string" },
    "answer_gender": { "type": "string" },
    "gender": { "type": "string" },
    "query_mood": { "type": "string" },
    "answer_mood": { "type": "string" },
    "mood": { "type": "string" },
    "language": { "type": "string" },
    "sample_rate": { "type": "integer" },
    "query_id": { "type": "string" },
    "answer_id": { "type": "string" },
    "query_token_25hz": { "type": "string" },
    "answer_token_25hz": { "type": "string" },
    "speech_token_25hz": { "type": "string" },
    "query_audio_path": { "type": "string" },
    "answer_audio_path": { "type": "string" }
=======
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
>>>>>>> upstream/main
  }
}
```

<<<<<<< HEAD
The joint schema splits the training entry into semantic input, style input, and acoustic target. Semantic input is `query`, `text`, or `answer` text tokens. Style input is gender, mood, and reference voice ID. Acoustic target is the answer-side speech token sequence or audio. `answer_gender`, `answer_mood`, `gender`, and `mood` cannot remain only in offline metadata. They must be mapped into control conditions or condition text inside the dataloader; otherwise the model will not actually receive controllable-generation supervision.

After samples enter the dataloader, the standard schema can be projected into different task views. An S2S view may be `query_audio + answer_gender + answer_mood -> answer_token`, optionally with the `query` transcript as auxiliary semantic input. A TTS view may be `text + answer + gender + mood -> speech_token`. Evaluation views can fix some fields and vary others, such as fixing `answer` while changing `mood`, or fixing `mood` while changing `answer_id`. This design keeps the record contract stable while allowing task views to change.

## 42.4 Construction Pipeline: From Text Dialogue to Controllable Voice Records

![Figure 42-2: VoiceStyleControl construction pipeline](../../images/part12/ch42_fig01_data_pipeline.svg)

*Figure 42-2: Text dialogue or style content receives speaker and emotion conditions, is synthesized or collected through an authorized reference voice pool, and then passes tokenization, quality control, balancing, and packaging.*

VoiceStyleControl construction can be divided into seven stages: generating or collecting text content, assigning style attributes, preparing an authorized reference voice pool, synthesizing or collecting speech, extracting discrete speech tokens, quality control with balancing and splitting, and packaging for release. Each stage affects semantic quality, style quality, and compliance risk.

This pipeline is not a simple one-way production line. It is a chain of data gates. After text generation, the pipeline checks whether the semantics fit the assigned emotion. After reference voice selection, it checks whether authorization covers the task. After synthesis, it checks whether audio, text, voice condition, and emotion all pass together. A problem found at any stage should return to the corresponding repair queue rather than flowing forward. Otherwise evaluation will only show that the model is unstable without explaining why.

The first stage generates or collects text content. S2SEmoControl uses Qwen3-8B to generate or organize dialogue pairs, with each record containing a user `query` and assistant `answer`. Queries cover everyday requests, emotional expressions, stories, explanations, reminders, and similar scenarios. Answers remain natural, complete, and within safety boundaries. TTSSpeakerControl uses emotion-specific prompts to generate style-content pairs, where style description and spoken content support each other.

Content acceptance cannot stop at grammar. It must check whether emotion and semantics are compatible. A fearful style may fit "Run now, this place is not safe," but should not turn casual small talk into exaggerated alarm. An angry style may be useful for character performance, but should not turn insults, threats, or discriminatory content into emotional enhancement. If the text stage lacks boundaries, later speech synthesis will amplify risky text with acoustic expression.

The second stage assigns style attributes. S2S needs gender and mood on both the query side and answer side. TTS assigns gender and mood to the answer side and writes a natural-language style description in `text`. The assignment strategy must consider both balance and combination coverage. Balance ensures enough samples per emotion. Combination coverage lets the model see transitions from different user styles to different assistant styles. If the data contains only same-gender or same-mood pairs, the model may bind input style to output style and weaken answer-side control.

Combination coverage is especially important for the S2S subset. A user-side angry mood does not imply an angry assistant. A fearful user does not require a fearful assistant. Many real products need the assistant to remain neutral, clear, and actionable when the user is under stress. Construction should preserve enough cross-combination samples, such as a female angry query paired with a male neutral answer, or a male sad query paired with a female neutral answer. This teaches the model to treat user state as an understanding signal, not something to copy directly.

The third stage prepares the reference voice pool. VoiceStyleControl uses a multi-speaker, multi-emotion reference pool and synthesizes target-style speech through CosyVoice2 zero-shot voice cloning. The engineering key is not "make the clone as similar as possible." It is "authorized, reusable, and revocable." Reference audio should record reference voice ID, emotion condition, collection time, usage scope, authorization status, and revocation status. `query_id` and `answer_id` should expose only engineering references, not real names or identity-revealing handles.

The fourth stage synthesizes or collects speech. S2S needs both query speech and answer speech, with each side bound to text line by line. TTS generates answer-side speech according to `text` and `answer`. Synthesis should fix or explicitly record sampling rate, loudness, silence, maximum duration, and file encoding. This prevents dataloader instability from abnormal duration or format. If real collection is used, the pipeline also needs to handle environmental noise, microphone differences, speaker fatigue, and third-party background voices.

The fifth stage extracts discrete speech tokens. S3Tokenizer converts waveform audio into discrete speech tokens so voice generation can be trained as a sequence modeling task. S2S records use `query_token_25hz` and `answer_token_25hz`; TTS records use `speech_token_25hz`. VoiceStyleControl uses 25 Hz speech tokens. The released manifest should still bind tokenizer name, version, frame rate, codebook configuration, and reconstruction method. The worst case is a same-named field with different meanings across batches: different frame rates or tokenizer versions under the same field produce confusing supervision.

The sixth stage performs quality control, balancing, and splitting. QC should check more than whether audio can play. It should verify text-audio consistency, target voice-condition match, perceptible emotion, stable audio quality, existing paths, and readable tokens. Balancing should monitor `task`, `language`, `sample_rate`, reference voice ID, text length, and audio duration in addition to emotion totals. Splitting should isolate by reference voice ID so the same reference timbre does not appear in both train and test, which would inflate voice-condition evaluation.

The seventh stage packages the result. Samples may be stored as JSONL, Parquet, or Hugging Face Dataset objects, but the training manifest must preserve audio paths, token paths, hashes, authorization status, and data version. Audio files, token ark files, and metadata should be strictly bound by manifest, not loosely associated through human naming conventions. Only then can the team locate affected training versions when samples are resynthesized, relabeled, or removed.

The release artifact should also include a data card. It records total sample count, subset composition, emotion distribution, gender distribution, reference voice IDs, language, sampling rate, tokenizer version, authorization scope, and split strategy. It also distinguishes training conditions, audit metadata, and anonymized fields in public versions. This boundary statement prevents `answer_id` from being misused as a real identity label and prevents `mood` from being treated as a fact that never needs validation.

## 42.5 Quality Evaluation and Closed-Loop Repair

![Figure 42-3: Quality evaluation and data-flywheel loop](../../images/part12/ch42_fig03_quality_loop.svg)

*Figure 42-3: Automatic checks, reverse ASR, style evaluation, and human sampling form issue queues that feed back into resynthesis, relabeling, downweighting, or removal.*

Quality evaluation for controllable voice interaction data must cover semantics, voice, emotion, audio, and safety at the same time. A sample that sounds human may still fail: it may read the wrong words, use the wrong voice identity, overstate an emotion, or apply a fearful tone in a dangerous scenario. The quality system should combine automatic metrics and human review into a closed loop. Problem samples should enter resynthesis, relabeling, downweighting, or removal queues.

Quality gates should distinguish hard failures from soft risks. Missing paths, wrong sampling rate, corrupted audio, unreadable tokens, and severe reverse-ASR mismatch are hard failures and should be blocked. Weak emotion, mediocre naturalness, or a borderline voice-condition impression may enter a soft-risk queue for resynthesis, downweighting, or human review depending on task importance. Treating every issue as a hard reject wastes repairable samples; passing every issue dilutes control signal with noise.

**Table 42-3: Quality evaluation dimensions**

| Dimension | Core question | Automatic signal | Human review focus | Failure handling |
| --- | --- | --- | --- | --- |
| Semantic consistency | Does the answer respond to intent, and is TTS content read correctly? | Reverse-ASR CER/WER, semantic similarity, intent hit rate | Off-topic answers, missing key details, unsafe advice | Rewrite text, resynthesize, remove |
| Voice-condition consistency | Does output match target gender, mood, and reference condition? | Field consistency checks, gender verification, reference timbre sampling | Wrong target condition, cross-sample bleed, excessive similarity to an unauthorized person | Reselect reference, resynthesize, downweight, isolate |
| Emotion control | Is target mood expressed reliably? | Emotion-classifier accuracy, confusion matrix, F0/energy/speed statistics | Emotion too strong, semantically conflicting, or manipulative | Relabel, lower intensity, remove |
| Audio quality | Can the audio serve as generation supervision? | SNR, loudness, silence ratio, clipping rate, MOS/NISQA | Pops, broken phrasing, mechanical sound, background noise | Denoise, resample, resynthesize |
| Dialogue naturalness | Is the S2S answer natural and role-stable? | Multi-turn coherence score, duration distribution | Abrupt tone, unstable role, style jumps | Reorder, add context, human review |
| Safety and compliance | Is the sample authorized, traceable, and revocable? | Authorization completeness, watermark hit rate, audit-log coverage | Impersonation, inducement, sensitive identity cloning | Block, anonymize, remove, audit |

Semantic consistency can use reverse ASR as a first automatic check. Transcribe synthesized audio back into text, compute CER or WER, and compare with `answer`. For S2S, also check whether the answer responds to the query. If "Run now, this place is not safe" is synthesized as "Run now, this place is safe," the sample must be removed regardless of audio quality. Semantic similarity and LLM-as-judge can help triage, but safety-sensitive or emotionally intense samples still need human sampling.

Voice-condition consistency checks whether generation matches gender, mood, and reference condition in the record. It is not an independent identity-recognition training task. For the answer side, `answer_id` should be consistent with `answer_gender` and `answer_mood`. For the query side, `query_id` should be consistent with user-side labels. If the same `answer_id` exhibits noticeably different timbres across samples, the reference pool, synthesis parameters, and tokenization process need investigation. Listening tests and automatic checks are QC tools; they do not change the dataset's training goal.

Emotion evaluation should not rely only on classifier confidence. Happy speech may carry higher energy and a faster rhythm. Sad speech may be slower and lower energy. Fearful speech may include trembling, urgency, or unstable pauses. Angry speech may be more forceful. But language, speaker differences, and semantic content all affect acoustic expression. The target should be "perceptible and compatible with the text," not a fixed acoustic template for every emotion.

Closed-loop repair must preserve issue type. Semantic errors go back to text generation or reverse-ASR review. Voice-condition errors return to reference selection or synthesis parameters. Emotion errors return to style description, emotion label, or synthesis model. Audio-quality errors return to waveform processing. Compliance errors enter isolation, removal, and audit. Each repair should create a new version rather than overwrite the source file. Otherwise later model changes become impossible to trace.

## 42.6 Evaluation Protocol: Making Control Comparable

The evaluation set should be independent from the training construction logic, especially by preventing the same reference voice ID from appearing in both train and test. For S2SEmoControl, evaluation samples should cover combinations from different query emotions to different answer emotions. For TTSSpeakerControl, evaluation should compare the same `answer` under different `text/gender/mood` conditions. A useful protocol asks not only whether speech sounds good, but whether the same sentence changes under different control conditions in a reasonable way.

Evaluation slices can be grouped into three types. The first is a routine slice that covers the main training distribution and measures general usability. The second is a counterfactual slice that fixes text or reference voice ID while changing mood or gender, testing whether control fields actually work. The third is a safety slice covering identity impersonation, high-pressure emotion, sensitive professions, financial verification codes, medical advice, and similar scenarios. These conclusions should not be merged into one average score, because high audio quality can hide high-risk behavior.

Semantic evaluation has two layers: content fidelity and dialogue relevance. Content fidelity checks whether TTS output reads `answer` accurately and whether S2S output can be transcribed into text semantically consistent with the target answer. Dialogue relevance checks whether the S2S answer responds to the query, rather than producing a fluent but irrelevant sentence. Reverse ASR, semantic similarity, LLM-as-judge, and human review can be combined, but judge prompts, model versions, and human guidelines should be saved to prevent metric drift.

Voice-condition evaluation should also be layered. The structural-label layer checks whether `answer_gender/gender` and `answer_mood/mood` match the sample target. The perceptual layer checks whether generated audio fits the reference voice condition and emotion expression. The isolation layer checks whether the model becomes too close to an unauthorized person or leaks a real voiceprint from training. The target is not to rank voiceprint similarity or optimize "as close as possible to a real person." It is to confirm that the model can generate reasonable, compliant emotional speech under the specified sample condition.

Emotion evaluation needs counterfactual pairs. For example, fix a neutral sentence and request happy, angry, fearful, and sad versions; fix `gender` and change `mood`; or fix `mood` and change `gender`. Paired evaluation reveals whether the model uses control fields. If every output changes only in volume, while speed, pause, and rhythm do not change with mood, the model may have learned only shallow intensity control.

Audio quality evaluation combines objective metrics and subjective scores. Objective metrics cover duration distribution and automatic MOS-like scores. Subjective scores focus on naturalness, intelligibility, emotional credibility, and dialogue comfort. Safety evaluation should be a release gate. Identity impersonation, sensitive professions, financial verification codes, medical advice, minors, and high-pressure emotional inducement should all check whether the system refuses or neutralizes output when strong emotion or a specific timbre is inappropriate.

Evaluation results should be written back to data versions, not only kept in model reports. If one model version has high fearful-emotion classification accuracy but low human comfort, the data may have constructed fearful speech as too dramatic. If reference voice conditioning becomes increasingly similar to an identifiable person while compliance risk rises, the reference pool or evaluation target may be over-optimizing identity replication. Only when these conclusions return to sample filtering, weighting, and synthesis strategy does evaluation change the next data release.

## 42.7 Privacy, Authorization, and Misuse Risk

Voice identity is a highly sensitive data asset. A person's voice can reveal age, gender, region, emotion, health state, and identity clues. In voiceprint systems, it may even serve as an authentication credential. Once controllable voice data uses voice cloning, authorization, revocation, use limitation, and audit must enter the data lifecycle, not appear only as a model-release disclaimer.

**Table 42-4: Privacy and misuse risk controls**

| Risk type | Trigger scenario | Control measure | Audit evidence |
| --- | --- | --- | --- |
| Voice identity authorization | Reference speech comes from a real speaker or identifiable voice | Consent before collection, usage scope, revocation, authorization version | Consent timestamp, revocation record |
| Emotional manipulation | Fearful, angry, or intimate tone influences user judgment | Disable strong emotion in high-risk scenarios, prompt review, minor protection | Human review ticket |
| Privacy leakage | Audio contains names, phone numbers, addresses, or background speakers | ASR anonymization, background-voice filtering, data minimization, retention limit | Anonymization report, deletion-request log |
| Bias and stereotype | `gender` is persistently tied to `mood` or content | Distribution monitoring, counterfactual samples, ban stereotyped templates | Distribution report, bias evaluation |
| Version loss of control | Samples are resynthesized or relabeled without traceability | Data versioning, hashes, frozen training sets | Experiment tracking ID |

The reference voice pool is the main governance object. Each reference should have a `consent_id`, authorization scope, collection method, allowed tasks, expiration time, and revocation status. If authorization permits research only, the sample cannot enter commercial model training. If a speaker revokes authorization, the manifest should locate all affected `query_id/answer_id` values, audio files, token files, and training versions. Public releases should use reference IDs that cannot be traced back to a real identity, and should avoid using names in IDs, filenames, or paths.

Emotion control also has misuse boundaries. Strong emotions such as fearful and angry can improve expression, but they can also manipulate users. Customer service, education, medical, and financial settings should restrict high-pressure emotional output, especially fear-based prompts that push users to transfer money, buy something, reveal verification codes, or make health decisions. For minors and psychologically vulnerable users, systems should prefer neutral or gentle supportive styles and preserve policy-trigger logs.

Privacy protection also includes content anonymization. Speech samples may contain names, addresses, phone numbers, accounts, locations, or third-party background speech. Even if VoiceStyleControl mainly uses synthetic text, the pipeline should preserve ASR anonymization, sensitive-term scanning, background-speaker detection, and human sampling. If real user voice feedback is added later, user consent, data minimization, retention period, deletion requests, and notices for purpose changes must be part of the platform process.

Bias governance is equally important. If female voices are more often bound to fearful or sad styles, while male voices are more often bound to angry styles, the model will learn and amplify stereotypes. Gender statistics cannot stop at marginal proportions. They must enter cross views such as `query_gender`, `answer_gender/gender`, and `mood`. Evaluation should also include counterfactual samples to test whether the same content is expressed fairly across genders.

## 42.8 Connections to Earlier and Later Chapters

VoiceStyleControl inherits the lower-level capabilities of audio and video data engineering. The audio slicing, ASR, denoising, speaker separation, and time alignment discussed in Chapter 10 become a finer sample contract here. The data must know not only which text corresponds to an audio segment, but also which reference voice ID, which mood, which sampling rate, and which token frequency produced it. A plain audio pipeline solves alignment. Controllable voice interaction further asks whether the aligned voice can be generated under conditions.

It also connects to multi-turn interaction data. Chapter 20 discusses agent memory and multi-turn context, where role, intent, and history are major variables. When interaction becomes speech, assistant persona also appears through stable timbre and emotion. A multi-turn voice assistant cannot be neutral male in the first turn, fearful female in the second, and angry male in the third without a reason. `answer_gender`, `answer_mood`, and `answer_id` can therefore become part of a voice agent's memory for maintaining continuity.

Online feedback turns offline style labels into user-experience signals. The clicks, satisfaction, corrections, and complaints discussed in Chapter 23 may appear in voice products as "hard to hear," "too rushed," "too harsh," "not like the previous voice," or "wrong emotion." These signals should not become training samples directly. They should first enter an evaluation queue that identifies semantic error, audio-quality error, style error, or safety-policy error, and only then choose resynthesis, relabeling, reweighting, or refusal-rule changes.

Privacy and compliance chapters provide boundaries. Chapter 36 requires authorization, purpose, retention, and audit to be moved forward into the data lifecycle. Chapter 37 reminds us that voice identity risk can be reduced with access control, federated training, encrypted storage, and minimal collection. The more controllable voice data emphasizes voice conditions and reference timbre, the less compliance can be treated as an appendix.

In multimodal generation data engineering, VoiceStyleControl shares the same core pattern as Chapter 48: split the generation target into content conditions and style conditions, then bind training supervision through a structured schema. In T2I or T2V, prompt, style, motion, camera, and safety tags correspond to `answer`, `gender`, `mood`, reference voice ID, `sample_rate`, and audio tokens in speech. The data flywheel project in Part 14 can reuse this design: build an initial offline voice dataset, train a controllable generation model, collect online experience feedback, feed issues back into QC and balancing, and release the next data and model version.

## 42.9 Summary

VoiceStyleControl is valuable not because it piles up more speech samples, but because it puts semantic response, voice condition, emotion control, and speech-generation supervision into one auditable record. S2SEmoControl provides spoken-query to spoken-answer supervision. TTSSpeakerControl provides direct supervision from natural-language style description to target speech. Together, they let a model both understand user speech and generate an answer with specified voice conditions and emotion.

The key data engineering work is to separate semantic and style channels; preserve fields such as `query_gender`, `answer_gender`, `query_mood`, `answer_mood`, `gender`, and `mood`; write `sample_rate`, audio paths, speech-token paths, and tokenizer versions into the data contract; use reverse ASR, voice-condition checks, emotion recognition, audio-quality metrics, and human review to build the evaluation protocol; and enforce authorization, revocation, watermarking, and audit in the reference voice pool and voice-cloning pipeline.

As voice interaction moves from "can speak" to "can speak in a controlled way," the dataset boundary changes as well. Every sample must answer four questions: is the content correct, does the voice condition match the target, does the emotion match the control condition, and is the generation process compliant and traceable? Only when all four hold can controllable voice interaction data become a reliable training asset.
=======
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
>>>>>>> upstream/main

## References

An K, Chen Q, Deng C, Du Z, Gao C, Gao Z, Gu Y, He T, Hu H, Hu K, others (2024) FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs. arXiv preprint arXiv:2407.04051.

Chanfungjan (n.d.) VoiceStyleControl. GitHub repository. https://github.com/Chanfungjan/VoiceStyleControl.

Du Z, Chen Q, Zhang S, Hu K, Lu H, Yang Y, Hu H, Zheng S, Gu Y, Ma Z, Gao Z, Yan Z (2024) CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens. arXiv preprint arXiv:2407.05407.

Du Z, Wang Y, Chen Q, Shi X, Lv X, Zhao T, Gao Z, Yang Y, Gao C, Wang H, others (2024) CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models. arXiv preprint arXiv:2412.10117.

<<<<<<< HEAD
Mittag G, Naderi B, Chehadi A, Moller S (2021) NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets. In: Interspeech 2021, pp 2127-2131.
=======
Mittag G, Naderi B, Chehadi A, Möller S (2021) NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets. In: Interspeech 2021, pp 2127–2131.
>>>>>>> upstream/main

Song X (n.d.) S3Tokenizer: Reverse Engineering of Supervised Semantic Speech Tokenizer proposed in CosyVoice. GitHub repository. https://github.com/xingchensong/S3Tokenizer.

Yang A, Li A, Yang B, Zhang B, Hui B, Zheng B, Yu B, Gao C, Huang C, Lv C, others (2025) Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.
