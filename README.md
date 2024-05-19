# Prompt Engineering Essentials

## Prompt Engineering

- When interacting with a Foundation Model, a Prompt is the Input that the Practitioner provide to the Model to generate a Response or Output. In essence, Prompts are Instructions for what the Practitioner want the Model to do.
- The Quality and Structure of the Prompt can significantly influence the Foundation Model's Performance on a given Task. This is where _Prompt Engineering_ becomes crucial.
- _Prompt Engineering_ is a new and essential Field focused on Optimizing the use, development, and understanding of Language Models, particularly large Models. At its Core, _Prompt Engineering_ involves designing Prompts and Interactions to:
  - Expand the Capabilities of Language Technologies
  - Address and mitigate their Weaknesses
  - Gain deeper Insights into their Functioning
- _Prompt Engineering_ equips Practitioners with Strategies and Techniques for pushing the Boundaries of what is possible with Language Models. It aims to:
- **Designing Effective Prompts**: Crafting Prompts that clearly and accurately convey the Task to the Model
- **Optimizing Interactions**: Tweaking and refining Prompts to improve the Model's Performance on specific Tasks
- **Understanding Model Behavior**: Gaining Insights into how Models interpret and respond to different Prompts, which can inform further Development and Application of these Technologies

### Difference Between Prompt Engineering and Fine-Tuning

- In Fine-tuning, the Weights or Parameters of the Model are adjusted using Training Data with the Goal of Optimizing a Cost Function. Fine-tuning can be an expensive Process, both in Terms of Computation Time and actual Cost.
- _Prompt Engineering_ attempts to guide the trained Foundation Model, an LLM, or a Text-to-Image Model, to give more relevant and accurate Answers.

### Benefits of Prompt Engineering

- _Prompt Engineering_ is the fastest Way to harness the Power of LLMs. By interacting with an LLM through a Series of Questions, Statements, or Instructions, the Practitioner can adjust LLM Output Behavior based on the specific Context of the Output they want to achieve.
- Effective Prompt Techniques can help Businesses accomplish the following Benefits:
  - **Boost a Model's Abilities and improve Safety**: Enhance the Performance and Reliability of the Model's Outputs.
  - **Augment the Model with Domain Knowledge and external Tools**: Incorporate additional Information and Tools without changing Model Parameters or Fine-tuning.
- **Interact with Language Models to grasp their full Capabilities**: Explore and utilize the Model's full Range of Functionalities.
- **Achieve better quality Outputs through better quality Inputs**: Ensure higher Accuracy and Relevance in the generated Outputs by optimizing the Inputs.

### Elements of a Prompt

- The Form of a Prompt depends on the Task that the Practitioner gives a Model.
- As Practitioners explore Examples of Prompt Design, they will see Prompts that contain some or all of the following Elements:
  - **Instructions**: This is a Task for the LLM to do. It provides a Task Description or Instruction for how the Model should perform.
  - **Context**: This is external Information to guide the Model.
  - **Input Data**: This is the Input for which the Practitioner want a Response.
  - **Output Indicator**: This is the Output Type or Format.

### Best Practices for designing effective Prompts

| Recommendation                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Bad Prompt                                                                  | Good Prompt                                                                                                                                                                                          |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Be clear and concise                             | Prompts should be straightforward and avoid Ambiguity. Clear Prompts lead to more coherent Responses. Craft Prompts with natural, flowing Language and coherent Sentence Structure. Avoid isolated Keywords and Phrases.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Compute the Sum Total of the subsequent Sequence of Numerals: 4, 8, 12, 16. | What is the Sum of these Numbers: 4, 8, 12, 16?                                                                                                                                                      |
| Include Context if needed                        | Provide any additional Context that would help the Model respond accurately. For Example, if the Practitioner ask a Model to analyze a Business, include Information about the Type of Business. What does the Company do? This Type of Detail in the Input provides more relevant Output. The Context the Practitioner provide can be common across multiple Inputs or specific to each Input.                                                                                                                                                                                                                                                                                                                                                                                                                              | Summarize this Article: [insert Article Text]                               | Provide a Summary of this Article to be used in a Blog Post: [insert Article Text]                                                                                                                   |
| Use Directives for the appropriate Response Type | If hte Practitioner want a particular Output Form, such as a Summary, Question, or Poem, specify the Response Type directly. The Practitioner can also limit Responses by Length, Format, included Information, excluded Information, and more.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | What is the Capital?                                                        | What is the Capital of New York? Provide the Answer in a full Sentence.                                                                                                                              |
| Consider the Output in the Prompt                | Mention the requested Output at the End of the prompt to keep the Model focused on appropriate Content.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Calculate the Area of a Circle.                                             | Calculate the Area of a Circle with a Radius of 3 Inches (7.5 cm). Round your answer to the nearest Integer.                                                                                         |
| Start Prompts with an Interrogation              | Phrase your Input as a Question, beginning with Words, such as who, what, where, when, why, and how.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Summarize this Event.                                                       | Why did this Event happen? Explain in three Sentences.                                                                                                                                               |
| Provide an Example Response                      | Use the expected Output Format as an Example Response in the Prompt. Surround it in Brackets to make it clear that it is an Example.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Determine the Sentiment of this Social Media Post: [insert Post]            | Determine the Sentiment of the following Social Media Post using these Examples: post: "great Pen" => Positive; post: "I hate when my phone Battery dies" => Negative; [insert Social Media Post] => |
| Break up complex Tasks                           | Foundation Models can get confused when asked to perform complex Tasks. Break up complex Tasks by using the following Techniques: Divide the Task into several Subtasks. If the Practitioner can not get reliable Results, try splitting the Task into multiple Prompts. Ask the Model if it understood the Instruction. Provide Clarification based on the Model's Response. If the Practitioners do not know how to break the Task into Subtasks, ask the Model to think Step by Step. This Method might not work for all Models, but the Practitioners can try to rephrase the Instructions in a Way that makes Sense for the Task. For Example, the Practitioners might request that the Model divides the Task into Subtasks, approaches the Problem systematically, or reasons through the Problem one Step at a Time. | -                                                                           | -                                                                                                                                                                                                    |
| Experiment and be creative                       | Try different Prompts to optimize the Model's Responses. Determine which Prompts achieve effective Results and which Prompts achieve inaccurate Results. Novel and thought-provoking Prompts can lead to innovative Outcomes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | -                                                                           | -                                                                                                                                                                                                    |

### Advanced Prompt Engineering Techniques

- When crafting and manipulating Prompts, there are certain Techniques that Practitioners can use to achieve the Response they want from Models.

#### Zero-Shot Prompting

- Zero-shot Prompting is a Technique where a Practitioner presents a Task to an LLM without giving the Model further Examples. Here, the Practitioner expects the model to perform the Task without a prior Understanding, or Shot, of the Task.
- Modern LLMs demonstrate remarkable Zero-shot Performance.

##### Tips for Zero-Shot Prompting

- The larger the LLM, the more likely the Zero-shot Prompt will yield effective Results.
- Instruction Tuning can improve Zero-shot Learning. To better adapt LLMs to Human Preferences, Reinforcement Learning from Human Feedback (RLHF) can be used to scale Instruction Tuning.

#### Few-Shot Prompting

- Few-shot Prompting is a Technique where the Practitioner give the Model contextual Information about the requested Tasks. In this Technique, Examples of both the Task and the Output are provided.
- Providing this Context, or a Few Shots, in the Prompt conditions the Model to follow the Task guidance closely.

##### Tips for Few-Shot Prompting

- The Labels in a Few-shot Prompt do not need to be correct to improve Model Performance. Applying random Labels often outperforms using no Labels at all. However, the Label Space and Distribution of the Input Text specified by the Demonstrations are important.
- If the Practitioners have access to a large Set of Examples, they should use Techniques to obey the Token Limits of the Model and dynamically populate Prompt Templates.

#### Chain-of-Thought Prompting

- Chain-of-Thought (CoT) Prompting breaks down complex Reasoning Tasks through intermediary Reasoning Steps.
- Practitioners can use both Zero-shot and Few-shot Prompting Techniques with CoT Prompts.
- Practitioners can use the Phrase "Think Step by Step" to invoke CoT Reasoning from the Model.

##### Tip for Chain-of-Thought Prompting

- Use CoT Prompting when the Task involves several Steps or requires a Series of Reasoning.

#### Self-Consistency

- Self-consistency is a Prompting Technique similar to CoT Prompting. However, instead of taking the obvious Step-by-Step or Greedy Path, Self-Consistency Prompts the Model to sample a variety of Reasoning Paths. Then, the Model aggregates the final Answer based on multiple Data Points from the various Paths.
- Self-consistency improves CoT Reasoning Prompting when used in a Range of common arithmetic and common-sense Reasoning Benchmarks.
- Using the Self-consistency Technique, the Model can separate the appropriate Data Points and then aggregate them into the correct Answer.

#### Tree of Thoughts

- Tree of Thoughts (ToT) is Technique that builds on the CoT Prompting Technique. CoT Prompting samples Thoughts sequentially, but ToT Prompting follows a Tree-branching Technique.
- With the ToT Technique, the LLM can learn in a nuanced Way, considering multiple Paths instead of one sequential Path.
- ToT Prompting is an especially effective Method for Tasks that involve important initial Decisions, Strategies for the Future, and Exploration of multiple Solutions.
- Most LLMs make Decisions by following a standard Left-to-Right Token-Level Inference, but with ToT, LLMs can self-evaluate Choices.
- ToT significantly improves Performance on Tasks that require nontrivial Planning.

#### Retrieval Augmented Generation

- Retrieval Augmented Generation (RAG) is a Prompting Technique that supplies domain-relevant Data as Context to produce Responses based on that Data and the Prompt.
- RAG is similar to Fine-tuning. However, rather than having to fine-tune an FM with a small Set of labeled Examples, RAG retrieves a small Set of relevant Documents from a large Corpus and uses that to provide Context to answer the Question.
- RAG will not change the Weights of the FM whereas Fine-tuning will. This Approach can be more cost-efficient than regular Fine-tuning because the RAG approach does not incur the Cost of fine-tuning a Model.
- RAG also addresses the Challenge of frequent Data Changes because it retrieves updated and relevant Information instead of relying on potentially outdated Sets of Data.
- In RAG, the external Data can come from multiple Data Sources, such as a Document Repository, Databases, or APIs, but before it can be used within LLMs, the Data must be prepared and kept currrent.

#### Automatic Reasoning and Tool-Use

- Like the Self-consistency and ToT Prompt Techniques, ART is a Prompting Technique that builds on the CoT Process.
- ART is used specifically for multi-step Reasoning Tasks. This Technique essentially deconstructs complex Tasks by having the Model select Demonstrations of multiple, or few-shot, Examples from the Task Öibrary. At the same Time, the Model is using this Few-shot Breakdown, it uses predefined external Tools such as Search and Code Generation to carry out the Task.
- ART performs substantially better than Few-shot Prompting and automatic CoT for unseen Tasks and matches the Performance of handcrafted CoT Prompts on a majority of Tasks.
- ART also makes it more efficient for Humans to update Information in the Task Libraries, which can correct Errors and ultimately improve Performance.

#### ReAct Prompting

- With ReAct Prompting, an LLM can combine Reasoning and Action. Models are often used for Reasoning or for Acting, but they are not always effectively used for both at the same Time.
- CoT Prompting shows Promise for an LLM to reason and generate Actions for straightforward Tasks, like Mathematics. However, the Inability to update Information or access external Contexts with CoT Prompting can lead to Errors in the Output, such as Fact Hallucination.
- With a ReAct Framework, LLMs can generate both Reasoning Traces and Task-specific Actions that are based on external Tools, such as Wikipedia Pages or SQL Databases. This external Context leads to more accurate and reliable Output.

## Generative AI and Foundation Models

- Generative AI is a Type of Artificial Intelligence that can create new Content and Ideas, including Conversations, Stories, Images, Videos, and Music.
- Like all other AI, generative AI is powered by Machine Learning (ML) Models
- Generative AI is powered by very large Models, commonly called Foundation Models. Foundation Models are pre-trained on a vast Corpus of Data, usually through self-supervised Learning.

## Understanding FM Functionality

- The Size and general-purpose Nature of Foundation Models make them different from traditional ML Models. Foundation Models use Deep Neural Networks to emulate Human Brain Functionality and handle complex Tasks. Foundation Models can be adapted for a broad Range of general Tasks, such as Text Generation, Text Summarization, Information Extraction, Image Generation, Chatbot, and Question Answering. Foundation Models can also serve as the Starting Point for developing more specialized Models.

<p align="center">
  <img src="images/understanding_fm.png" alt="Shared Responsibility" width="75%"/>
</P>

### Self-Supervised Learning

- Although traditional ML Models rely on supervised, semi-supervised, or unsupervised Learning Patterns
- Foundation Models are typically pre-trained through self-supervised Learning.
- With self-supervised Learning, labeled Examples are not required. Self-supervised Learning makes Use of the Structure within the Data to autogenerate Labels.

### Training, Fine-Tuning, and Prompt Engineering

- Foundation Models go through various Stages of Training to achieve the best Results:
  1. **Pre-training**: During the Training Stage, Foundation Models use self-supervised Learning or Reinforcement Learning from Human Feedback (RLHF) to capture Data from vast Datasets. The Foundation Model's Algorithm can learn the Meaning, Context, and Relationship of the Words in the Datasets. In addition, RLHF can be used during Pre-training to better align the Model with Human Preferences. In this Approach, Humans provide Feedback on the Model Outcomes, and that Information is used by the Model to change its Behavior.
  2. **Fine-tuning**: Though Foundation Models are pre-trained through self-supervised Learning and have the inherent Capability of understanding Information, Fine-tuning the Foundation Model Base Model can improve Performance. Fine-tuning is a supervised Learning Process that involves taking a pre-trained Model and adding specific, smaller Datasets. Adding these narrower Datasets modifies the Weights of the Data to better align with the Task. There are two ways to fine-tune a Model:
     1. **Instruction Fine-Tuning**: Uses Examples of how the Model should respond to a specific Instruction. _Prompt tuning_ is a Type of Instruction Fine-tuning.
     2. **RLHF**: Provides Human Feedback Data, resulting in a Model that is better aligned with Human Preferences.
  3. **Prompt Engineering**: Prompts act as Instructions for Foundation Models. They are similar to Fine-tuning, but the Practitioner does not need to provide labeled Sample Data as they would to fine-tune a Model. Practitioners use various Prompt Techniques to achieve better Performance. _Prompt Engineering_ is a more efficient Way to tune (large Language Model) LLM Responses, as opposed to Fine-tuning, which requires labeled Data and Training Infrastructure.

### Categories of Foundation Models

- Foundation Models can be categorized into multiple Types. Two of the most frequently used Models are _Text-to-Text Models_ and _Text-to-Image Models_.
-

#### Text-to-Text Models

- Text-to-text Models are LLMs pre-trained to process vast Quantities of textual Data and Human Language.

##### Natural Language Processing (NLP)

- NLP is a Machine Learning Technology that gives Machines the Ability to interpret and manipulate Human Language. NLP does this by analyzing the Data, Intent, or Sentiment in the Message and responding to Human Communication.
- Typically, NLP Implementation begins by gathering and preparing unstructured Text or Speech Data from different Sources and processing the Data.
- It uses Techniques such as Tokenization, Stemming, Lemmatization, Stop Word Removal, Part-of-Speech Tagging, Named Entity Recognition, Speech Recognition, Sentiment Analysis, and so on. However, modern LLMs do not require using these intermediate Steps.

##### Recurrent Neural Network (RNN)

- RNNs use a Memory Mechanism to store and apply Data from previous Inputs. This Mechanism makes RNNs effective for sequential Data and Tasks, such as Natural Language Processing, Speech Recognition, or Machine Translation.
- RNNs are slow and complex to train, and they ca not be used for Training Parallelization.

##### Transformer

- A Transformer is a Deep-Learning Architecture that has an Encoder Component that converts the Input Text into Embeddings. It also has a Decoder Component that consumes the Embeddings to emit some Output Text.
- Unlike RNNs, Transformers are extremely parallelizable, which means instead of processing Text Words one at a Time during the Learning Cycle, Transformers process Input all at the same Time. It takes Transformers significantly less Time to train, but they require more Computing Power to speed Training.
- The Transformer Architecture was the Key to the Development of LLMs. These Days, most LLMs only contain a Decoder Component.

#### Text-to-Image Models

- Text-to-Image Models take natural Language Input and produce a high-quality Image that matches the Input Text Description.

##### Diffusion Architecture

- Diffusion is a deep-learning Architecture System that learns through a two-step Process.
  1. The first Step is called Forward Diffusion. Using Forward Diffusion, the System gradually introduces a small Amount of Noise to an Input Image until only the Noise is left over. There is a U-Net Model that tracks and predicts the Noise Level.
  2. In the subsequent Reverse Diffusion Step, the noisy Image is gradually denoised until a new Image is generated. During the Training Process, the Model gets the Feed of Text, which is added to the Image Vector.

## Large Language Models (LLMs)

- LLMs are a Subset of Foundation Models and are trained on vast Text Corpus across many natural Language Tasks.
- They can understand, learn, and generate Text that is nearly indistinguishable from Text produced by Humans.
- LLMs can also engage in interactive Conversations, answer Questions, summarize Dialogues and Documents, and provide Recommendations.
- Because of their sheer Size and AI Acceleration, LLMs can process vast Amounts of textual Data.

### Understanding LLM Functionality

- Most LLMs are based on a Transformer Model. They receive Input, encode the Data, and then decode the Data to produce an Output Prediction.

#### Neural Network Layers

- Transformer Models are effective for Natural Language Processing because they use Neural Networks to understand the Nuances of Human Language.
- Neural Networks are Computing Systems modeled after the Human Brain.
- There are multiple Layers of Neural Networks in a single LLM that work together to process Input and generate Output. The following are three Key Categories of Neural Network Layers:
  - **Embedding Layer**: The Embedding Layer converts Input Text to Vector Representations called Embeddings. This Layer can capture complex Relationships between the Embeddings, so the Model can understand the Context of the Input Text.
  - **Feedforward Layer**: The Feedforward Layer consists of several connected Layers that transform the Embeddings into more weighted Versions of themselves. Essentially, this Layer continues to contextualize the Language and helps the Model better understand the Input Text's Intent.
  - **Attention Mechanism**: With the Attention Mechanism, the Model can focus on the most relevant Parts of the Input Text. This Mechanism, a central Part of the Transformer Model, helps the Model achieve the most accurate Output Results.
