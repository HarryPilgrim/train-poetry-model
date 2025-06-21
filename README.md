As part of my machine learning studies, I am fine-tuning a small language model (DistilGPT-2) using Hugging Face’s transformers library. The goal: generate original poetry based on the styles of poets.

![hugging face model](https://github.com/user-attachments/assets/19899951-5fd9-4aae-a3d9-e78055822dd4)

Trained a custom DistilGPT2 model using PyTorch and Hugging Face Transformers on a corpus of poetry. Fine-tuned the model locally, then pushed it to the Hugging Face Model Hub for inference. Integrated the hosted model into a Flask-based poetry website (poetryplatform.org) via Hugging Face’s Inference API, enabling users to generate AI-composed poems from text prompts directly on the site.

![poetry platform ai](https://github.com/user-attachments/assets/d14cbb9a-511b-4fff-878d-45db309e00f6)

Transformers work by learning the probability of a word given the context of previous words—ideal for text generation. By training on a dataset of real poems, the model begins to capture stylistic elements like tone, rhythm, and phrasing.

Here’s an eerie early sample from the model. the model that produced this was trained on 20 poets. It starts from a short prompt and completes the thought in its own strange, learned voice:


Generated Poem:
 Write a poem about the last sunset on Earth.
 
 With a poem of an echo-chamber.
So we must ask the question: if we're going to be a poet?
"What kind of poems do I write," the poet replied, "as if I were a wiper's flower, and not another flower? What kind of poems do I write?"...
By the dawn, I could not believe it.
And when I made this verse, then, I was born in the cold.
And why did we dream of something?
And how did I write it?
And how did I write it?
And then?
And why did I write it?
And why didn't I write it?
And why didn't we dream of something?
And why didn't we dream of something?
And why didn't we dream of an unbroken sun?
And why didn't we dream of something?
And why didn't we dream of something?
And why didn't we dream of something?
And why didn't we dream of something?
And why didn't we dream of something?
And why didn't we dream of something?
And why didn't we dream of something?
And why didn't we dream of something?
