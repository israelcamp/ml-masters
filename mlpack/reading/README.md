## Papers

- BART
  - [Link](https://arxiv.org/pdf/1910.13461.pdf)
  - Model similar to BERT with the differences
    - GeLUs as activation functions
    - Encoder - Decoder for text generation
    - Cross attention between every decoder layer to the last encoder layer
    - No feedforward NNs
  - Model is pre trained to denoise a noised document
  - Pre-training differences
    - Token Deletion
      - A token is deleted from the original text, the model must output the deleted token
    - Text Infilling
      - A span of text is replaced completely by the mask token
    - Sentence Permutation
      - The text is divided by full stop words and the sequences are shuffled
    - Document Rotation
      - A token is randomly chosen and the text is rotated so that this token is placed as the start of the document
  - The authors point that bidirectional encoder are crucial for good performance on question and answering tasks.
- T5
  - [Link](https://arxiv.org/pdf/1910.10683.pdf)
  - Baseline model
    - Similar to BERT base
    - Encoder-decoder
    - Pre trained models do not improve much the performance on translation tasks
  - Pre-trained with Text Infiling
- XLNet
  - [Link](https://arxiv.org/pdf/1906.08237.pdf)
- GeLUs
  - [Link](https://arxiv.org/abs/1606.08415)
- Are Sixteen Heads Really Better than One?
  - [Link](https://arxiv.org/pdf/1905.10650.pdf)
  - Looks like not all the attention head in BERT are needed, as the model sometimes even performs better with less heads, but the algorithm involved in pruning is expensive and only 20% of the head are not useful.
  - The speed-up is around 17%, also not much.
- Zero-Shot Entity Linking by Reading Entity Descriptions
  - [Link](https://arxiv.org/pdf/1906.07348.pdf)
  - Linking entities on a text
    - Given a text and a dictionary of entities, link the entities mentions on the text to the corresponding entities on the dictionary.
    - The problem is that is assumes this dictionary of entities for a given text.
- How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?
  - [Link](https://www.aclweb.org/anthology/K19-1043.pdf)
  - Words with the same gender tend to have closer embeddings than synonyms.
    - For example the word _viaggio (journey) \_in italian is masculine and the word \_gita (trip) \_feminin, although similar the top-10 nearest neighbours of the word _ gita \_are also feminin.
  - “We find that such methods are effective in reducing the effect, but are also language specific and tricky to get right”.
  - There are two options to solve this problem before training
    - **Lemmatization**: use the lemmatization on the context words
      - Better in German
    - **Gender change: **change all the context words to the same gender
      - Better in Italian
      - _“As bibliotecas fecharam as portas pelos donos”_ would become _“Os bibliotecas fecharam os portas pelos donos”_ if we could change all gender context words to masculine.
  - A good analysis must be made for every language, the portuguese case is probably similar to the Italian case, however some of the problems related to the Italian language would not occur in portuguese (I think)
    - _“In some cases, a single word might have multiple forms in the opposite gender. For example, the Italian “delle” is the feminine form of both “dei” and “degli”, depending on the phonetic context”_
  - **_Is this also a problem with BERT embeddings? Would this method improve performance?_**
- Convolutional Self Attention Networks
  - [Link](https://arxiv.org/pdf/1904.03107.pdf)
- Attention Augmented Convolutional Networks
  - [Link](https://arxiv.org/pdf/1904.09925.pdf)

## Datasets

- BioCreative III
  - Entity normalization
  - Given an article output the list of gene ids mentioned
  - State of Art
    - GNormPlus
      - [https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/)
      - [http://downloads.hindawi.com/journals/bmri/2015/918710.pdf](http://downloads.hindawi.com/journals/bmri/2015/918710.pdf)
      - From what I understand this solution was trained on a NER dataset and applied to the BioCreative III dataset.

## Articles

- Encoder-decoders in Transformers: a hybrid pre-trained architecture for seq2seq
  - [Link](https://medium.com/huggingface/encoder-decoders-in-transformers-a-hybrid-pre-trained-architecture-for-seq2seq-af4d7bf14bb8)
- The Illustrated Transformer
  - [Link](https://jalammar.github.io/illustrated-transformer/)
- All the ways you can compress BERT
  - [Link](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter)
  - Some methods to make BERT smaller and faster
  - Some methods look good since they make a smaller model with the same performance
  - Some are actually models, like [DistillBERT](https://arxiv.org/abs/1910.01108) from Huggingface, this model seems good for real life applications, but of course it does not come pre trained in portuguese.

## Misc

- AI Dungeon
  - [Link](https://www.aidungeon.io/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter)

## Things to Check

- NBDev
  - [Link](https://nbdev.fast.ai/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter#Installing)
  - [https://www.fast.ai/2019/12/02/nbdev/](https://www.fast.ai/2019/12/02/nbdev/)
  - Looks cool, but it sounds like it relies too much on Github
- Microsoft NLP Resources
  - [Link](https://github.com/microsoft/nlp-recipes)
  - Contents about State of Art NLP, but relies too much on Azure.
