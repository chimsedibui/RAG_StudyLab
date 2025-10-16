import os
import wget

file_links = [
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762"
    },
    {
        "title": "Adding Conditional Control to Text-to-Image Diffusion Models",
        "url": "https://arxiv.org/pdf/2302.05543"
    },
    {
        "title": "BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/pdf/1810.04805"
    },
    {
        "title": "Layer Normalization",
        "url": "https://arxiv.org/pdf/1607.06450"
    },
    {
        "title": "1D Convolutional Neural Networks and Applications: A Survey",
        "url": "https://arxiv.org/abs/1905.03554"
    },
    {
        "title": "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling",
        "url": "https://arxiv.org/pdf/1412.3555"
    },
    {
        "title": "A Survey on the Transformer Model and Its Generalizations in Artificial Intelligence",
        "url": "https://arxiv.org/pdf/2301.00249"
    }, 
    {
        "title": "Instruction Tuning for Large Language Models- A Survey",
        "url": "https://arxiv.org/pdf/2308.10792"
    },
    {
        "title": "Llama 2- Open Foundation and Fine-Tuned Chat Models",
        "url": "https://arxiv.org/pdf/2307.09288"
    },
    {
        "title": "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
        "url": "https://arxiv.org/pdf/1703.10593"
    },
    {
        "title": "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS",
        "url": "https://arxiv.org/pdf/1511.06434"
    }
]

def is_exist(file_link):
    return os.path.exists(f"./{file_link['title']}.pdf")

for file_link in file_links:
    if not is_exist(file_link):
        wget.download(file_link["url"], out=f"./{file_link['title']}.pdf")
