# Retriever: Retrieval-based Data Mining and Conditioning for Zero-shot Image Captioning

A second place winning solution to the 2023 Challenge on zero-shot image captioning evaluation at [CVPR 2023 NICE Workshop](https://nice.lgresearch.ai/).


> Retriever: Retrieval-based Data Mining and Conditioning for Zero-shot Image Captioning
> 
> [Youngtaek Oh](https://ytaek-oh.github.io), [Jae Won Cho](https://chojw.github.io/), [Dong-Jin Kim](https://sites.google.com/site/djkimcv/), [In So Kweon](http://rcv.kaist.ac.kr/index.php?mid=rcv_faculty), and [Junmo Kim](https://siit.kaist.ac.kr/Faculty)


## Abstract
This report introduces Retriever, our solution for the 2023 Challenge on zero-shot image captioning evaluation at the New Frontiers for Zero-Shot Image Captioning Evaluation (NICE) Workshop. Retriever efficiently improves image captioners by retrieving from an external memory of image-text pairs in two steps. First, a set of image-text pairs for training is fetched by applying explicit retrieval module to the intended target dataset. In addition, we fuse the knowledge associated with the input sample queried from the retrieval module during training and inference. With this complete framework, specific knowledge in captions can be easily incorporated into the captioner even in the absence of ground-truth captions, and the model can generate better captions conditioned on relevant knowledge from an external data source. Experimentally, Retriever improves the base image captioner by the CIDEr score by 229.4 in (held-out) validation data of NICE Challenge 2023 despite its simplicity. On the test data, notably, we ranked 2nd in CIDEr score, and 1st in all the other metrics.

<p align='center'>
  <img width='100%' src='./assets/figure1.png'/>
</p>


## TODO

- [x] Upload codes
- [ ] Documentation
- [ ] Share technical report
- [ ] Writing bash scripts for training and evaluation
- [ ] Share FAISS index file, checkpoint, and logs
