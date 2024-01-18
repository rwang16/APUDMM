# APU-DMM (Adaptive Pólya Urn Dirichlet Multinomial Mixture)
APU-DMM is an open-source Java package implementing the algorithm proposed in the paper (Li, M.J. et al., ICONIP 2023), created by Rui Wang. For more details, please refer to [this paper](https://link.springer.com/chapter/10.1007/978-981-99-8181-6_28).

If you use this package, please cite the paper: **Li, M.J. et al. (2024). Topic Modeling for Short Texts via Adaptive Pólya Urn Dirichlet Multinomial Mixture. In: Luo, B., Cheng, L., Wu, ZG., Li, H., Li, C. (eds) Neural Information Processing. ICONIP 2023. Communications in Computer and Information Science, vol 1968. Springer, Singapore. https://doi.org/10.1007/978-981-99-8181-6_28**

If you have any questions or bug reports, please contact Rui Wang (2210273056@email.szu.edu.cn).

## 1. Requirements

- Java （Version=1.8）

## 2. Prepare datasets
The datasets (Tweet, SearchSnippets, and GoogleNews) in the path ./datasets has been preprocessed with tokenization, filtering non-Latin characters, etc before, from [STTM](https://github.com/qiang2100/STTM). The vocab file is the dictionary file corresponding to the dataset. For the corresponding word_wiki and the word2VecSim, you can download from [this](https://drive.google.com/drive/folders/1RhvgiD57TDy4Ea6ZsTAFubI7i7QlaPpD?usp=sharing). 

Taking Tweet as an example, the final dataset file path is as follows.

>datasets
>> Tweet
>>> word_wiki
>>> 
>>> Tweet.txt
>>> 
>>> Tweet_vocab.txt
>>> 
>>> Tweet_Word2VecSim.txt

## 3. Run APU-DMM
    bash run.sh

`-algorithm`: APUDMM.

`-dataname`: Specify the name of dataset (Tweet, SearchSnippets, or GoogleNews).

`-alpha`: Specify the value of the Dirichlet prior. The default value is 50/ntopics.

`-beta`: Specify the value of the Dirichlet prior. The default value is 0.01.

`--ntopics`: Specify the number of topics. The default value is 50.

`-schema`: Specify the file of word similarity matrix.

`-GPUthreshold`: Specify the threshold for semantic similarity of words.

`-corpus`: Specify the file of the input corpus file.

`-output`: Specify the path to the output directory. The default output dir is './results/dataname/dataname_ntopics/'.

`-name`: Specify the name of the output file.

`-nBurnIn`: Specify the number of BurnIn. The default value is 500.

`-niters`: Specify the number of iterations.  The default value is 100.

## 4. Evaluation
Topic Coherence: [PMI](https://github.com/jhlau/topic_interpretability).

Topic Diversity: TU.

Topic Quality (TQ) = PMI x TU

    bash evaluate.sh


## Citation
If you want to use our code, please cite as

	@inproceedings{DBLP:conf/iconip/LiWLBHCH23,
	  author       = {Mark Junjie Li and
	                  Rui Wang and
	                  Jun Li and
	                  Xianyu Bao and
	                  Jueying He and
	                  Jiayao Chen and
	                  Lijuan He},
	  editor       = {Biao Luo and
	                  Long Cheng and
	                  Zheng{-}Guang Wu and
	                  Hongyi Li and
	                  Chaojie Li},
	  title        = {Topic Modeling for Short Texts via Adaptive P{\textdollar}{\textbackslash}acute\{o\}{\textdollar}lya
	                  Urn Dirichlet Multinomial Mixture},
	  booktitle    = {Neural Information Processing - 30th International Conference, {ICONIP}
	                  2023, Changsha, China, November 20-23, 2023, Proceedings, Part {XIV}},
	  series       = {Communications in Computer and Information Science},
	  volume       = {1968},
	  pages        = {364--376},
	  publisher    = {Springer},
	  year         = {2023},
	  url          = {https://doi.org/10.1007/978-981-99-8181-6\_28},
	  doi          = {10.1007/978-981-99-8181-6\_28},
	  timestamp    = {Tue, 28 Nov 2023 09:46:07 +0100},
	  biburl       = {https://dblp.org/rec/conf/iconip/LiWLBHCH23.bib},
	  bibsource    = {dblp computer science bibliography, https://dblp.org}
	}
