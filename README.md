# Introduction
While methods based on deep learning have achieved state-of-the-art results on numerous tasks, most of these methods are based on the assumption that the data used to train the model is drawn from the same distribution that the data used to test the model is drawn from. This assumption can be violated in the real world, where AI models should be able to adapt themselves to new environments with little human supervision. In order to address this problem, methods based on single-source unsupervised domain adaptation utilize labeled data from a source domain to achieve satisfactory performance on unlabeled data from a different but related target domain [1].

This project is an implementation of "Adversarial Discriminative Domain Adaptation" [2]. We implemented the proposed learning framework and replicated all seven experiments of [2]. For further information, please refer to [Docs/Docs.pdf](./Docs/Docs.pdf).  

# References
[1] G. Wilson and D. J. Cook, “A survey of unsupervised deep domain adaptation,” ACM Transactions on Intelligent Systems and Technology (TIST),
vol. 11, no. 5, pp. 1–46, 2020.  

[2] E. Tzeng, J. Hoffman, K. Saenko, and T. Darrell, “Adversarial discriminative domain adaptation,” in Proceedings of the IEEE conference on computer
vision and pattern recognition, pp. 7167–7176, 2017.