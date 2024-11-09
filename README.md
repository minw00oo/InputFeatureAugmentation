# InputFeatureAugmentation

## Abstract

This study proposes an efficient method for analyzing complex fracture patterns in the cross-sections of unidirectional (UD) composites, influenced by the volume fraction (VF) and fiber arrangement, and for predicting the corresponding transverse mechanical responses. Traditional finite element (FE) analysis incurs high computational costs when evaluating responses for every configuration. To address this, deep learning (DL), particularly convolutional neural networks (CNNs), have been applied, but these approaches have typically focused on limited VF spaces, leading to large data requirements and reduced prediction accuracy for new configurations.	In this research, we introduce a novel DL approach that can be effectively extended to broader VF spaces by integrating low-cost, physically insightful auxiliary features as multi-modal inputs. Specifically, we selected the Mori-Tanaka (MT) feature and stress concentration factor (SCF) as auxiliary inputs and incorporated them into the conventional CNN model for comparative analysis. The results showed that the model incorporating the MT feature significantly improved extrapolation performance in unseen VF spaces while maintaining robust predictive performance as the training dataset size increased. In contrast, the SCF feature did not demonstrate similar benefits. These findings illustrate that integrating advanced features like the MT feature into the DL model can offer more effective and versatile solutions for predicting material properties.

Keywords Unidirectional Composites, Convolutional Neural Network, Transverse Mechanical Behavior, Homogenization

## FeatureAugmentedCNN.py

"Provides a model combining Mori-Tanaka (MT) features and Stress Concentration Factor (SCF) features."
