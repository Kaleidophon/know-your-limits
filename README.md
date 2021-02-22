# Know Your Limits: Uncertainty Estimation with ReLU Classifiers Fails at Reliable OOD Detection

@TODO
* Add link to published paper
* Add citation 

This is the Github repository for the UAI 2021 of the same name, investigating 
how ReLU activation functions and the softmax function in neural classifiers 
This repository give an overview over findings, explain the repository structure and gives instruction on installation 
and usage. 

## Findings 

We build on previous research by Arora et al. (2018), showing that neural networks with piece-wise linear activation 
functions divide the feature space into polytopal regions, on which they can be expressed as an affine function. The 
plots below show the predictive entropy of a neural classifier on a synthetic multi-class classification problem (left)
as well as the polytopes it induces in the feature space (right; using the code of Jordan et al., 2019).

<p align="middle">
    <img src="plots/uncertainty.png" width="40%" />
    <img src="plots/polytopes.png" width="40%" />
</p>

We show that on the open-ended polytopes that extend infinitely, the predicted class probabilities for a point, but also 
the uncertainty of the classifier will approach fixed points in the limit. We illustrate this by measuring the magnitude
of the model output (or uncertainty) w.r.t. to the input x. In regions where these scores don't change, a budge in the
input won't change the score - thus the gradient will be short or even zero. We show this behavior for different models 
and uncertainty metrics, see some plots below (and more in the paper section 6 and appendix B).

<p align="middle">
    <figure>
        <img src="plots/nn_max_prob_grads.png" width="30%" />
        <figcaption>Neural discriminator with max. prob. (Hendrycks & Gimpel, 2017)</figcaption>
    </figure>
    <figure>
        <img src="plots/mcdropout_mutual_information_grads.png" width="30%" />
        <figcaption>MC Dropout (Gal & Ghahramani, 2016) with mutual information (Smith & Gal, 2018)</figcaption>
    </figure>
    <figure>
        <img src="plots/nnensemble_var_grads.png" width="30%" />
        <figcaption>Neural ensemble (Lakshminarayanan et al., 2017) with class variance (Smith & Gal, 2018)</figcaption>
    </figure>
</p>



## Installation

@TODO 

## Repository Structure

@TODO

## Usage

@TODO

## Citation

If you are using any code in this repository or cite our work, please cite us using the 
information below:

@TODO

## Bibliography 

@TODO