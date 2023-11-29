##### Problem Statement
- Train a model that performs multi-label classification of PubMed articles (inputs can be associated with multiple classes)
- The dataset consists of ~50k research articles from PubMed 
- There are 14 possible labels to assign to each article
- **Multi-label classification at root level**
    * `meshMajor` column values represent a MeSH Major Topic; one of the main topics discussed in an article
    * `meshroot` column values represent the multiple possible labels for each article
        * Columns `'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z'` correspond to the labels (i.e. `B` is a label associated with a meshroot value `'Organisms'`) 
- MLOps tools such as “weights and biases” or tensorboard.
    - [PyTorch TensorBoard Support](https://pytorch.org/tutorials/beginner/introyt/tensorboardyt_tutorial.html)
    - [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
    - [MLFlow](https://mlflow.org/)
    - [Comet ML](https://www.comet.ml/site/) 
    - [Neptune](https://neptune.ai/) 
    - [Weights and Biases](https://www.wandb.com/)  


##### Solution
* In a multi-label classification setting, the objective function is formulated like a binary classifier where each neuron in the output layer is responsible for one classification. 
- Sigmoid activation function for the output layer in a multi-label classification setting, allowing us to have a high probability for all classes or some of them, or none of them.
- `binary_crossentropy` loss is used for the multi-label classification.
* The solution focuses strictly on the `abstractText` feature with their MeSH classification (A: Anatomy, B: Organism, C: Diseases, etc.)
* The solution performs transfer learning on BERT (`"bert-base-uncased"`)
    * This version has only lowercase letters ("uncased") and is the smaller version of the two ("base" vs "large").
- The model is hosted at https://huggingface.co/DAfromsky/Multi-Label-Classification-PubMed-Articles
- The HF space is located here: https://huggingface.co/spaces/DAfromsky/DAfromsky-Multi-Label-Classification-PubMed-Articles
##### Inference	
* The solution is deployed/packaged as a Linux container (Docker container), with all the dependencies and necessary files to keep the application isolated. 
* The Docker container will have its own isolated running process, file system, and network. 
* The container is started/ran from the container image (static version of all the files, environment variables, and the default command/program present in the container). 
* Any changes caused by the running container will exist only in that container, but would not persist in the underlying container image (would not be saved to disk). 
* The container image includes in its metadata the default program/command that should be run when the container is started and the parameters to be passed to that program/command. 
* The container image is a Docker image, based on an official Python image.
    ###### How to Use
    * Requires an installation of **Docker** then can run the following commands:
        ###### Build the Docker image
        * `docker build -t <image_name> --target base .`
        
        ###### Run the Docker image
        * `docker run -it -v $PWD:/data --rm <image_name> bash`, where `$PWD` is the absolute file path to the current working directory with only the custom input file in its path
        * `--rm` option tells Docker to automatically remove the container when it exits.
    
    * This command will bring up a Linux terminal and bring you to the `/code` directory. From here you can run `python3 main.py --input-file /data/<name_of_file.txt>` and use tab/auto-complete after the `/data` directory to auto-grab the file.
    * To exit the Docker shell, run `exit`

##### Considerations
* **Container Memory**
    * Running a single process per container will have a more or less stable, and limited amount of memory consumed by the container
    * If we wanted to deploy this solution to a cluster, we'd be able to set those same memory limits and requirements in a configuration for the container management system like Kubernetes. That way it will be able to replicate the container in the available machines taking into account the amount of memory needed by them, and the amount available in the machines in the cluster. The app is simple, so this wouldn't necessarily be a problem, but something to consider for more resource-intensive applications, where we would want to adjust the number of container in each machine or add more machines to the cluster
* 
* **Sharded data parallelism:**
    - The **model parameters** include the floats that make up our model layers
    - The **gradients** are needed to do back-propagation
    - The **optimizer states** include statistics about the gradients
    - Sending a **batch of data** for model development also takes up GPU memory
    - Options:
        - [ZeRO](https://arxiv.org/pdf/1910.02054.pdf) that ***shards the optimizer states, the gradients, and the model parameters.*** This allowed for memory reduction and effectively increase in batch sizes.
        - [DeepSpeed](https://github.com/microsoft/DeepSpeed) library and Facebook's [FairScale](https://github.com/facebookresearch/fairscale) library
        - Natively by PyTorch: [Fully-Sharded DataParallel](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- **Pipelined Model-Parallelism:**
    * Put each layer of the model on each GPU
    * Need to tune the amount of pipelining on the batch size to the exact degree of how you will split up the model on the GPU.
* **Tensor-Parallelism**
    * Matrices can be distributed over multiple GPUs

##### Appendix
- Inference output from the Docker shell:

```
root@4b24ba0267ed:/code# python3 main.py --input-file /data/test_input.txt 

[
    [
        {'label': 'Organisms [B]', 'score': 0.6440446972846985}, 
        {'label': 'Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]', 'score': 0.14034181833267212}, 
        {'label': 'Diseases [C]', 'score': 0.06631450355052948}, 
        {'label': 'Phenomena and Processes [G]', 'score': 0.04863746464252472}, 
        {'label': 'Named Groups [M]', 'score': 0.046737637370824814},
        {'label': 'Health Care [N]', 'score': 0.041156988590955734},
        {'label': 'Anatomy [A]', 'score': 0.007876417599618435}, 
        {'label': 'Chemicals and Drugs [D]', 'score': 0.0015288549475371838},
        {'label': 'Geographicals [Z]', 'score': 0.000815624080132693},
        {'label': 'Psychiatry and Psychology [F]', 'score': 0.0008115688688121736}, 
        {'label': 'Anthropology, Education, Sociology, and Social Phenomena [I]', 'score': 0.0006262866663746536},
        {'label': 'Information Science [L]', 'score': 0.0004905554233118892}, 
        {'label': 'Technology, Industry, and Agriculture [J]', 'score': 0.00031426598434336483},
        {'label': 'Disciplines and Occupations [H]', 'score': 0.00030347766005434096}
    ]
]
[
    [
        {'label': 'Organisms [B]', 'score': 0.27970150113105774},
        {'label': 'Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]', 'score': 0.20454664528369904}, 
        {'label': 'Named Groups [M]', 'score': 0.20238064229488373}, 
        {'label': 'Health Care [N]', 'score': 0.17900340259075165}, 
        {'label': 'Diseases [C]', 'score': 0.07676520943641663}, 
        {'label': 'Phenomena and Processes [G]', 'score': 0.042748354375362396}, 
        {'label': 'Geographicals [Z]', 'score': 0.005497759208083153}, 
        {'label': 'Psychiatry and Psychology [F]', 'score': 0.003129183081910014}, 
        {'label': 'Anthropology, Education, Sociology, and Social Phenomena [I]', 'score': 0.0017124471487477422}, 
        {'label': 'Chemicals and Drugs [D]', 'score': 0.0017090318724513054}, 
        {'label': 'Information Science [L]', 'score': 0.0012271283194422722}, 
        {'label': 'Anatomy [A]', 'score': 0.0009800974512472749}, 
        {'label': 'Disciplines and Occupations [H]', 'score': 0.00041991667239926755}, 
        {'label': 'Technology, Industry, and Agriculture [J]', 'score': 0.00017875021148938686}
    ]
]
```