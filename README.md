# Graph Neural Networks for Quantum Error: Detection

Nathan Hilsenbrand | Department of Computer Science | Virginia Commonwealth University | Richmond, VA 23284 | nhellenbrand@vcu.edu

## Abstract
Quantum Computers are inherently prone to errors due to the nature of quantum information where qubits can be prone to decoherence and noise. To handle these effects and achieve fault tolerant quantum computers, researchers have developed Quantum Error Correcting Codes (or Surface Codes) to detect and correct these errors [[3]](#3). These codes often take a topological form and thus make a very practical use for Graph Neural Networks, where Nodes are measured qubit and Edges are connected qubits effected by errors. In this project I created a Surface Code generator that generates multiple graphs with a given error rate and feeds them into 2 different Graph Neural Networks (Graph Attention, and Graph Convolution) for Node Error Classification.

## Introduction
### Problem Description
Quantum computing holds immense promise for solving problems that are currently too complex for classical computers. However, quantum computers can only achieve speeds greater than classical computers when a significant number of qubits are involved. The delicate nature of quantum information makes error detection and correction crucial for realizing the full potential of quantum computers. As more qubits are introduced into the system so is the probability of errors propagating through the system. One of the leading error correction techniques is the surface code [[3]](#3), which efficiently detects and corrects errors using a two-dimensional or more lattice of qubits. The primary objective of this research is to investigate and compare the effectiveness of Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs), in detecting errors within quantum surface code graphs. The surface code generator produces graphs representing the spatial arrangement of qubits and their connectivity, simulating the physical layout of a quantum surface code and then feeding the generated graphs to the Graph Neural Networks.

### Motivation
As quantum computers incorporate more qubits and continually perform operations on the qubits they increase their chances of introducing errors in the system due to noise and decoherence. The significance of this research lies in leveraging machine learning techniques to enhance the accuracy and scalability of error detection processes in quantum systems. By taking advantage of the topological surface code we can find a good use for Graph Neural Networks to learn the local error patterns and how they correlate with neighboring nodes in order to apply corrections. Other methods such as minimum weight perfect matching (MWPM) [[1]](#1) can become difficult to scale with the increasing surface code size.

### Background
While classical error correction is nothing new it is something that has been successfully applied through the use of redundancy and repetition code. How classical error correction can be achieved, albeit in an inefficient and simple manor, is by repeatedly sending the code. With multiple codes sent and an error detected; the receiver can take a majority vote of the code to get the correct code. 

$$
\left[
\begin{array}{c}
000=0 \\
001=0 \\
\vdots \\
110=1 \\
111=1
\end{array}
\right]
$$


Quantum computers have a different set of problems when faced against this challenge. Instead of a classical 0 or 1, a qubit can be written as a superposition of state: 
$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$ where $\alpha$ and $\beta$ represent complex numbers with the condition $|\alpha|^{2} + |\beta|^{2} = 1$ and $|0\rangle$ and $|1\rangle$ represent the basis states. Due to their use of quantum mechanics the quantum information is susceptible to noise (external interference) and decoherence (internal interference) which can change properties of the qubit such as a bit flip (from $|0\rangle$ to $|1\rangle$), a phase flip (from $|+\rangle$ to $|-\rangle$, where $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$ and $\frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |-\rangle$), or both. In quantum error correction (QEC) these errors are referred to as X errors for it flip, Z errors for phase flip, and XZ errors for both (sometimes referred to as Y errors). These errors can be represented as Pauli matrices which have the same representation as their gates. 
```math
X = \left[\begin{array}{cc}
    0 & 1 \\\
    1 & 0 
\end{array}\right], 
Z = \left[\begin{array}{cc}
    1 & 0 \\\
    0 & -1 
\end{array}\right], 
Y = XZ = \left[\begin{array}{cc}
    0 & -i \\\
    i & 0 
\end{array}\right]
```


An additional problem when applying classical techniques to quantum is due to the quantum no cloning theorem [[5]](#5) where quantum states cannot be copied without destroying the state. This makes it difficult to observe the error and correct it.

### Related Work
To do QEC we can apply techniques similar to the classical solution by adding redundancy in the form of an extra qubit and entangling the two qubits to be a logical bit [[2]](#2) $|\psi\rangle_{L}$. $$|\psi\rangle_{L} = \alpha|00\rangle + \beta|11\rangle$$ We can take a protective measurement of the encoded bits through the use of stabilizers such as $Z_{1}Z_{2}$ [[2]](#2) to get a $|+\rangle$ eigenvalue. $$Z_{1}Z_{2}|\psi\rangle_{L}=Z_{1}Z_{2}(\alpha|00\rangle + \beta|11\rangle) = (+1)|\psi\rangle_{L}$$
An observer qubit called an ancilla is then applied to extract the syndrome for qec, this allows for the detection of errors without directly measuring the state of the encoded information. Based on the measurement outcome on the ancilla qubit determines the syndrome or error that occurred. 

This is a good start for detecting errors but the issues stem from the high number of needed qubits in a good quantum system and what the error rate is. With a higher number of qubits we have an exponentially larger graph which increases training and execution time. And a high error rate for a quantum system could have a densely packed set of errors. Densely packed errors can flip the same ancilla bit multiple times from multiple edges making it tricky to determine all the error sources. Additionally when an ancilla qubit is measured it can also produce an X, Z, or Y error. 


\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{"img/surface-code-25-17-13.png"}
    \caption{FIG. 1. An example of surface codes [[4]](#4). Surface code layouts with (a) 25, (b) 17, (c) 13 qubits. white nodes are logical qubits and black nodes are the ancilla qubits. Green patches represent X stabilizers and yellow patches represent Z stabilizers. (c) demonstrates the reuse of ancilla qubits to measure a set of 4 logical qubits and then a set of 2 logical qubits with different stabilizers.}
    \label{fig:enter-label}
\end{figure}
\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{img/surface code.png}
    \caption{Generated surface code graph where circle nodes are logical qubits, square nodes are ancilla qubits. Ancilla that are green show no error, and red ancilla show an error. This graph was generated with a distance of 3 and an error rate set to 10\%}
\end{figure}


It is very common to weave data and ancilla qubits together into a lattice to create a surface code [[4]](#4). A surface code can be multi-dimensional but is typically defined as a 2d grid of a given distance $d$ where there are $(2d-1) x (2d-1)$ vertices and $d$ is the length of the shortest logical operator or undetectable error chain [[4]](#4). Many different structures have been investigated to determine which structure works best for error detection. 

Given the surface code very much resembles a typical graph with vertices and edges, there has been recent work to look into the use of Graph Neural Networks (GNN) to predict the errors to be corrected. A recent article [[6]](#6) benchmarks 7 different kinds of GNN's to determine which ones may be best suited for the task.

## Proposed Method
### Graph Representation
The surface code I use can be defined as a 2-dimensional lattice of size $(2d-1) * (2d-1) + 4(d-1)$ with a given distance $d$ and additional ancilla qubits along each side represented as $4(d-1)$, alternating edges with logical qubits similar to the structure of FIG.1.(a) but with the extra ancilla along the sides like in FIG.1.(b) and as displayd in FIG.2. The extra ancilla qubits along the sides of the lattice allow for easier error detection among the logical qubits with lower number of edges. The graph can be further defined as $G=(V, E)$ where $V = {v_{1}, ...v_{2d-1}^{2}}$ and $E=\{e_{i,j}|(i,j)\subseteq|V|x|V|\}$. $V$ also consists of 2 alternating node types of either a logical qubit or ancilla qubit. Edges in $E$ only connect logical and ancilla qubits in that any ancilla qubit in the surface code can be defined as the errors on at minimum 2, and at maximum 4 logical qubits. The set of edges for an ancilla qubit in relation to the logical qubits it is connected to can be defined with an edge between ancilla qubit$v_{i}$ and logical qubit $v_{j}$ as  
$$e_{1} = v_{i}, v_{j-1}$$
$$e_{2} = v_{i}, v_{j+1}$$
$$e_{3} = v_{i}, v_{j+ (2d-1)}$$
$$e_{4} = v_{i}, v_{j- (2d-1)}$$
 so long as they are within the bounds of the graph, for example: $i, j <= (2d-1)^{2}$.

 ### GNN Architecture
 For this graph I looked at 2 different architectures with several different combinations of hyper parameters to determine what conditions give the best accuracy. The first model I used was a Graph Convolutional Network (GCN). These are good basic models for QEC in that it does a good job of aggregating messages from neighboring nodes to define local error patterns. However this locality becomes tricky to define as the error rate increases where multiple errors happen and the need to factor in the measurements of distance ancilla qubits comes into play. Adding additional convolutional layers helps with this but the returns in accuracy are very small with the most accurate model being with 2~3 layers.

The second model I looked at was a Graph Attention Model (GAT). GAt models are good at utilizing attention mechanisms to selectively aggregate information from neighboring nodes in a graph. This attention mechanism makes attention coefficients that determine the importance of neighboring nodes' features for a particular node. It then performs weighted aggregation where the coefficients serve as weights, determining how much attention each neighboring node's feature receives during aggregation. We can also use multi-head attention mechanisms where multiple sets of attention weights are learned by the model for a more in depth learning of the different aspects of the graph's structure and features.

### Training and Evaluation
For training the models I did an 8:2 split with 80\% of the graphs were used in training and 20\% were used for testing.
The loss function I used is Cross entropy, as it is a common function for classification tasks.
For evaluating the models I looked at the accuracy and plotted the accuracy and loss to see how well the model was performing or to look for any outstanding issues.

## Experiments
### Dataset
For this task there weren't any datasets available from what I could find. There were a couple helpful resources through 3rd party sfotware kits such as Qiskit and Stim that would generate the surface code with stabilizers and detectors, but the knowledge to use these tools efficiently enough before the project deadline became an obstacle. Instead I created my own surface code generator based on the design in a paper by Yue Zhao [[6]](#6) with some additional stabilizers on the outer edge. This allowed me to create however many homogeneous graphs I wanted to train the models on, adjust error rates, and package it all up to work nicely with torch-geometric's datasets. Because I'm not working on the evolution of the graph over time, I assume the starting state of the graph is all 0's.

In the paper I followed [[6]](#6), they used about 10,000,000 and 1,000,000 graphs for training which seems a bit excessive so I only started with a dataset of 100 graphs and didn't see a need to go any higher.

For each graph the data gets represented in 2 parts with an edge index and node embeddings. The edge index is (2 x number of edges) list with each index representing a non-directed edge between 2 node indexes. The node embeddings are made of 3 features: 
\begin{enumerate}
    \item Node Type
    \item Input Graph
    \item Ground Truth
\end{enumerate}

The node type can be either 0 or 1 and is assigned based on whether the node is an ancilla qubit or logical qubit. The Input graph is the measurement of what we observe from the ancilla qubits on the graph and can be either 0 or 1 resulting from the number of times the bit was flipped. for example node\_embedding$[i, 1] = 1$ if $i\%2$ else $0$. Finally the ground truth graph is the presence of actual errors on the graph that can be a range from 0-3 where the possibilities are 0 for no error, 1 for X error, 2 for Z error, and 3 for Y error.  These errors can appear on either node type.

To work with torch-geometric I found it most helpful to wrap all of the graphs and edge indexes into their dataset class before working with their model layers.


### Experimental Setup
The parameters of the graph generator were set between a distance of 3 (29 nodes), at minimum, and 7 (181 nodes) for comparisons between the 2 models. Error rates were set at 1\%, 2\%, 5\%, 10\%, 20\%, 30\%, 40\%. For the GCN model the hyperparameters that I used were hidden size: 16, 32, 64; number of layers: 2, 3, 4, 5; learning rate: 0.01, 0.001; weight decay: 0.0005, 0.0008. I had planned to have the number of graphs and number of epochs be adjustable but found pretty good results early on so I left those both at 100. I had also planned to include batch sizing and a hidden dropout but initial results didn't look like it needed the extra work. The other parameters were adjusted based on what gave the best accuracy. The whole project was setup and run in Google's Colab on GPU when available but mostly CPU.

### Results and Discussion
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GCN BEST LOSS.png}
        \caption{Plot of the best loss for the GCN; 1\% error rate}
        \label{fig:gcn1}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GCN BEST ACC.png}
        \caption{Plot of the best accuracy for the GCN; 1\% error rate}
        \label{fig:gcn2}
    \end{subfigure}
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GCN WORST LOSS.png}
        \caption{Plot of the worst loss for the GCN; 40\% error rate}
        \label{fig:gcn3}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GCN WORST ACC.png}
        \caption{Plot of the worst accuracy for the GCN; 40\% error rate}
        \label{fig:gcn4}
    \end{subfigure}
\end{figure}
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GAT BEST LOSS.png}
        \caption{Plot of the best loss for the GAT; 1\% error rate}
        \label{fig:gat1}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GAT BEST ACC.png}
        \caption{Plot of the best accuracy for the GAT; 1\% error rate}
        \label{fig:gat2}
    \end{subfigure}
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GAT WORST LOSS.png}
        \caption{Plot of the worst loss for the GAT; 40\% error rate}
        \label{fig:gat3}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{img/GAT WORST ACC.png}
        \caption{Plot of the worst accuracy for the GAT; 40\% error rate}
        \label{fig:gat4}
    \end{subfigure}
\end{figure}

When I first started testing I used the GCN as a baseline and saw very good results very quickly and performed well across varying sizes of graphs. It wasn't until I had increased the error rate to past 10\% that I really started to see the accuracy of the model fall below 90\%, I also plotted the accuracy and the loss to see how the model was doing during training.

The fine tuning for the best accuracy of the GCN was with a Distances of 3, Error Rate of 1\%, Hidden Size set to 32, 2 layers, a learning rate of 0.01 and a weight decay of 0.0008. This tuning gave an accuracy of 99.23\%.

The GAT was built and tested next and similarly it performed quite well right away, also across varying sizes of graphs. I never saw a huge drop off in accuracy as the error rate was increased, showing great scalability.

The fine tuning for the best accuracy of the GAT was with a Distances of 7, Error Rate of 1\%, Hidden Size set to 32, 2 layers, a learning rate of 0.01, weight decay of 0.0008, and 8 heads. This tuning gave an accuracy of 99.94\%.

Both models performed surprisingly well when the error rate remained low and the size of the graph increased. This may attribute to the models being able to recognize smaller patterns in a larger space, as the real complications come from multiple errors overlapping one another and hiding each other. Because of this I chose to stick to smaller graphs as the denser the graph the more trouble the GNN should have.

So with the distance of 3 and an error rate set to 40\% we can see how the models start to break down in accuracy, at least for the GCN with 61.38\% accuracy. GAT, surprisingly maintained a high accuracy of 91.64\%. The reasoning for this difference is that GAT's excel very well at capturing global information compared to GCN's, and as more errors are introduced the telling effects of the errors will be pushed farther and farther away from the effected nodes and the patterns become too big for a local problem. This is because as more errors are introduced, so are more overlapping errors which obscures the ground truth. GAT's also apply their own weights to neighboring nodes which can be helpful as of the 4 labels (0 error, X error, Z error, Y error) only the Y error can effect 4 possible nodes and Z and X errors can effect up to 2 possible nodes.

### Conclusion and Future Work
In my study of graph neural networks for the use of quantum error detection I found that what benefits a GNN most is being able to determine local patterns from a global space. This is shown in the performance of the GAT relative to the GCN in it's ability to accurately perform node classifications and detect errors in a quantum surface code. This work is a great start but there are many areas it can be expanded upon. For example the surface codes used in my experiment shows a repeatable pattern for sub graphs within the main graph. These sub graphs could be exploited for models that work well with smaller graphs and parallelized for speedup or classified into their own local patterns and how they effect other sub groups. The graph generator can also be improved to better simulate a quantum environment, as the current implementation mainly focuses on external noise (for example a fixed error rate). The generator in future versioning could include data for decoherence errors (errors that build up over time as operations are performed on the qubit) this would allow for better modeling of a quantum system. There are also many different forms of surface codes being discovered or looked into, and there could be greater success in such forms.

## References
<a id="1">[1]</a> Ben Criger and Imran Ashraf. Multi-path summation for decoding 2d topological codes. arXiv162 preprint arXiv:1709.02154, 2018.163
<a id="2">[2]</a> Daniel Gottesman. An introduction to quantum error correction and fault-tolerant quantum164 computation. Quantum information science and its contributions to mathematics, Proceedings of165 Symposia in Applied Mathematics, 2010.166
<a id="3">[3]</a> Joschka Roffe. Quantum error correction: An introductory guide. arXiv preprint167 arXiv:1907.11157, 2019.168
<a id="4">[4]</a> Yu Tomita and Krysta M. Svore. Low-distance surface codes under realistic quantum noise.169 arXiv preprint arXiv:1404.3747, 2014.170
<a id="5">[5]</a> William K Wootters and Wojciech H Zurek. A single quantum cannot be cloned. Na-171 ture,299(5886):802–803, 1982.172
<a id="6">[6]</a> Yue Zhao. Benchmarking machine learning models for quantum error correction. arXiv preprint173 arXiv:1709.02154, 2024.174
