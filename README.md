# HAIN
Implementation of HAIN in paper IEEE BigDATA 2020: "Hypergraph Attention Isomorphism Network by Learning Line Graph Expansion".


Hypergraph Attention Isomorphism Network in PyTorch
====

This is a PyTorch implementation of Hypergraph Attention Isomorphism Network (HAIN) model for semi-supervised hypernode classification task in a hyper-graph, as described in our IEEE-Bigdata 2020 paper:

"Hypergraph Attention Isomorphism Network by Learning Line Graph Expansion", Sambaran Bandyopadhyay, Kishalay Das, and M. Narasimha Murty.

## Installation
Run following command in your virtual environment

	pip install -r requirements.txt
	

## How to use
 Go to src directory

	python train.py --dataset <Dataset_Name>
	Dataset Name could be "cora","citeseer","pubmed","dblp"
