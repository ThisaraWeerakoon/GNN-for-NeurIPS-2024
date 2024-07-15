
<div align="center">    
 
# Predict small molecule-protein interactions using BELKA   

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)


<!--  
Conference   
-->   
</div>




![images (11)](https://github.com/user-attachments/assets/cd1090d6-261f-4765-929a-7a5273f749fe)

Initially motivated by a kaggle competition <a href="https://www.kaggle.com/competitions/leash-BELKA">NeurIPS 2024 - Predict New Medicines with BELKA</a>


## Description   
This project aims to develop machine learning model to predict the binding affinity of small molecules to specific protein targets.

- **Motivation:** This is a critical step in drug development for the pharmaceutical industry that would pave the way for more accurate drug discovery
- **Why:** There are likely effective treatments for human ailments hiding in that chemical space, and better methods to find such treatments are desirable to us all.Searching chemical space by inference using this type of computational models rather than running laboratory experiments contribute to advances in small molecule chemistry used to accelerate drug discovery.

- **Problem Solved:** The number of chemicals in druglike space has been estimated to be 10^60, a space far too big to physically search.ML approaches suggest it might be possible to search chemical space by inference using well-trained computational models rather than running laboratory experiments.
- **What I Learned:**
  - **1:** Fundamentals of Graph Neural Networks
  - **2:** Working with chemical data including chemical specifications like <b>Simplified Molecular-Input Line-Entry System (SMILES)</b>
  - **3:** Working with analytical database like <b>DuckDB</b>

## Background
Small molecule drugs are chemicals that interact with cellular protein machinery and affect the functions of this machinery in some way. Often, drugs are meant to inhibit the activity of single protein targets, and those targets are thought to be involved in a disease process. A classic approach to identify such candidate molecules is to physically make them, one by one, and then expose them to the protein target of interest and test if the two interact. This can be a fairly laborious and time-intensive process.

## Dataset
To evaluate potential search methods in small molecule chemistry, <a href="https://www.leash.bio/">Leash Biosciences</a> physically tested some 133M small molecules for their ability to interact with one of three protein targets using DNA-encoded chemical library (DEL) technology. This dataset, the <b>Big Encoded Library for Chemical Assessment (BELKA)</b>, provides an excellent opportunity to develop predictive models that may advance drug discovery.

## Methodology

1. **Data Visualization:**
   - We used the BELKA dataset
   - Chemical proteins were visualized using rdkit.
   - Here you can see a visualization of sample chemical structures of the dataset

<img width="1143" alt="molecules " src="https://github.com/user-attachments/assets/4e42dc86-e4d1-4e4c-bb8b-db83c90019ff">


2. **Data Preprocessing:**
   - Chemical compounds were in SMILES format.To use them in Graph Neural Networks need to convert them into Graph data object.
   - Use <a href="https://deepchem.io/">DeepChem</a> library to convert SMILES formatted chemical compound into data object with chemical data.
   - In simple terms,<a href="https://deepchem.io/">DeepChem</a>'s <b>MolGraphConvFeaturizer</b> converts SMILES into object with the atoms and its featutes plus bonds and bonding features.
   - Finally we convert SMILES formatted molecule into a graph object using <a href="https://pytorch-geometric.readthedocs.io/en/latest/">Pytorch Geometric</a> Custom Dataset Class.
   - Code used to convert data
   ```python
	import os.path as osp
	import torch
	from torch_geometric.data import Dataset
	import pandas as pd
	import deepchem as dc
	from tqdm import tqdm
	from rdkit import Chem 
	import numpy as np

	class MoleculeDataset(Dataset):
    		def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        		"""
        		root = Where the dataset should be stored. This folder is split
        		into raw_dir (downloaded dataset) and processed_dir (processed data). 
        		"""
        		self.test = test
        		self.filename = filename
        		super(MoleculeDataset, self).__init__(root, transform, pre_transform)
    
   		@property
    		def raw_file_names(self):
        		""" If this file exists in raw_dir, the download is not triggered.
            			(The download func. is not implemented here)  
        		"""
        		return self.filename

		@property
    		def processed_file_names(self):
        		""" If these files are found in raw_dir, processing is skipped"""
        		self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        		if self.test:
            			return [f'data_test_{i}.pt' for i in list(self.data.index)]
        		else:
            			return [f'data_{i}.pt' for i in list(self.data.index)]
    
    		def download(self):
        		pass

    		def process(self):
        		self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        		featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        		for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            			# Featurize molecule
            			mol = Chem.MolFromSmiles(row["molecule_smiles"])
            			f = featurizer._featurize(mol)
            			data = f.to_pyg_graph()
            			if not self.test:  # Only assign label if not a test set
                			data.y = self._get_label(row["binds"])
            				data.smiles = row["molecule_smiles"]
            			if self.test:
                			torch.save(data, 
                    				osp.join(self.processed_dir, 
                             				f'data_test_{index}.pt'))
            			else:
                			torch.save(data, 
                    				osp.join(self.processed_dir, 
                             				f'data_{index}.pt'))

   		def _get_label(self, label):
        		label = np.asarray([label])
        		return torch.tensor(label, dtype=torch.int64)

    		def len(self):
        		return self.data.shape[0]

    		def get(self, idx):
        		""" - Equivalent to __getitem__ in pytorch
            		- Is not needed for PyG's InMemoryDataset
        		"""
        			if self.test:
            				data = torch.load(osp.join(self.processed_dir, 
                             			f'data_test_{idx}.pt'))
   				else:
   					data = torch.load(osp.join(self.processed_dir, 
                             				f'data_{idx}.pt'))        
        			return data


## Visualization of the model using tensorboard
![Jul15_09-05-34_kumaras-MacBook-Air local](https://github.com/user-attachments/assets/650cf8a6-1382-4070-b75a-05d98bafba8d)

## Code Implementation

The project's code is organized in a Jupyter notebook. Key libraries used in the project include:

- `PyTorch Geometric` for train graph neural networks
- `rdkit` for chemical data processing
- `DeepChem` for converting SMILES into chemical data objects

## Credits

We used several third-party assets and tutorials, including:

- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [DeepChem](https://deepchem.io/)
- [RDkit](https://www.rdkit.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Badges

![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
	![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## How to Contribute

We welcome contributions from the community! If you are interested in contributing, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    ```sh
    git checkout -b feature-or-bugfix-name
    ```
3. Commit your changes:
    ```sh
    git commit -m "Description of the feature or bug fix"
    ```
4. Push to the branch:
    ```sh
    git push origin feature-or-bugfix-name
    ```
5. Open a pull request and provide a detailed description of your changes.
