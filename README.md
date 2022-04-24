# MDRL
The code for paper "Noninvasive Diagnosis of Oral Squamous Cell Carcinoma by Multi-level Deep Residual Learning on Optical Coherence Tomography Images"


Prerequisites

	python3.7
	
	GPU  2080ti 12G
	
	pytorch 1.10.0
	
	matplotlib
	
	pandas
	
	numpy
	
	requests
	
	wget
	
	xlrd
	
Getting Started

	Dataset Preparation
	
		Download the dataset [Baidu Disk]
		
		Preparation: To generate train or test .txt file. Youmay use generate_txt_file.py
		
		Note to modify the dataset path to your own path.
		
		You need move txt files to the 'data'.
		
		Note that the training data are .tif files and the test data are .tiff files, please modify the code in 				generate_txt_file.py before you generate txt files 		.
				
	Testing
		
		Download the trained model
		
		We provide our trained model. You may download it from Baidu Disk (password: xxxx). You may download and move it to the path the same to 				net_train.py.

		├ oct_evaluation
		     ├── data/oral/oct/
		│        ├── train_part.txt, test_part.txt
		     ├── model
		 

		1.make sure you put all files into corect path     
		
		2.run test(test_loader=test_loader, lmda=0.5, mode='part_acc') in net_train.py

	Training
	
		If you want train the net from scratch, you need to delete the checkpoint.pth.tar and the model_best.pth.tar.
		Then run  main(lmda=0.5) in net_train.py.

		
