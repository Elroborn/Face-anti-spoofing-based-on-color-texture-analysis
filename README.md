This is a python script for face anti-spoofing methods based on color texture features. If you find it useful please cite the following papers:
@inproceedings{boulkenafet2015, title={Face anti-spoofing based on color texture analysis}, author={Boulkenafet, Zinelabidine and Komulainen, Jukka and Hadid, Abdenour}, booktitle={IEEE International Conference on Image Processing (ICIP), 2015}, pages={2636--2640}, year={2015}, organization={IEEE} }

@ARTICLE{boulkenafet2016, author={Z. Boulkenafet and J. Komulainen and A. Hadid}, journal={IEEE Transactions on Information Forensics and Security}, title={Face Spoofing Detection Using Colour Texture Analysis}, year={2016}, volume={11}, number={8}, pages={1818-1830}, doi={10.1109/TIFS.2016.2555286}, ISSN={1556-6013}, month={Aug},}

# Introduction   

First of all, you need to do some processing on the data set such as sampling, face recognition.  We will get `train_file_list.txt` and `test_file_list.txt`. Their format is as follows:
image_file_path label

/home/CASIA_frames/test_release/27/1/frame_42.jpg 1

You can refer to [this project](https://github.com/coderwangson/Learn-Convolutional-Neural-Network-for-Face-Anti-Spoofing_pytorch) to do these process.

After that, you can run `feature_extract.py` to generate features. `main.py` to train and test you model.  In the end you will get the following result:  

![](https://github.com/coderwangson/Face-anti-spoofing-based-on-color-texture-analysis/blob/master/roc.png)  

The pipeline of the method as follows:  

![](https://github.com/coderwangson/Face-anti-spoofing-based-on-color-texture-analysis/blob/master/architecture.png)


