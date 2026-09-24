Attribution
-----------
If you use the dataset in scientific work, please cite:

Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten
Steger: The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly
Detection; arXiv preprint arXiv:2503.21622, 2025.

https://arxiv.org/abs/2503.21622


Content
-------
* script to check and prepare data for upload to the evaluation server
    check_and_prepare_data_for_upload.py
* utility functions:
    utils.py
* script to measure runtime and memory
    measure_runtime_and_memory.py
* dataset class for MVTecAD2
    mvtec_ad_2_public_offline.py
* required packages to run the code
    requirements.txt
* license file
    license.txt
* this readme file
    README.txt


Evaluation Code
---------------
If you wish to evaluate the publicly available part of our test set 
quantitatively, please visit
https://www.mvtec.com/company/research/datasets/mvtec-ad
and download the evaluation code for AU-PRO.
However, since the ground truth for the real test set is not publicly
available, please use our evaluation server under 
https://www.benchmark.mvtec.com to evaluate your method's performance and to
become part of the official leaderboard.


License
-------
Copyright 2025 MVTec Software GmbH

This work is licensed under a Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this work.
If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.

For using the data in a way that falls under the commercial use clause
of the license, please contact us.


Contact
-------
If you have any questions or comments about the dataset, feel free to
contact us via: benchmark@mvtec.com