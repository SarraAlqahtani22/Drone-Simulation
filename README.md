# Drone_Simulation
*Description: This Repo aimed in providing toolchain for simulating output result provided by neural network to help test autonomous navigation algorithms for UAVs*
This simulation has been used in the following papers:
1) shley Peake, Joe McCalmon, Yixin Zhang, Daniel Myers, Sarra Alqahtani, Victor Paúl Pauca, “Deep Reinforcement Learning for Adaptive Exploration of Unknown Environments”, the IEEE International Conference on Unmanned Aircraft Systems, (ICUAS ’21).
2) Yixin Zhang, Joe McCalmon, Ashley Peake, Sarra Alqahtani, Victor Paúl Pauca, “A Symbolic-AI Approach for UAV Exploration Tasks”, the 8th International Conference on Automation, Robotics and Applications (ICARA 2021), accepted.
3) Ashley Peake*, Joe McCalmon*, Yixin Zhang, Benjamin Raiford, and Sarra Alqahtani, “Wilderness Search and Rescue Missions using Deep Reinforcement Learning,” IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR), 2020.

Default setting: the default setting of this program will generate 2000*2000 pixel image with different features on it.

Please read the comments in the files to see which parameters are available for changes.

## *File explanation (By execution sequence):*

   ### 1. `Generator.py`: Generating the positions, size and rotation angle of all the feature on the simulated map and then draw them out.
    
        Output: GT_Full.png (the raw ground truth image)
        
                wFakeObjects_Full.png (the high resolution image with artificially added inaccuracy)

                GT_Info_Full.png (information quantity in the ground truth image, with vale 0~255

                GT_Info_Full.npy (GT_Info_Full.png stored in a 2D np-array)

        Key functions:

            feature_gen(k, n, r)
                k: how many square regions is the original image to be cut into.
                n: number of target feature
                r: maximum size of the target feature.

            load_feature(feature, shape, target, feature_type)
                feature: a n*3 or n*4 array containing information of the feature, typically generated by feature_gen
                shape: for now we can do 'ellipse' or 'circle', more shapes can be customized
                target: enhance(), real(), or reduce()
                feature type: 'gradient' or None, when choosing None it gives a flat color representation

        Helper functions:
            enhance(): color for fake feature
            real(): color for real feature
            reduce(): color for obstacle

            Returned by those functions are the color and alpha to be used by different features.

                
   ### 2. `blurry.py`: *Taking the two inputs from Generator.py and add salt and pepper noise w/ MODIFIABLE intensity and proportion to testing1.png.*
    
        Output: wFakeObjects_Full+s&p.png (Added salt and pepper noise to testing1.png)

        Key functions:
            salt_noisy(image, density=0.04, portion=0.5)
                image: an image opened by cv2
                density: the intensity of the salt and pepper noise, the larger, the more noise
                portion: the proportion of salt noise in the overall noise

    
###    3. `Interpreter.py`: *Calculating the information within each region(smaller square) and outputting the RAW information map. Also adding the gaussian noise to it.*
    
        Output: GT_Info_Convoluted.npy (The information map, an 2D array which contains the ground-truth information of each subregion)
                
                wFakeObjects_Info_Convoluted+s&p+Gaussian.npy (The information map, an 2D array which
                                                                    contains the information of each region
                                                                    WITH the salt&pepper and gaussian noise in simulating blurry)
                
                Note: Both of the .npy files take the salt and pepper noise into account and both gives RAW information. Normalization can be made directly on it.

        Key functions:

            calculate_info(index, GT_Img0, n_worker0, boxSize0)
                index: parameter for parallelization
                GT_Img0: the blurred image with all feature that is to be calculated information map
                n_worker0: # of processor to be used
                boxSize0: the edge length in pixel for each information region
                *** Inside this function a formula for calculating information is used. But this formula can be changed.

                For now, it is in line 31 and looks like this: "val = (loc[2] - (float(loc[0]) + loc[1]) / 2) / 255";
                What it means is:(R-(B+G)/2)/255; More information is available within the file

     
 ###   4. `Reconstruct.py`: *Visualization of the information map. Can be used to compare with the result given by GT-testing_bw.png. We can visually find that we have some ROI being*
        hidden and some False positive (FP) start to appear on the product map. This fits exactly with the nature of a typical output from a Neural Network.
     
  ###  5. `Normalizer.py`: *Turn the information map from raw data into NN-like output (0-1 range). In this case, sigmoid function is used.*
