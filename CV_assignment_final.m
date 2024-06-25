datasetPath = 'images';
maskPath = 'masks';

imageFiles = dir(fullfile(datasetPath, '*.jpg'));
maskFiles =  dir(fullfile(maskPath, '*.png'));

features = [];

% extract labels

groundtruth_labels = [];
labelsData = load('groundtruth_labels.mat'); 
fieldNames = fieldnames(labelsData);
groundtruth = labelsData.groundtruth_labels;
imageNames = groundtruth(:, 1);  
labels = groundtruth(:, 2);

for i = 1:length(labels)
    if strcmp(labels{i}, 'malignant')
        groundtruth_labels = [groundtruth_labels; 0];
    else
        groundtruth_labels = [groundtruth_labels; 1];
    end
end

 disp('Extracting features...')

% Loop through images and extract features

for i = 1:length(imageFiles)

    imagePath = fullfile(datasetPath, imageFiles(i).name)
    currentMaskPath = fullfile(maskPath, maskFiles(i).name);
    
    img = imread(imagePath);
    mask = imread(currentMaskPath);

    % Pre-process
    pre_processed_img = pre_process(img, mask);

  
    
    % COLOUR

   % Colour hist 1

    % Calculate average/standard deviation histogram 
    histograms = create_histogram(pre_processed_img);
    histograms = histograms.joint;
    histogramsNumeric = cell2mat(struct2cell(histograms)');

    SDColour = std(histogramsNumeric);
    SDColour = zscore(SDColour);

 


    % Colour hist 2

    hh = colourhist(img);


    % Colour moments

    colourMoments = calculateColourMoments(img);

    RMean = colourMoments.mean.red;
    GMean = colourMoments.mean.green;
    BMean = colourMoments.mean.blue;
    RStd = colourMoments.std.red ;
    GStd = colourMoments.std.green;
    BStd = colourMoments.std.blue;
    RGBMomentFeature = [RMean, GMean, BMean];

    RGBMomentSTD = [RStd, GStd, BStd];
    RGBMomentSTD = zscore(RGBMomentSTD);

    LRMean = colourMoments.mean.lab.red;
    LGMean = colourMoments.mean.lab.green;
    LBMean = colourMoments.mean.lab.blue;
    LRStd = colourMoments.std.red ;
    LGStd = colourMoments.std.green;
    LBStd = colourMoments.std.blue;

    LMomentFeature = [LRMean, LGMean,LBMean];
    LMomentFeature = zscore(LMomentFeature);
    LSMomentSTD = [LRStd, LGStd, LBStd];

 

    ColourFeatures = [RGBMomentFeature, LMomentFeature, SDColour];


    % EDGE
  
    % Extract edge detection
    edge_img = apply_edge_detection(pre_processed_img);
    
    %standard devation edge feature
    edge_std = std(double(edge_img(:)));
    %number of uneven edges feature
    intensityChangeThreshold = 50; 
    unevenEdges = sum(abs(diff(edge_img(:))) > intensityChangeThreshold);
     %edge hist feature
    grayScale = rgb2gray(pre_processed_img);
    [Gx, Gy] = imgradientxy(grayScale);
    gradientOrientation = atan2d(Gy, Gx);
    gradientOrientation = mod(gradientOrientation, 360);
    edgeHist = histcounts(gradientOrientation(edge_img), 360, 'Normalization', 'probability');


    



    % TEXTURE

    % texture 
    textureFeatures = apply_texture(pre_processed_img);
    textureFeatures = zscore(textureFeatures);



    % SYMMETRY/SIMILARITY

    % jaccard similarity 
    maskVerticallyFlipped = flipud(mask);
    maskHorizontallyFlipped = fliplr(mask);
    verticalSimilarity = calculateJaccardSimilarity(mask,maskVerticallyFlipped);
    horizontalSimilarity = calculateJaccardSimilarity(mask, maskHorizontallyFlipped);

    edgeVerticallyFlipped = flipud(edge_img);
    edgeHorizontallyFlipped = fliplr(edge_img);
    verticalSimilarityEdge = calculateJaccardSimilarity(edge_img, edgeVerticallyFlipped);
    horizontalSimilarityEdge = calculateJaccardSimilarity(edge_img, edgeHorizontallyFlipped);

   
    %Similarity
    SimFeature = calculateSim(pre_processed_img);


    % HOG
    hogFeatures = calculateHOG(pre_processed_img);
   

    % concatenate features
    featureVector = [ColourFeatures, textureFeatures, verticalSimilarity, horizontalSimilarity, hh];
  
    % Append the feature vector to the features array
    features = [features; featureVector];

end

disp('performing classification...')
% SVM classification

% perform classification using 10CV
rng(1); 
svm = fitcsvm(features, groundtruth_labels);
cvsvm = crossval(svm);
pred = kfoldPredict(cvsvm);

% confusion matrix
figure;
[cm, order] = confusionmat(groundtruth_labels, pred);
confusionchart (cm, order);

% class loss
classLoss = kfoldLoss(cvsvm)
disp(classLoss)

%accuracy

% Extract values from confusion matrix
truePositives = cm(1, 1);
trueNegatives = cm(2, 2);
falsePositives = cm(2, 1);
falseNegatives = cm(1, 2);

% Calculate accuracy
accuracy = (truePositives + trueNegatives) / sum(cm(:));

disp(['Accuracy: ' num2str(accuracy)]);


% Specificity
specificity = trueNegatives / (trueNegatives + falsePositives);
disp(['Specificity: ' num2str(specificity)]);

% Sensitivity (Recall)
sensitivity = truePositives / (truePositives + falseNegatives);
disp(['Sensitivity (Recall): ' num2str(sensitivity)]);


%pre-process function

function pre_processed_img = pre_process(img, mask)

    % apply mask
    mask = logical(mask);
    
    %source: %https://uk.mathworks.com/matlabcentral/answers/1609450-background-removal-in-matlab
    pre_processed_img = bsxfun(@times, img, cast(mask, 'like', img));
    pre_processed_img= imresize(pre_processed_img, [1022, 1022]);

    % hist eq code source: Lab 2, solution 3:

    %hist eq
    %h = imhist(pre_processed_img);
    %ch = cumsum(h);
    %cdf = ch/ch(end);
   % mapping = floor(255*cdf);
    %pre_processed_img = uint8(mapping(pre_processed_img+1));

    %filter = fspecial('laplacian', 0) ;
    %filteredimg = imfilter(pre_processed_img, filter);
    %pre_processed_img = pre_processed_img - filteredimg;

end


% Colour variation function
function colour_hist = create_histogram(img)

    Red = img(:,:,1);
    Green = img(:,:,2);
    Blue = img(:,:,3);

    % Get histValues for each channel
    colour_hist.red = imhist(Red, 256);
    colour_hist.green = imhist(Green, 256);
    colour_hist.blue = imhist(Blue, 256);

    colour_hist.joint = struct('Red', colour_hist.red, 'Green', colour_hist.green, 'Blue', colour_hist.blue);

end


% edge detection

function edge_detected_img = apply_edge_detection(img)

    img = im2gray(img);
    edge_detected_img = edge(img, 'Canny');

end


%
function texture_extraction = apply_texture(img)

    img = im2gray(img);
    texture_extraction = extractLBPFeatures(img,'Upright',false);

    
end


function  jaccardSimilarity = calculateJaccardSimilarity(mask, flippedMask)
    % Consider only the nonzero pixels
    intersect = mask > 0 & flippedMask > 0;
    union = mask > 0 | flippedMask > 0;
    numerator = sum(intersect(:));
    denominator = sum(union(:));
    jaccardIndex = numerator / denominator;
    jaccardSimilarity = 1 - jaccardIndex;
end


function colourMoments = calculateColourMoments(image)

    % Compute color moments for the entire image
    
    colourMoments.mean.red = mean(image(:, :, 1), 'all');
    colourMoments.mean.green = mean(image(:, :, 2), 'all');
    colourMoments.mean.blue = mean(image(:, :, 3), 'all');

    colourMoments.std.red = std(double(image(:, :, 1)), 0, 'all');
    colourMoments.std.green = std(double(image(:, :, 2)), 0, 'all');
    colourMoments.std.blue = std(double(image(:, :, 3)), 0, 'all');

    % Convert RGB image to Lab color space
    lab_image = rgb2lab(image);

    colourMoments.mean.lab.red = mean(lab_image(:, :, 1), 'all');
    colourMoments.mean.lab.green = mean(lab_image(:, :, 2), 'all');
    colourMoments.mean.lab.blue = mean(lab_image(:, :, 3), 'all');

    colourMoments.std.red = std(double(lab_image(:, :, 1)), 0, 'all');
    colourMoments.std.green = std(double(lab_image(:, :, 2)), 0, 'all');
    colourMoments.std.blue = std(double(lab_image(:, :, 3)), 0, 'all');


end


function hogFeatures = calculateHOG(img)

    hogFeatures = extractHOGFeatures(img);
    
end



function simScore = calculateSim(img)

    %grey scale
    img = rgb2gray(img);

    % Flip the image horizontally
    flippedImg = fliplr(img);

    % Calculate the correlation coefficient between the original and flipped images
    correlationCoeff = corr2(img, flippedImg);

    % Symmetry score is complement of the correlation coefficient
    simScore = 1 - correlationCoeff;
end







function H = colourhist(img)
    
    noBins = 8; 
    binWidth = 256 / noBins; 
    %initialise hist
    H = zeros(1, noBins * noBins * noBins * 3); 

     %reshape to 2d matrix
    [n, m, ~] = size(img);

    %each row corresponds to a pixel and each column represents a colour
    %channel
    data = reshape(img, n * m, 3); 

    %index of the bin to which each pixel belongs for each channel
    ind = floor(double(data) / binWidth) + 1;

    %Flatten
    for i = 1:length(ind)
        index = sub2ind([noBins, noBins, noBins, 3], ind(i, 1), ind(i, 2), ind(i, 3), mod(i, 3) + 1);
        H(index) = H(index) + 1;
    end

    %normalise
    H = H / sum(H); 
end































