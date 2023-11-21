
addpath('./common/');

%%
% Load all training windows and get their HOG descriptors.
load('model.mat');

% Get the list of all images in the directory.
posFiles = getImagesInDir('./Images/Validation/Positive/', true);
negFiles = getImagesInDir('./Images/Validation/Negative/', true);

numOfAllPositive=length(posFiles);
numOfAllNegative=length(negFiles);

% Combine the file lists to get a list of all training images.
fileList = [posFiles; negFiles];

positiveNum=0;
negativeNum=0;

fprintf('Computing descriptors for %d Positive windows: \n', length(fileList(1,:)));

% For all training window images...
for i = 1 : length(fileList(1,:))

    % Get the next filename.
    imgFile = char(fileList(1,i));

    % Print the current iteration (using some clever formatting to
    % overwrite).
    printIteration(i);
    
    % Load the image into a matrix.
    img = imread(imgFile);
    
    % Calculate the HOG descriptor for the image.
    H = getHOGDescriptor(hog, img);
    
    % Evaluate the linear SVM on the descriptor.
   %%%%% p = H' * hog.theta;
    p=svmclassify(model,H');

    % (the SVM was trained to output -1 for no person and +1 for a person).
    if (p > 0)
      positiveNum=positiveNum+1;
    else
        fprintf('\n');
      fprintf(imgFile);
      fprintf('\n');
    end
    
    
end
fprintf('\n positiveNum is %f .\n',positiveNum);
    fprintf('\n TP is %f .\n',((positiveNum/numOfAllPositive)*100));
    fprintf('  FN is %f .\n',(100-((positiveNum/numOfAllPositive)*100)));
    
fprintf('Computing descriptors for %d Negative windows: \n', length(fileList(2,:)));
% For all training window images...
for i = 1 : length(fileList(2,:))

    % Get the next filename.
    imgFile = char(fileList(2,i));

    % Print the current iteration (using some clever formatting to
    % overwrite).
    printIteration(i);
    
    % Load the image into a matrix.
    img = imread(imgFile);
    
    % Calculate the HOG descriptor for the image.
    H = getHOGDescriptor(hog, img);
    
    % Evaluate the linear SVM on the descriptor.
   %%%%% p = H' * hog.theta;
    p=svmclassify(model,H');

    % (the SVM was trained to output -1 for no person and +1 for a person).
    if (p > 0)
     fprintf('\n');
     fprintf(imgFile);
     fprintf('\n');
    else
      negativeNum=negativeNum+1;
    end
    
    
end
fprintf('\n negativeNum is %f .\n',negativeNum);
    fprintf('\n  TN is %f .\n',((negativeNum/numOfAllNegative)*100));
    fprintf('  FP is %f .\n',(100-((negativeNum/numOfAllNegative)*100)));

