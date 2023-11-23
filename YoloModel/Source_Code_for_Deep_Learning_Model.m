%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 18th Aug 2023 by Do-Kun Yoon              %
%                  Industrial R&D Center, KAVILAB Co. Ltd                %
%                       Email: louis_youn@kavilab.ai                     %
%                              Co-developers                             %
%                        Min-Jun Kang, Hayeong Cha                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close all

datamat = load("Elbow_label.mat"); %load the bounding box data

folderPath = 'c:\file_path\file_path\file_path\file_path\'; % Filepath
fileList = dir(fullfile(folderPath, '*.png'));  % Lists only PNG files
origin_filename = cell(length(fileList), 1);

for i = 1:length(fileList)
    origin_filename{i} = fullfile(folderPath, fileList(i).name);
end

origin_bbox = datamat.gTruth.LabelData;

FractureDataset = cat(2,origin_filename,origin_bbox);
FractureDataset.Properties.VariableNames("Var1") = "imageFilename"; % name change

emptyRows = cellfun('isempty', FractureDataset.Fracture);   % empty rows 
FractureDataset(emptyRows,:) = [];  % empty rows delete

% shuffle
shuffledIndices = randperm(height(FractureDataset));
idx = length(shuffledIndices);
FractureDataTbl = FractureDataset(shuffledIndices(1:idx),:);  

numFracture = size(FractureDataTbl, 1);     % data number
c = cvpartition(numFracture, 'HoldOut', 0.1);   % train+val : 0.9   test : 9.1

TrainValIndices = training(c);
TestIndices = test(c);
trainvaldatas = FractureDataTbl(TrainValIndices, :);
testDataTbl = FractureDataTbl(TestIndices, :);

% data score
imdsTest = imageDatastore(testDataTbl{:,"imageFilename"});
bldsTest = boxLabelDatastore(testDataTbl(:,"Fracture"));

% data combine
testData = combine(imdsTest,bldsTest);

inputSize = [512 512 3];
className = "Fracture";

% train, validation
numFolds = 5;
numFracture_trainval = size(trainvaldatas, 1); 
cv = cvpartition(numFracture_trainval, 'KFold', numFolds);

for fold = 1:numFolds
    trainingIndices = training(cv, fold);
    validationIndices = test(cv, fold);

    trainingDataTbl = FractureDataTbl(trainingIndices, :);
    validationDataTbl = FractureDataTbl(validationIndices, :);

    % data score
    imdsTrain = imageDatastore(trainingDataTbl{:,"imageFilename"});
    bldsTrain = boxLabelDatastore(trainingDataTbl(:,"Fracture"));
    
    imdsValidation = imageDatastore(validationDataTbl{:,"imageFilename"});
    bldsValidation = boxLabelDatastore(validationDataTbl(:,"Fracture"));


    % data combine
    trainingData = combine(imdsTrain,bldsTrain);
    validationData = combine(imdsValidation,bldsValidation);

    % show
    data = read(trainingData);
    I = data{1};
    bbox = data{2};
    annotatedImage = insertShape(I,"Rectangle",bbox);
    annotatedImage = imresize(annotatedImage,2);

    reset(trainingData);


    rng("default")
    trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
    numAnchors = 9;
    [anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);
    

    area = anchors(:, 1).*anchors(:,2);
    [~,idx] = sort(area,"descend");
    
    anchors = anchors(idx,:);
    anchorBoxes = {anchors(1:3,:)
        anchors(4:6,:)
        anchors(7:9,:)
        };

    % Network Model --------------------------------------------------------

    net=dlnetwork(layerGraph(), Initialize=false);

    tempNet = [
        imageInputLayer([512 512 3],"Name","input_1","Normalization","none")
        convolution2dLayer([3 3],32,"Name","conv_2","Padding","same")
        batchNormalizationLayer("Name","bn_2")
        functionLayer(@vision.cnn.mish,"Name","mish_2")
        convolution2dLayer([3 3],64,"Name","conv_3","Padding",[1 1 1 1],"Stride",[2 2])
        batchNormalizationLayer("Name","bn_3")
        functionLayer(@vision.cnn.mish,"Name","mish_3")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
        batchNormalizationLayer("Name","bn_4")
        functionLayer(@vision.cnn.mish,"Name","mish_4")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],64,"Name","conv_6","Padding","same")
        batchNormalizationLayer("Name","bn_6")
        functionLayer(@vision.cnn.mish,"Name","mish_6")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],32,"Name","conv_7","Padding","same")
        batchNormalizationLayer("Name","bn_7")
        functionLayer(@vision.cnn.mish,"Name","mish_7")
        convolution2dLayer([3 3],64,"Name","conv_8","Padding","same")
        batchNormalizationLayer("Name","bn_8")
        functionLayer(@vision.cnn.mish,"Name","mish_8")];
    net = addLayers(net,tempNet);

    tempNet = [
        additionLayer(2,"Name","add_9")
        convolution2dLayer([1 1],64,"Name","conv_10","Padding","same")
        batchNormalizationLayer("Name","bn_10")
        functionLayer(@vision.cnn.mish,"Name","mish_10")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_11")
        convolution2dLayer([1 1],64,"Name","conv_12","Padding","same")
        batchNormalizationLayer("Name","bn_12")
        functionLayer(@vision.cnn.mish,"Name","mish_12")
        convolution2dLayer([3 3],128,"Name","conv_13","Padding",[1 1 1 1],"Stride",[2 2])
        batchNormalizationLayer("Name","bn_13")
        functionLayer(@vision.cnn.mish,"Name","mish_13")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],64,"Name","conv_14","Padding","same")
        batchNormalizationLayer("Name","bn_14")
        functionLayer(@vision.cnn.mish,"Name","mish_14")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],64,"Name","conv_16","Padding","same")
        batchNormalizationLayer("Name","bn_16")
        functionLayer(@vision.cnn.mish,"Name","mish_16")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],64,"Name","conv_17","Padding","same")
        batchNormalizationLayer("Name","bn_17")
        functionLayer(@vision.cnn.mish,"Name","mish_17")
        convolution2dLayer([3 3],64,"Name","conv_18","Padding","same")
        batchNormalizationLayer("Name","bn_18")
        functionLayer(@vision.cnn.mish,"Name","mish_18")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_19");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],64,"Name","conv_20","Padding","same")
        batchNormalizationLayer("Name","bn_20")
        functionLayer(@vision.cnn.mish,"Name","mish_20")
        convolution2dLayer([3 3],64,"Name","conv_21","Padding","same")
        batchNormalizationLayer("Name","bn_21")
        functionLayer(@vision.cnn.mish,"Name","mish_21")];
    net = addLayers(net,tempNet);

    tempNet = [
        additionLayer(2,"Name","add_22")
        convolution2dLayer([1 1],64,"Name","conv_23","Padding","same")
        batchNormalizationLayer("Name","bn_23")
        functionLayer(@vision.cnn.mish,"Name","mish_23")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_24")
        convolution2dLayer([1 1],128,"Name","conv_25","Padding","same")
        batchNormalizationLayer("Name","bn_25")
        functionLayer(@vision.cnn.mish,"Name","mish_25")
        convolution2dLayer([3 3],256,"Name","conv_26","Padding",[1 1 1 1],"Stride",[2 2])
        batchNormalizationLayer("Name","bn_26")
        functionLayer(@vision.cnn.mish,"Name","mish_26")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_27","Padding","same")
        batchNormalizationLayer("Name","bn_27")
        functionLayer(@vision.cnn.mish,"Name","mish_27")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_29","Padding","same")
        batchNormalizationLayer("Name","bn_29")
        functionLayer(@vision.cnn.mish,"Name","mish_29")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_30","Padding","same")
        batchNormalizationLayer("Name","bn_30")
        functionLayer(@vision.cnn.mish,"Name","mish_30")
        convolution2dLayer([3 3],128,"Name","conv_31","Padding","same")
        batchNormalizationLayer("Name","bn_31")
        functionLayer(@vision.cnn.mish,"Name","mish_31")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_32");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_33","Padding","same")
        batchNormalizationLayer("Name","bn_33")
        functionLayer(@vision.cnn.mish,"Name","mish_33")
        convolution2dLayer([3 3],128,"Name","conv_34","Padding","same")
        batchNormalizationLayer("Name","bn_34")
        functionLayer(@vision.cnn.mish,"Name","mish_34")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_35");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_36","Padding","same")
        batchNormalizationLayer("Name","bn_36")
        functionLayer(@vision.cnn.mish,"Name","mish_36")
        convolution2dLayer([3 3],128,"Name","conv_37","Padding","same")
        batchNormalizationLayer("Name","bn_37")
        functionLayer(@vision.cnn.mish,"Name","mish_37")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_38");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_39","Padding","same")
        batchNormalizationLayer("Name","bn_39")
        functionLayer(@vision.cnn.mish,"Name","mish_39")
        convolution2dLayer([3 3],128,"Name","conv_40","Padding","same")
        batchNormalizationLayer("Name","bn_40")
        functionLayer(@vision.cnn.mish,"Name","mish_40")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_41");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_42","Padding","same")
        batchNormalizationLayer("Name","bn_42")
        functionLayer(@vision.cnn.mish,"Name","mish_42")
        convolution2dLayer([3 3],128,"Name","conv_43","Padding","same")
        batchNormalizationLayer("Name","bn_43")
        functionLayer(@vision.cnn.mish,"Name","mish_43")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_44");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_45","Padding","same")
        batchNormalizationLayer("Name","bn_45")
        functionLayer(@vision.cnn.mish,"Name","mish_45")
        convolution2dLayer([3 3],128,"Name","conv_46","Padding","same")
        batchNormalizationLayer("Name","bn_46")
        functionLayer(@vision.cnn.mish,"Name","mish_46")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_47");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_48","Padding","same")
        batchNormalizationLayer("Name","bn_48")
        functionLayer(@vision.cnn.mish,"Name","mish_48")
        convolution2dLayer([3 3],128,"Name","conv_49","Padding","same")
        batchNormalizationLayer("Name","bn_49")
        functionLayer(@vision.cnn.mish,"Name","mish_49")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_50");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_51","Padding","same")
        batchNormalizationLayer("Name","bn_51")
        functionLayer(@vision.cnn.mish,"Name","mish_51")
        convolution2dLayer([3 3],128,"Name","conv_52","Padding","same")
        batchNormalizationLayer("Name","bn_52")
        functionLayer(@vision.cnn.mish,"Name","mish_52")];
    net = addLayers(net,tempNet);

    tempNet = [
        additionLayer(2,"Name","add_53")
        convolution2dLayer([1 1],128,"Name","conv_54","Padding","same")
        batchNormalizationLayer("Name","bn_54")
        functionLayer(@vision.cnn.mish,"Name","mish_54")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_55")
        convolution2dLayer([1 1],256,"Name","conv_56","Padding","same")
        batchNormalizationLayer("Name","bn_56")
        functionLayer(@vision.cnn.mish,"Name","mish_56")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([3 3],512,"Name","conv_57","Padding",[1 1 1 1],"Stride",[2 2])
        batchNormalizationLayer("Name","bn_57")
        functionLayer(@vision.cnn.mish,"Name","mish_57")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_58","Padding","same")
        batchNormalizationLayer("Name","bn_58")
        functionLayer(@vision.cnn.mish,"Name","mish_58")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_60","Padding","same")
        batchNormalizationLayer("Name","bn_60")
        functionLayer(@vision.cnn.mish,"Name","mish_60")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_61","Padding","same")
        batchNormalizationLayer("Name","bn_61")
        functionLayer(@vision.cnn.mish,"Name","mish_61")
        convolution2dLayer([3 3],256,"Name","conv_62","Padding","same")
        batchNormalizationLayer("Name","bn_62")
        functionLayer(@vision.cnn.mish,"Name","mish_62")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_63");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_64","Padding","same")
        batchNormalizationLayer("Name","bn_64")
        functionLayer(@vision.cnn.mish,"Name","mish_64")
        convolution2dLayer([3 3],256,"Name","conv_65","Padding","same")
        batchNormalizationLayer("Name","bn_65")
        functionLayer(@vision.cnn.mish,"Name","mish_65")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_66");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_67","Padding","same")
        batchNormalizationLayer("Name","bn_67")
        functionLayer(@vision.cnn.mish,"Name","mish_67")
        convolution2dLayer([3 3],256,"Name","conv_68","Padding","same")
        batchNormalizationLayer("Name","bn_68")
        functionLayer(@vision.cnn.mish,"Name","mish_68")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_69");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_70","Padding","same")
        batchNormalizationLayer("Name","bn_70")
        functionLayer(@vision.cnn.mish,"Name","mish_70")
        convolution2dLayer([3 3],256,"Name","conv_71","Padding","same")
        batchNormalizationLayer("Name","bn_71")
        functionLayer(@vision.cnn.mish,"Name","mish_71")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_72");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_73","Padding","same")
        batchNormalizationLayer("Name","bn_73")
        functionLayer(@vision.cnn.mish,"Name","mish_73")
        convolution2dLayer([3 3],256,"Name","conv_74","Padding","same")
        batchNormalizationLayer("Name","bn_74")
        functionLayer(@vision.cnn.mish,"Name","mish_74")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_75");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_76","Padding","same")
        batchNormalizationLayer("Name","bn_76")
        functionLayer(@vision.cnn.mish,"Name","mish_76")
        convolution2dLayer([3 3],256,"Name","conv_77","Padding","same")
        batchNormalizationLayer("Name","bn_77")
        functionLayer(@vision.cnn.mish,"Name","mish_77")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_78");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_79","Padding","same")
        batchNormalizationLayer("Name","bn_79")
        functionLayer(@vision.cnn.mish,"Name","mish_79")
        convolution2dLayer([3 3],256,"Name","conv_80","Padding","same")
        batchNormalizationLayer("Name","bn_80")
        functionLayer(@vision.cnn.mish,"Name","mish_80")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_81");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_82","Padding","same")
        batchNormalizationLayer("Name","bn_82")
        functionLayer(@vision.cnn.mish,"Name","mish_82")
        convolution2dLayer([3 3],256,"Name","conv_83","Padding","same")
        batchNormalizationLayer("Name","bn_83")
        functionLayer(@vision.cnn.mish,"Name","mish_83")];
    net = addLayers(net,tempNet);

    tempNet = [
        additionLayer(2,"Name","add_84")
        convolution2dLayer([1 1],256,"Name","conv_85","Padding","same")
        batchNormalizationLayer("Name","bn_85")
        functionLayer(@vision.cnn.mish,"Name","mish_85")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_86")
        convolution2dLayer([1 1],512,"Name","conv_87","Padding","same")
        batchNormalizationLayer("Name","bn_87")
        functionLayer(@vision.cnn.mish,"Name","mish_87")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([3 3],1024,"Name","conv_88","Padding",[1 1 1 1],"Stride",[2 2])
        batchNormalizationLayer("Name","bn_88")
        functionLayer(@vision.cnn.mish,"Name","mish_88")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],512,"Name","conv_89","Padding","same")
        batchNormalizationLayer("Name","bn_89")
        functionLayer(@vision.cnn.mish,"Name","mish_89")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],512,"Name","conv_91","Padding","same")
        batchNormalizationLayer("Name","bn_91")
        functionLayer(@vision.cnn.mish,"Name","mish_91")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],512,"Name","conv_92","Padding","same")
        batchNormalizationLayer("Name","bn_92")
        functionLayer(@vision.cnn.mish,"Name","mish_92")
        convolution2dLayer([3 3],512,"Name","conv_93","Padding","same")
        batchNormalizationLayer("Name","bn_93")
        functionLayer(@vision.cnn.mish,"Name","mish_93")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_94");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],512,"Name","conv_95","Padding","same")
        batchNormalizationLayer("Name","bn_95")
        functionLayer(@vision.cnn.mish,"Name","mish_95")
        convolution2dLayer([3 3],512,"Name","conv_96","Padding","same")
        batchNormalizationLayer("Name","bn_96")
        functionLayer(@vision.cnn.mish,"Name","mish_96")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_97");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],512,"Name","conv_98","Padding","same")
        batchNormalizationLayer("Name","bn_98")
        functionLayer(@vision.cnn.mish,"Name","mish_98")
        convolution2dLayer([3 3],512,"Name","conv_99","Padding","same")
        batchNormalizationLayer("Name","bn_99")
        functionLayer(@vision.cnn.mish,"Name","mish_99")];
    net = addLayers(net,tempNet);

    tempNet = additionLayer(2,"Name","add_100");
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],512,"Name","conv_101","Padding","same")
        batchNormalizationLayer("Name","bn_101")
        functionLayer(@vision.cnn.mish,"Name","mish_101")
        convolution2dLayer([3 3],512,"Name","conv_102","Padding","same")
        batchNormalizationLayer("Name","bn_102")
        functionLayer(@vision.cnn.mish,"Name","mish_102")];
    net = addLayers(net,tempNet);

    tempNet = [
        additionLayer(2,"Name","add_103")
        convolution2dLayer([1 1],512,"Name","conv_104","Padding","same")
        batchNormalizationLayer("Name","bn_104")
        functionLayer(@vision.cnn.mish,"Name","mish_104")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_105")
        convolution2dLayer([1 1],1024,"Name","conv_106","Padding","same")
        batchNormalizationLayer("Name","bn_106")
        functionLayer(@vision.cnn.mish,"Name","mish_106")
        convolution2dLayer([1 1],512,"Name","conv_107","Padding","same")
        batchNormalizationLayer("Name","bn_107")
        leakyReluLayer(0.1,"Name","leaky_107")
        convolution2dLayer([3 3],1024,"Name","conv_108","Padding","same")
        batchNormalizationLayer("Name","bn_108")
        leakyReluLayer(0.1,"Name","leaky_108")
        convolution2dLayer([1 1],512,"Name","conv_109","Padding","same")
        batchNormalizationLayer("Name","bn_109")
        leakyReluLayer(0.1,"Name","leaky_109")];
    net = addLayers(net,tempNet);

    tempNet = maxPooling2dLayer([5 5],"Name","maxPool_110","Padding","same");
    net = addLayers(net,tempNet);

    tempNet = maxPooling2dLayer([9 9],"Name","maxPool_112","Padding","same");
    net = addLayers(net,tempNet);

    tempNet = maxPooling2dLayer([13 13],"Name","maxPool_114","Padding","same");
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(4,"Name","concat_115")
        convolution2dLayer([1 1],512,"Name","conv_116","Padding","same")
        batchNormalizationLayer("Name","bn_116")
        leakyReluLayer(0.1,"Name","leaky_116")
        convolution2dLayer([3 3],1024,"Name","conv_117","Padding","same")
        batchNormalizationLayer("Name","bn_117")
        leakyReluLayer(0.1,"Name","leaky_117")
        convolution2dLayer([1 1],512,"Name","conv_118","Padding","same")
        batchNormalizationLayer("Name","bn_118")
        leakyReluLayer(0.1,"Name","leaky_118")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_119","Padding","same")
        batchNormalizationLayer("Name","bn_119")
        leakyReluLayer(0.1,"Name","leaky_119")
        resize2dLayer("Name","up2d_120_new","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],256,"Name","conv_122","Padding","same")
        batchNormalizationLayer("Name","bn_122")
        leakyReluLayer(0.1,"Name","leaky_122")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_123")
        convolution2dLayer([1 1],256,"Name","conv_124","Padding","same")
        batchNormalizationLayer("Name","bn_124")
        leakyReluLayer(0.1,"Name","leaky_124")
        convolution2dLayer([3 3],512,"Name","conv_125","Padding","same")
        batchNormalizationLayer("Name","bn_125")
        leakyReluLayer(0.1,"Name","leaky_125")
        convolution2dLayer([1 1],256,"Name","conv_126","Padding","same")
        batchNormalizationLayer("Name","bn_126")
        leakyReluLayer(0.1,"Name","leaky_126")
        convolution2dLayer([3 3],512,"Name","conv_127","Padding","same")
        batchNormalizationLayer("Name","bn_127")
        leakyReluLayer(0.1,"Name","leaky_127")
        convolution2dLayer([1 1],256,"Name","conv_128","Padding","same")
        batchNormalizationLayer("Name","bn_128")
        leakyReluLayer(0.1,"Name","leaky_128")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_129","Padding","same")
        batchNormalizationLayer("Name","bn_129")
        leakyReluLayer(0.1,"Name","leaky_129")
        resize2dLayer("Name","up2d_130_new","GeometricTransformMode","half-pixel","Method","bilinear","NearestRoundingMode","round","Scale",[2 2])];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([1 1],128,"Name","conv_132","Padding","same")
        batchNormalizationLayer("Name","bn_132")
        leakyReluLayer(0.1,"Name","leaky_132")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_133")
        convolution2dLayer([1 1],128,"Name","conv_134","Padding","same")
        batchNormalizationLayer("Name","bn_134")
        leakyReluLayer(0.1,"Name","leaky_134")
        convolution2dLayer([3 3],256,"Name","conv_135","Padding","same")
        batchNormalizationLayer("Name","bn_135")
        leakyReluLayer(0.1,"Name","leaky_135")
        convolution2dLayer([1 1],128,"Name","conv_136","Padding","same")
        batchNormalizationLayer("Name","bn_136")
        leakyReluLayer(0.1,"Name","leaky_136")
        convolution2dLayer([3 3],256,"Name","conv_137","Padding","same")
        batchNormalizationLayer("Name","bn_137")
        leakyReluLayer(0.1,"Name","leaky_137")
        convolution2dLayer([1 1],128,"Name","conv_138","Padding","same")
        batchNormalizationLayer("Name","bn_138")
        leakyReluLayer(0.1,"Name","leaky_138")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([3 3],256,"Name","conv_139","Padding","same")
        batchNormalizationLayer("Name","bn_139")
        leakyReluLayer(0.1,"Name","leaky_139")
        convolution2dLayer([1 1],18,"Name","convOut1","Padding","same")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([3 3],256,"Name","conv_143","Padding",[1 1 1 1],"Stride",[2 2])
        batchNormalizationLayer("Name","bn_143")
        leakyReluLayer(0.1,"Name","leaky_143")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_144")
        convolution2dLayer([1 1],256,"Name","conv_145","Padding","same")
        batchNormalizationLayer("Name","bn_145")
        leakyReluLayer(0.1,"Name","leaky_145")
        convolution2dLayer([3 3],512,"Name","conv_146","Padding","same")
        batchNormalizationLayer("Name","bn_146")
        leakyReluLayer(0.1,"Name","leaky_146")
        convolution2dLayer([1 1],256,"Name","conv_147","Padding","same")
        batchNormalizationLayer("Name","bn_147")
        leakyReluLayer(0.1,"Name","leaky_147")
        convolution2dLayer([3 3],512,"Name","conv_148","Padding","same")
        batchNormalizationLayer("Name","bn_148")
        leakyReluLayer(0.1,"Name","leaky_148")
        convolution2dLayer([1 1],256,"Name","conv_149","Padding","same")
        batchNormalizationLayer("Name","bn_149")
        leakyReluLayer(0.1,"Name","leaky_149")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([3 3],512,"Name","conv_150","Padding","same")
        batchNormalizationLayer("Name","bn_150")
        leakyReluLayer(0.1,"Name","leaky_150")
        convolution2dLayer([1 1],18,"Name","convOut2","Padding","same")];
    net = addLayers(net,tempNet);

    tempNet = [
        convolution2dLayer([3 3],512,"Name","conv_154","Padding",[1 1 1 1],"Stride",[2 2])
        batchNormalizationLayer("Name","bn_154")
        leakyReluLayer(0.1,"Name","leaky_154")];
    net = addLayers(net,tempNet);

    tempNet = [
        depthConcatenationLayer(2,"Name","concat_155")
        convolution2dLayer([1 1],512,"Name","conv_156","Padding","same")
        batchNormalizationLayer("Name","bn_156")
        leakyReluLayer(0.1,"Name","leaky_156")
        convolution2dLayer([3 3],1024,"Name","conv_157","Padding","same")
        batchNormalizationLayer("Name","bn_157")
        leakyReluLayer(0.1,"Name","leaky_157")
        convolution2dLayer([1 1],512,"Name","conv_158","Padding","same")
        batchNormalizationLayer("Name","bn_158")
        leakyReluLayer(0.1,"Name","leaky_158")
        convolution2dLayer([3 3],1024,"Name","conv_159","Padding","same")
        batchNormalizationLayer("Name","bn_159")
        leakyReluLayer(0.1,"Name","leaky_159")
        convolution2dLayer([1 1],512,"Name","conv_160","Padding","same")
        batchNormalizationLayer("Name","bn_160")
        leakyReluLayer(0.1,"Name","leaky_160")
        convolution2dLayer([3 3],1024,"Name","conv_161","Padding","same")
        batchNormalizationLayer("Name","bn_161")
        leakyReluLayer(0.1,"Name","leaky_161")
        convolution2dLayer([1 1],18,"Name","convOut3","Padding","same")];
    net = addLayers(net,tempNet);

    clear tempNet;

    net = connectLayers(net,"mish_3","conv_4");
    net = connectLayers(net,"mish_3","conv_6");
    net = connectLayers(net,"mish_4","concat_11/in2");
    net = connectLayers(net,"mish_6","conv_7");
    net = connectLayers(net,"mish_6","add_9/in2");
    net = connectLayers(net,"mish_8","add_9/in1");
    net = connectLayers(net,"mish_10","concat_11/in1");
    net = connectLayers(net,"mish_13","conv_14");
    net = connectLayers(net,"mish_13","conv_16");
    net = connectLayers(net,"mish_14","concat_24/in2");
    net = connectLayers(net,"mish_16","conv_17");
    net = connectLayers(net,"mish_16","add_19/in2");
    net = connectLayers(net,"mish_18","add_19/in1");
    net = connectLayers(net,"add_19","conv_20");
    net = connectLayers(net,"add_19","add_22/in2");
    net = connectLayers(net,"mish_21","add_22/in1");
    net = connectLayers(net,"mish_23","concat_24/in1");
    net = connectLayers(net,"mish_26","conv_27");
    net = connectLayers(net,"mish_26","conv_29");
    net = connectLayers(net,"mish_27","concat_55/in2");
    net = connectLayers(net,"mish_29","conv_30");
    net = connectLayers(net,"mish_29","add_32/in2");
    net = connectLayers(net,"mish_31","add_32/in1");
    net = connectLayers(net,"add_32","conv_33");
    net = connectLayers(net,"add_32","add_35/in2");
    net = connectLayers(net,"mish_34","add_35/in1");
    net = connectLayers(net,"add_35","conv_36");
    net = connectLayers(net,"add_35","add_38/in2");
    net = connectLayers(net,"mish_37","add_38/in1");
    net = connectLayers(net,"add_38","conv_39");
    net = connectLayers(net,"add_38","add_41/in2");
    net = connectLayers(net,"mish_40","add_41/in1");
    net = connectLayers(net,"add_41","conv_42");
    net = connectLayers(net,"add_41","add_44/in2");
    net = connectLayers(net,"mish_43","add_44/in1");
    net = connectLayers(net,"add_44","conv_45");
    net = connectLayers(net,"add_44","add_47/in2");
    net = connectLayers(net,"mish_46","add_47/in1");
    net = connectLayers(net,"add_47","conv_48");
    net = connectLayers(net,"add_47","add_50/in2");
    net = connectLayers(net,"mish_49","add_50/in1");
    net = connectLayers(net,"add_50","conv_51");
    net = connectLayers(net,"add_50","add_53/in2");
    net = connectLayers(net,"mish_52","add_53/in1");
    net = connectLayers(net,"mish_54","concat_55/in1");
    net = connectLayers(net,"mish_56","conv_57");
    net = connectLayers(net,"mish_56","conv_132");
    net = connectLayers(net,"mish_57","conv_58");
    net = connectLayers(net,"mish_57","conv_60");
    net = connectLayers(net,"mish_58","concat_86/in2");
    net = connectLayers(net,"mish_60","conv_61");
    net = connectLayers(net,"mish_60","add_63/in2");
    net = connectLayers(net,"mish_62","add_63/in1");
    net = connectLayers(net,"add_63","conv_64");
    net = connectLayers(net,"add_63","add_66/in2");
    net = connectLayers(net,"mish_65","add_66/in1");
    net = connectLayers(net,"add_66","conv_67");
    net = connectLayers(net,"add_66","add_69/in2");
    net = connectLayers(net,"mish_68","add_69/in1");
    net = connectLayers(net,"add_69","conv_70");
    net = connectLayers(net,"add_69","add_72/in2");
    net = connectLayers(net,"mish_71","add_72/in1");
    net = connectLayers(net,"add_72","conv_73");
    net = connectLayers(net,"add_72","add_75/in2");
    net = connectLayers(net,"mish_74","add_75/in1");
    net = connectLayers(net,"add_75","conv_76");
    net = connectLayers(net,"add_75","add_78/in2");
    net = connectLayers(net,"mish_77","add_78/in1");
    net = connectLayers(net,"add_78","conv_79");
    net = connectLayers(net,"add_78","add_81/in2");
    net = connectLayers(net,"mish_80","add_81/in1");
    net = connectLayers(net,"add_81","conv_82");
    net = connectLayers(net,"add_81","add_84/in2");
    net = connectLayers(net,"mish_83","add_84/in1");
    net = connectLayers(net,"mish_85","concat_86/in1");
    net = connectLayers(net,"mish_87","conv_88");
    net = connectLayers(net,"mish_87","conv_122");
    net = connectLayers(net,"mish_88","conv_89");
    net = connectLayers(net,"mish_88","conv_91");
    net = connectLayers(net,"mish_89","concat_105/in2");
    net = connectLayers(net,"mish_91","conv_92");
    net = connectLayers(net,"mish_91","add_94/in2");
    net = connectLayers(net,"mish_93","add_94/in1");
    net = connectLayers(net,"add_94","conv_95");
    net = connectLayers(net,"add_94","add_97/in2");
    net = connectLayers(net,"mish_96","add_97/in1");
    net = connectLayers(net,"add_97","conv_98");
    net = connectLayers(net,"add_97","add_100/in2");
    net = connectLayers(net,"mish_99","add_100/in1");
    net = connectLayers(net,"add_100","conv_101");
    net = connectLayers(net,"add_100","add_103/in2");
    net = connectLayers(net,"mish_102","add_103/in1");
    net = connectLayers(net,"mish_104","concat_105/in1");
    net = connectLayers(net,"leaky_109","maxPool_110");
    net = connectLayers(net,"leaky_109","maxPool_112");
    net = connectLayers(net,"leaky_109","maxPool_114");
    net = connectLayers(net,"leaky_109","concat_115/in4");
    net = connectLayers(net,"maxPool_110","concat_115/in3");
    net = connectLayers(net,"maxPool_112","concat_115/in2");
    net = connectLayers(net,"maxPool_114","concat_115/in1");
    net = connectLayers(net,"leaky_118","conv_119");
    net = connectLayers(net,"leaky_118","concat_155/in2");
    net = connectLayers(net,"up2d_120_new","concat_123/in2");
    net = connectLayers(net,"leaky_122","concat_123/in1");
    net = connectLayers(net,"leaky_128","conv_129");
    net = connectLayers(net,"leaky_128","concat_144/in2");
    net = connectLayers(net,"up2d_130_new","concat_133/in2");
    net = connectLayers(net,"leaky_132","concat_133/in1");
    net = connectLayers(net,"leaky_138","conv_139");
    net = connectLayers(net,"leaky_138","conv_143");
    net = connectLayers(net,"leaky_143","concat_144/in1");
    net = connectLayers(net,"leaky_149","conv_150");
    net = connectLayers(net,"leaky_149","conv_154");
    net = connectLayers(net,"leaky_154","concat_155/in1");
    net = initialize(net);

    % ---------------------------------------------------------------------

    if fold==1
        detector = yolov4ObjectDetector(net,className,anchorBoxes,InputSize=inputSize);
    end
    augmentedTrainingData = transform(trainingData,@augmentData);

    % edit by file
    options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="piecewise",...
    LearnRateDropFactor=0.5,...
    LearnRateDropPeriod=20,...
    MiniBatchSize=4,...
    L2Regularization=0.0005,...
    MaxEpochs=50,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationFrequency=1000,...
    CheckpointPath=tempdir,...
    ValidationData=validationData);

    % train
    [detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);
    
end


% Saving Weight File
Fracture_detection_elbow_5 = detector;      % edit by file
save Filename % file name

%Function -----------------------------------------------------------------

function data = augmentData(A)

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.4,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.0);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.3]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii,1:2) = {I,bboxes};
end
end
